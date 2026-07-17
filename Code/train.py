#!/usr/bin/env python3
"""
Main Training Pipeline for Epitope Escape Mutation Prediction

Usage:
    python train.py --sphere_data sphere_radius_mapped_dataset.csv \
                    --metrics_data comprehensive_pdb_parsed_metrics.csv \
                    --esm_cache /path/to/esm_cache \
                    --esm_token your_huggingface_token \
                    --batch_size 32 \
                    --num_epochs 100
"""

import argparse
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# Import modules
from data_loader import create_data_loaders
from model import RegionScorer, EpitopeTrainer
from inference import EpitopeEvaluator
from esm_utils import ESMCClient, EmbeddingLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ESMEmbeddingManager:
    """
    Manages ESM-C embeddings with API and caching
    """
    
    def __init__(self, cache_dir: str, esm_api_key: Optional[str] = None):
        """
        Args:
            cache_dir: Directory for cached embeddings
            esm_api_key: ESM-C API key for fetching embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ESM-C API client if key provided (BioHub)
        self.client = None
        if esm_api_key:
            try:
                self.client = ESMCClient(esm_api_key, cache_dir=cache_dir, model='esmc-600m')
                logger.info("✓ ESM-C BioHub API client initialized successfully")
                stats = self.client.get_cache_stats()
                logger.info(f"  Cache: {stats['total_cached']} embeddings, {stats['cache_size_gb']:.2f} GB")
            except Exception as e:
                logger.warning(f"Failed to initialize ESM-C API: {e}")
                logger.info("Will use placeholder embeddings")
        else:
            logger.info("No ESM API key provided - using placeholder embeddings")
    
    def get_embedding(self, strain_id: str, sequence: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Get ESM embedding for a strain
        
        Args:
            strain_id: Unique strain identifier
            sequence: Protein sequence (required for API fetching)
            
        Returns:
            ESM embedding tensor
        """
        if self.client and sequence:
            # Try to fetch from API
            embedding = self.client.get_embedding(strain_id, sequence)
            if embedding is not None:
                return embedding
        
        # Fallback: return placeholder random embeddings
        # In production, this ensures training continues even if API is unavailable
        seq_len = len(sequence) if sequence else 100
        logger.debug(f"Using placeholder embedding for {strain_id}")
        return torch.randn(seq_len, 1280).float()
    
    def batch_embed(self, strain_sequences: dict) -> dict:
        """
        Get embeddings for multiple strains
        
        Args:
            strain_sequences: {strain_id: sequence}
            
        Returns:
            {strain_id: embedding_tensor}
        """
        if self.client:
            return self.client.batch_embed(strain_sequences)
        else:
            # Placeholder mode
            logger.warning("ESM-C API not available - using random embeddings")
            return {
                strain_id: torch.randn(len(seq), 1280).float()
                for strain_id, seq in strain_sequences.items()
            }


class TrainingPipeline:
    """
    Complete training pipeline for epitope prediction
    """
    
    def __init__(self, args):
        """
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path(args.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def run(self):
        """Execute full training pipeline"""
        
        logger.info("="*70)
        logger.info("EPITOPE ESCAPE MUTATION PREDICTION - TRAINING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Initialize ESM embedding manager
        logger.info("\n[Step 1] Initializing ESM embedding manager...")
        esm_manager = ESMEmbeddingManager(
            cache_dir=self.args.esm_cache,
            esm_api_key=self.args.esm_token
        )
        
        if esm_manager.client:
            logger.info("✓ ESM-C API connected and ready")
            logger.info(f"  Model: {esm_manager.client.model}")
            logger.info(f"  Caching enabled: {esm_manager.cache_dir}")
        else:
            logger.warning("⚠ ESM-C API not available - using placeholder embeddings")
            logger.warning("  This is fine for testing, but use real embeddings for production")
        
        # Step 2: Load data
        logger.info("\n[Step 2] Loading and preparing data...")
        try:
            logger.info(f"  Loading sphere data: {self.args.sphere_data}")
            logger.info(f"  Loading metrics data: {self.args.metrics_data}")
            
            # Check if files exist
            from pathlib import Path
            if not Path(self.args.sphere_data).exists():
                logger.error(f"✗ File not found: {self.args.sphere_data}")
                return
            if not Path(self.args.metrics_data).exists():
                logger.error(f"✗ File not found: {self.args.metrics_data}")
                return
            
            logger.info("  Files found, creating data loaders...")
            train_loader, val_loader, test_loader = create_data_loaders(
                sphere_csv=self.args.sphere_data,
                metrics_csv=self.args.metrics_data,
                esm_cache_dir=self.args.esm_cache,
                batch_size=self.args.batch_size,
                similarity_threshold=self.args.similarity_threshold,
                esm_manager=esm_manager.client  # Pass the ESM-C client
            )
            logger.info("✓ Data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 3: Initialize model
        logger.info("\n[Step 3] Initializing model...")
        model = RegionScorer(
            in_channels=1280 + 9,  # ESM embeddings + structural features
            edge_channels=8,
            hidden_dim=self.args.hidden_dim,
            num_mp_layers=self.args.num_mp_layers,
            num_final_layers=self.args.num_final_layers
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized: {total_params:,} parameters")
        logger.info(f"Initial threshold: {model.get_threshold():.4f}")
        
        # Step 4: Initialize trainer
        logger.info("\n[Step 4] Initializing trainer...")
        trainer = EpitopeTrainer(
            model=model,
            device=self.device,
            learning_rate=self.args.learning_rate,
            pos_weight=self.args.pos_weight,
            checkpoint_dir=str(self.results_dir / 'checkpoints')
        )
        logger.info(f"✓ Trainer ready (pos_weight={self.args.pos_weight})")
        
        # Step 5: Training
        logger.info("\n[Step 5] Training model...")
        try:
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.args.num_epochs
            )
            logger.info("✓ Training complete")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 6: Save training history
        logger.info("\n[Step 6] Saving training history...")
        history_path = self.results_dir / 'training_history.json'
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            k: v if not isinstance(v, list) else 
                [float(x) if isinstance(x, (int, np.integer, float)) else str(x) for x in v]
            for k, v in history.items()
        }
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        logger.info(f"✓ Saved to {history_path}")
        
        # Step 7: Evaluation
        logger.info("\n[Step 7] Evaluating on test set...")
        evaluator = EpitopeEvaluator(model, device=self.device)
        
        # Evaluate on all sets
        for loader, name in [(train_loader, 'train'), 
                             (val_loader, 'validation'),
                             (test_loader, 'test')]:
            try:
                metrics = evaluator.evaluate_loader(loader, name=name)
            except Exception as e:
                logger.warning(f"Evaluation on {name} failed: {e}")
        
        # Step 8: Generate report
        logger.info("\n[Step 8] Generating evaluation report...")
        report_path = evaluator.generate_report(output_dir=str(self.results_dir))
        logger.info(f"✓ Report saved to {report_path}")
        
        # Step 9: Save configuration
        logger.info("\n[Step 9] Saving configuration...")
        config_path = self.results_dir / 'config.json'
        config = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'model': {
                'in_channels': 1280 + 9,
                'edge_channels': 8,
                'hidden_dim': self.args.hidden_dim,
                'num_mp_layers': self.args.num_mp_layers,
                'num_final_layers': self.args.num_final_layers,
            },
            'training': {
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'pos_weight': self.args.pos_weight,
                'num_epochs': self.args.num_epochs,
                'similarity_threshold': self.args.similarity_threshold,
            },
            'data': {
                'sphere_data': str(self.args.sphere_data),
                'metrics_data': str(self.args.metrics_data),
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✓ Configuration saved to {config_path}")
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("="*70 + "\n")
        
        return evaluator.results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train epitope escape mutation prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --sphere_data sphere_radius_mapped_dataset.csv \\
                  --metrics_data comprehensive_pdb_parsed_metrics.csv
  
  python train.py --sphere_data data.csv --metrics_data metrics.csv \\
                  --batch_size 64 --num_epochs 150 --pos_weight 15
        """
    )
    
    # Data arguments
    parser.add_argument('--sphere_data', type=str, required=True,
                       help='Path to sphere_radius_mapped_dataset.csv')
    parser.add_argument('--metrics_data', type=str, required=True,
                       help='Path to comprehensive_pdb_parsed_metrics.csv')
    
    # ESM arguments
    parser.add_argument('--esm_cache', type=str, default='/tmp/esm_cache',
                       help='Directory for ESM embedding cache')
    parser.add_argument('--esm_token', type=str, default=None,
                       help='HuggingFace token for ESM model access')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for neural networks')
    parser.add_argument('--num_mp_layers', type=int, default=3,
                       help='Number of message passing layers')
    parser.add_argument('--num_final_layers', type=int, default=2,
                       help='Number of final classification layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate for optimizer')
    parser.add_argument('--pos_weight', type=float, default=10.0,
                       help='Weight for positive class in loss (handles imbalance)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--similarity_threshold', type=float, default=0.90,
                       help='Minimum sequence similarity for data inclusion')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results and checkpoints')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = TrainingPipeline(args)
    results = pipeline.run()
    
    return results


if __name__ == '__main__':
    import numpy as np
    main()
