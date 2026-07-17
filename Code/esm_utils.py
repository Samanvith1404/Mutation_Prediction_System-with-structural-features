"""
ESM-C Embedding Utility with Official ESM SDK
Uses ESMCForgeInferenceClient from esm.sdk.forge for efficient inference
Handles embedding generation, caching, and batch processing
"""

import numpy as np
import torch
from pathlib import Path
import logging
from typing import Optional, Dict, List
import json
from tqdm import tqdm
import hashlib

# Official ESM SDK
try:
    from esm.sdk.forge import ESMCForgeInferenceClient
    ESM_SDK_AVAILABLE = True
    logger_init = logging.getLogger("esm.sdk")
    logger_init.setLevel(logging.WARNING)
except ImportError:
    ESM_SDK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESMCClient:
    """
    Client for ESM-C (Evolutionary Scale Modeling - Contact) embeddings
    Uses official ESMCForgeInferenceClient from esm.sdk.forge
    Supports BioHub.ai inference endpoint with intelligent caching
    """
    
    # BioHub ESM-C endpoint
    BIOHUB_URL = "https://biohub.ai"
    
    # Available ESM-C models
    MODELS = {
        'esmc-6b-2024-12': 'esmc-6b-2024-12',  # Latest recommended model
        'esmc-2b-2024-12': 'esmc-2b-2024-12',
        'esmc-300m-2024-12': 'esmc-300m-2024-12',
    }
    
    def __init__(self, api_key: str, cache_dir: str = '/tmp/esm_cache', 
                 model: str = 'esmc-6b-2024-12'):
        """
        Initialize ESM-C client using official SDK
        
        Args:
            api_key: BioHub.ai API token
            cache_dir: Directory to cache embeddings locally
            model: ESM-C model variant
        """
        if not ESM_SDK_AVAILABLE:
            raise ImportError(
                "ESM SDK not installed. Install with:\n"
                "  pip install 'esmfold[esmfold]' or pip install fm-core"
            )
        
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate model
        if model not in self.MODELS:
            logger.warning(f"Unknown model {model}, using esmc-6b-2024-12")
            self.model = 'esmc-6b-2024-12'
        else:
            self.model = model
        
        # Initialize ESMCForgeInferenceClient
        logger.info(f"Initializing ESMCForgeInferenceClient")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  URL: {self.BIOHUB_URL}")
        logger.info(f"  Cache: {cache_dir}")
        
        try:
            self.client = ESMCForgeInferenceClient(
                model=self.model,
                url=self.BIOHUB_URL,
                token=api_key
            )
            logger.info("✓ ESM-C client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ESM-C client: {e}")
            raise
        
        # Cache manifest
        self.manifest_file = self.cache_dir / 'manifest.json'
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        """Load cache manifest"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'embeddings': {}, 'failed': {}}
    
    def _save_manifest(self):
        """Save cache manifest"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _get_cache_path(self, sequence_hash: str) -> Path:
        """Get cache file path for sequence hash"""
        return self.cache_dir / f"{sequence_hash}.pt"
    
    def _hash_sequence(self, sequence: str) -> str:
        """Create hash of sequence"""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def get_embedding(self, strain_id: str, sequence: str, 
                     force_recompute: bool = False) -> Optional[torch.Tensor]:
        """
        Get ESM-C embedding for a sequence
        
        Args:
            strain_id: Unique identifier for strain
            sequence: Amino acid sequence
            force_recompute: If True, always fetch from API (skip cache)
            
        Returns:
            torch.Tensor of shape (seq_len, embedding_dim) or None if failed
        """
        seq_hash = self._hash_sequence(sequence)
        
        # Check if in cache
        if not force_recompute and seq_hash in self.manifest['embeddings']:
            cache_path = self._get_cache_path(seq_hash)
            if cache_path.exists():
                try:
                    embedding = torch.load(cache_path)
                    logger.debug(f"✓ Loaded cached embedding for {strain_id}")
                    return embedding
                except Exception as e:
                    logger.warning(f"Failed to load cache for {strain_id}: {e}")
        
        # Check if previously failed
        if seq_hash in self.manifest['failed']:
            logger.warning(f"Previously failed to fetch {strain_id}, skipping")
            return None
        
        # Fetch from API
        logger.info(f"Fetching embedding for {strain_id} ({len(sequence)} residues)...")
        embedding = self._fetch_from_api(sequence, strain_id)
        
        if embedding is not None:
            # Cache it
            cache_path = self._get_cache_path(seq_hash)
            try:
                torch.save(embedding, cache_path)
                self.manifest['embeddings'][seq_hash] = {
                    'strain_id': strain_id,
                    'seq_length': len(sequence),
                    'embedding_dim': embedding.shape[1],
                    'cached': True
                }
                self._save_manifest()
                logger.info(f"✓ Cached embedding for {strain_id} ({embedding.shape})")
            except Exception as e:
                logger.error(f"Failed to cache embedding: {e}")
            
            return embedding
        else:
            # Mark as failed
            self.manifest['failed'][seq_hash] = {
                'strain_id': strain_id,
                'reason': 'API error'
            }
            self._save_manifest()
            return None
    
    def _fetch_from_api(self, sequence: str, strain_id: str) -> Optional[torch.Tensor]:
        """
        Fetch embedding from ESM-C API using official SDK
        
        Args:
            sequence: Amino acid sequence
            strain_id: For logging
            
        Returns:
            Embedding tensor (seq_len, embedding_dim) or None
        """
        try:
            # ESM-C API workflow:
            # 1. Create ESMProtein object from sequence
            # 2. Call client.encode() to get ESMProteinTensor (structure)
            # 3. Call client.logits() with return_embeddings=True to get embeddings
            
            logger.info(f"Encoding sequence for {strain_id} ({len(sequence)} residues)...")
            
            # Import required classes
            from esm.sdk.api import ESMProtein
            from esm.sdk.api import LogitsConfig
            
            # Create ESMProtein object from sequence
            logger.debug(f"Creating ESMProtein object...")
            protein = ESMProtein(sequence=sequence)
            
            # Encode to get ESMProteinTensor (structure)
            logger.debug(f"Calling client.encode()...")
            protein_tensor = self.client.encode(protein)
            
            # Check for errors
            if hasattr(protein_tensor, 'error'):
                logger.error(f"Encode error: {protein_tensor.error}")
                return None
            
            # Get logits with embeddings enabled
            logger.debug(f"Calling client.logits() with return_embeddings=True...")
            config = LogitsConfig(
                return_embeddings=True,  # ✅ KEY: Get embeddings
                return_hidden_states=False
            )
            
            result = self.client.logits(protein_tensor, config)
            
            # Check for errors
            if hasattr(result, 'error'):
                logger.error(f"Logits error: {result.error}")
                return None
            
            logger.debug(f"Result type: {type(result)}")
            
            # Extract embeddings from LogitsOutput
            embeddings = None
            
            if hasattr(result, 'embeddings'):
                embeddings = result.embeddings
                logger.debug(f"✓ Found result.embeddings: {type(embeddings)}")
            elif hasattr(result, 'mean_embedding'):
                embeddings = result.mean_embedding
                logger.debug(f"✓ Found result.mean_embedding")
            else:
                logger.error(f"Could not find embeddings in LogitsOutput")
                logger.error(f"Result attributes: {[a for a in dir(result) if not a.startswith('_')]}")
                return None
            
            if embeddings is None:
                logger.error("Embeddings is None")
                return None
            
            # Convert to torch tensor
            logger.debug(f"Embeddings type: {type(embeddings)}")
            
            if isinstance(embeddings, torch.Tensor):
                embedding_tensor = embeddings.float()
                logger.debug(f"Already torch tensor")
            elif isinstance(embeddings, np.ndarray):
                embedding_tensor = torch.from_numpy(embeddings).float()
                logger.debug(f"Converted from numpy array")
            else:
                embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
                logger.debug(f"Converted from {type(embeddings)}")
            
            # Verify shape
            if len(embedding_tensor.shape) != 2:
                logger.error(f"Unexpected embedding shape: {embedding_tensor.shape}")
                return None
            
            logger.info(f"✓ Generated embedding for {strain_id}: shape {embedding_tensor.shape}")
            return embedding_tensor
        
        except Exception as e:
            logger.error(f"Failed to fetch embedding for {strain_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def batch_embed(self, strain_sequences: Dict[str, str], 
                   progress_bar: bool = True) -> Dict[str, torch.Tensor]:
        """
        Fetch embeddings for multiple strains
        
        Args:
            strain_sequences: Dict of {strain_id: sequence}
            progress_bar: Show progress bar
            
        Returns:
            Dict of {strain_id: embedding_tensor}
        """
        results = {}
        failed = []
        
        iterator = tqdm(strain_sequences.items(), disable=not progress_bar, 
                       desc='ESM-C Embeddings')
        
        for strain_id, sequence in iterator:
            embedding = self.get_embedding(strain_id, sequence)
            if embedding is not None:
                results[strain_id] = embedding
            else:
                failed.append(strain_id)
        
        logger.info(f"Successfully fetched {len(results)}/{len(strain_sequences)} embeddings")
        if failed:
            logger.warning(f"Failed to fetch {len(failed)} embeddings: {failed[:5]}...")
        
        return results
    
    def get_cache_stats(self) -> dict:
        """Get statistics about cache"""
        cache_files = list(self.cache_dir.glob('*.pt'))
        total_size = sum(f.stat().st_size for f in cache_files) / (1024**3)  # GB
        
        return {
            'total_cached': len(self.manifest['embeddings']),
            'total_failed': len(self.manifest['failed']),
            'cache_size_gb': total_size,
            'cache_dir': str(self.cache_dir),
            'model': self.model,
            'url': self.BIOHUB_URL
        }
    
    def clear_cache(self, older_than_days: int = None):
        """Clear cache files"""
        import time
        
        if older_than_days:
            cutoff_time = time.time() - (older_than_days * 86400)
            for f in self.cache_dir.glob('*.pt'):
                if f.stat().st_mtime < cutoff_time:
                    f.unlink()
                    logger.info(f"Deleted {f.name}")
        else:
            # Clear all
            for f in self.cache_dir.glob('*.pt'):
                f.unlink()
            self.manifest = {'embeddings': {}, 'failed': {}}
            self._save_manifest()
            logger.info("Cache cleared")


class EmbeddingLoader:
    """
    High-level interface for loading ESM-C embeddings
    """
    
    def __init__(self, api_key: str, cache_dir: str = '/tmp/esm_cache',
                 model: str = 'esmc-6b-2024-12'):
        """
        Args:
            api_key: BioHub.ai API token
            cache_dir: Cache directory
            model: ESM-C model
        """
        self.client = ESMCClient(api_key, cache_dir, model)
        self.embedding_dim = 1536  # ESM-C-6b output dimension
    
    def load_for_dataset(self, strain_sequences: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Load embeddings for dataset
        
        Args:
            strain_sequences: {strain_id: sequence}
            
        Returns:
            {strain_id: embedding_tensor}
        """
        return self.client.batch_embed(strain_sequences)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension for model architecture"""
        # ESM-C-6b outputs 1536-dimensional embeddings
        return 1536


if __name__ == "__main__":
    import sys
    
    api_key = sys.argv[1] if len(sys.argv) > 1 else "your_api_key_here"
    
    # Test
    print("Testing ESM-C client...")
    
    try:
        client = ESMCClient(api_key)
        
        test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
        print(f"Testing with: {test_sequence[:50]}...")
        
        embedding = client.get_embedding("test", test_sequence)
        if embedding is not None:
            print(f"✓ Success! Shape: {embedding.shape}")
        else:
            print("✗ Failed")
    except Exception as e:
        print(f"✗ Error: {e}")
