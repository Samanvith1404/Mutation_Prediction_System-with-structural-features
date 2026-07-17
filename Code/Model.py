"""
EGNN-based Model for Epitope Escape Mutation Prediction
Incorporates dMaSIF-style structural features via message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuralMessagePassing(MessagePassing):
    """
    EGNN-style message passing with structural features
    
    Combines:
    - ESM embeddings (semantic sequence context)
    - dMaSIF-style structural features (surface normals, geometry)
    - Edge attributes (orientation between residue pairs)
    """
    
    def __init__(self, in_channels: int, edge_channels: int, hidden_dim: int = 128, 
                 num_layers: int = 3):
        """
        Args:
            in_channels: Node feature dimension (ESM 1280 + structural features)
            edge_channels: Edge attribute dimension
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of message passing layers
        """
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Message passing layers
        self.message_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        
        for _ in range(num_layers):
            # Message MLP: processes edge features + source/target node features
            message_mlp = nn.Sequential(
                nn.Linear(in_channels * 2 + edge_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_channels)
            )
            self.message_mlps.append(message_mlp)
            
            # Update MLP: processes aggregated messages + node features
            update_mlp = nn.Sequential(
                nn.Linear(in_channels * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_channels)
            )
            self.update_mlps.append(update_mlp)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through message passing layers
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_channels)
            
        Returns:
            Updated node embeddings (num_nodes, in_channels)
        """
        for layer_idx in range(self.num_layers):
            message_mlp = self.message_mlps[layer_idx]
            update_mlp = self.update_mlps[layer_idx]
            
            # Message passing
            aggregated = self.propagate(
                edge_index, 
                x=x, 
                edge_attr=edge_attr,
                message_mlp=message_mlp
            )
            
            # Update node features
            x_combined = torch.cat([x, aggregated], dim=1)
            x = update_mlp(x_combined)
            x = F.relu(x)
        
        return x
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: torch.Tensor, message_mlp: nn.Module) -> torch.Tensor:
        """
        Construct messages: concatenate edge info + node features
        """
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return message_mlp(msg_input)


class RegionScorer(nn.Module):
    """
    Graph Neural Network for epitope escape mutation prediction
    
    Architecture:
    1. Structural message passing (EGNN-style)
    2. Per-node feature aggregation
    3. Learnable threshold for mutation probability
    """
    
    def __init__(self, in_channels: int, edge_channels: int, hidden_dim: int = 128,
                 num_mp_layers: int = 3, num_final_layers: int = 2):
        """
        Args:
            in_channels: Node feature dimension
            edge_channels: Edge attribute dimension
            hidden_dim: Hidden dimension
            num_mp_layers: Number of message passing layers
            num_final_layers: Number of final classification layers
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Message passing module
        self.mp = StructuralMessagePassing(
            in_channels=in_channels,
            edge_channels=edge_channels,
            hidden_dim=hidden_dim,
            num_layers=num_mp_layers
        )
        
        # Final classification head (per-residue prediction)
        final_layers = []
        for i in range(num_final_layers):
            if i == 0:
                final_layers.append(nn.Linear(in_channels, hidden_dim))
            else:
                final_layers.append(nn.Linear(hidden_dim, hidden_dim))
            final_layers.append(nn.ReLU())
            final_layers.append(nn.Dropout(0.1))
        
        final_layers.append(nn.Linear(hidden_dim, 1))  # Output: mutation probability
        self.classifier = nn.Sequential(*final_layers)
        
        # Learnable threshold for mutation prediction
        # (initialized at 0.5, will be learned during training)
        self.threshold = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric batch
            
        Returns:
            (logits, predictions at threshold)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if data.edge_attr is not None else torch.zeros(edge_index.shape[1], 8, device=x.device)
        
        # Message passing with structural features
        x = self.mp(x, edge_index, edge_attr)
        
        # Per-node classification
        logits = self.classifier(x)
        logits = logits.squeeze(-1)  # (num_nodes,)
        
        # Compute predictions using learnable threshold
        probs = torch.sigmoid(logits)
        predictions = (probs > self.threshold).float()
        
        return logits, predictions, probs
    
    def get_threshold(self) -> float:
        """Get current threshold value"""
        return self.threshold.item()
    
    def set_threshold(self, value: float):
        """Set threshold value"""
        self.threshold.data = torch.tensor(value)


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss
    
    Heavily penalizes false negatives on rare mutations
    """
    
    def __init__(self, pos_weight: float = 10.0):
        """
        Args:
            pos_weight: Weight for positive class (mutations)
                      Balances rare mutations vs common non-mutations
        """
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss
        
        Args:
            logits: Model predictions (num_nodes,)
            targets: Binary labels (num_nodes,)
            
        Returns:
            Scalar loss
        """
        # Compute per-element BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        
        # Weight by class
        weights = torch.where(
            targets == 1,
            torch.full_like(targets, self.pos_weight, dtype=torch.float),
            torch.ones_like(targets, dtype=torch.float)
        )
        
        weighted_loss = (bce_loss * weights).mean()
        return weighted_loss


class EpitopeTrainer:
    """
    Trainer for epitope escape mutation prediction model
    """
    
    def __init__(self, model: RegionScorer, device: str = 'cuda', 
                 learning_rate: float = 1e-3, pos_weight: float = 10.0,
                 checkpoint_dir: str = './checkpoints'):
        """
        Args:
            model: RegionScorer model
            device: 'cuda' or 'cpu'
            learning_rate: Optimizer learning rate
            pos_weight: Weight for positive class in loss
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.loss_fn = WeightedBCELoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _, _ = self.model(batch)
            
            # Compute loss
            loss = self.loss_fn(logits, batch.y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch train loss: {avg_loss:.6f}")
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, Dict]:
        """
        Validation pass
        
        Returns:
            (validation loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            logits, predictions, probs = self.model(batch)
            loss = self.loss_fn(logits, batch.y)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        preds_np = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        probs_np = np.concatenate(all_probs)
        
        metrics = self._compute_metrics(preds_np, targets_np, probs_np)
        
        logger.info(f"Val loss: {avg_loss:.6f} | Metrics: {metrics}")
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, num_epochs: int = 100) -> Dict:
        """
        Complete training loop
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of training epochs
            
        Returns:
            Training history dict
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # LR scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
                logger.info(f"✓ New best model saved (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.max_patience}")
            
            if self.patience_counter >= self.max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training complete!")
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"best_model_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'threshold': self.model.get_threshold()
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.model.set_threshold(checkpoint['threshold'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    @staticmethod
    def _compute_metrics(predictions: np.ndarray, targets: np.ndarray, 
                        probs: np.ndarray) -> Dict:
        """
        Compute precision, recall, F1, MCC
        
        Args:
            predictions: Binary predictions (0/1)
            targets: Ground truth labels (0/1)
            probs: Probability predictions (0-1)
            
        Returns:
            Metrics dictionary
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
        
        metrics = {}
        
        try:
            metrics['precision'] = precision_score(targets, predictions, zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, zero_division=0)
            metrics['f1'] = f1_score(targets, predictions, zero_division=0)
            metrics['mcc'] = matthews_corrcoef(targets, predictions)
            
            if len(np.unique(targets)) > 1:
                metrics['auroc'] = roc_auc_score(targets, probs)
            else:
                metrics['auroc'] = 0.5
                
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            metrics = {k: 0.0 for k in ['precision', 'recall', 'f1', 'mcc', 'auroc']}
        
        return metrics


if __name__ == "__main__":
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RegionScorer(
        in_channels=1280 + 9,  # ESM 1280 + structural features
        edge_channels=8,
        hidden_dim=128,
        num_mp_layers=3,
        num_final_layers=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Learnable threshold: {model.get_threshold()}")
    
    trainer = EpitopeTrainer(model, device=device, pos_weight=10.0)
    print("Trainer initialized successfully")
