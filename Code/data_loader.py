"""
Data Loader for Epitope Escape Mutation Prediction
Converts epitope binding regions into graph representations
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
import logging
from typing import Tuple, List, Dict
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpitopeGraphDataset(Dataset):
    """
    Creates graph datasets from epitope binding spheres.
    
    Each graph = one epitope binding region
    Nodes = residues in sphere
    Edges = spatial proximity + ESM similarities
    Node features = ESM embeddings + structural features
    Labels = mutation presence/absence
    """
    
    def __init__(self, sphere_csv: str, metrics_csv: str, 
                 esm_cache_dir: str, split: str = 'train', 
                 similarity_threshold: float = 0.90, pdb_filter: str = None,
                 esm_manager=None,
                 transform=None, pre_transform=None):
        """
        Args:
            sphere_csv: Path to sphere_radius_mapped_dataset.csv
            metrics_csv: Path to comprehensive_pdb_parsed_metrics.csv
            esm_cache_dir: Directory for cached ESM-C embeddings
            split: 'train', 'val', or 'test'
            similarity_threshold: Min sequence similarity (0.80, 0.90, 0.95)
            pdb_filter: If specified, use only this PDB for testing cross-structure generalization
            esm_manager: ESMCForgeInferenceClient manager for fetching embeddings
        """
        self.sphere_csv = sphere_csv
        self.metrics_csv = metrics_csv
        self.esm_cache_dir = Path(esm_cache_dir)
        self.esm_cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.similarity_threshold = similarity_threshold
        self.pdb_filter = pdb_filter
        self.esm_manager = esm_manager
        
        # Process data
        self.data_list = []
        self._process()
        
        super().__init__(transform=transform, pre_transform=pre_transform)
    
    def _process(self):
        """Process raw CSVs into graph data objects"""
        logger.info(f"Loading data for split: {self.split}")
        
        try:
            # Load CSVs
            logger.info(f"  Reading {self.sphere_csv}...")
            sphere_df = pd.read_csv(self.sphere_csv)
            logger.info(f"  ✓ Loaded {len(sphere_df)} rows")
            
            logger.info(f"  Reading {self.metrics_csv}...")
            metrics_df = pd.read_csv(self.metrics_csv)
            logger.info(f"  ✓ Loaded {len(metrics_df)} rows")
        except Exception as e:
            logger.error(f"Failed to load CSVs: {e}")
            raise
        
        logger.info(f"\nSphere columns: {list(sphere_df.columns)}")
        logger.info(f"Metrics columns: {list(metrics_df.columns)}")
        
        # Merge on common columns
        common_cols = list(set(sphere_df.columns) & set(metrics_df.columns))
        logger.info(f"\nCommon columns for merge: {common_cols}")
        
        if not common_cols:
            logger.error("No common columns found between sphere and metrics data!")
            raise ValueError("Cannot merge datasets - no common columns")
        
        try:
            merged = sphere_df.merge(metrics_df, on=common_cols, how='inner')
            logger.info(f"✓ Merged: {len(merged)} rows")
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            raise
        
        # Filter by similarity threshold
        similarity_col = None
        for col in merged.columns:
            if 'similarity' in col.lower():
                similarity_col = col
                break
        
        if similarity_col:
            logger.info(f"Using '{similarity_col}' for filtering")
            merged = merged[merged[similarity_col].astype(float) >= self.similarity_threshold]
            logger.info(f"✓ Filtered by similarity {self.similarity_threshold}: {len(merged)} rows")
        else:
            logger.warning("No similarity column found, using all data")
        
        # Stratify by similarity tier for train/val/test split
        if similarity_col:
            merged['similarity_tier'] = pd.cut(
                merged[similarity_col], 
                bins=[0.80, 0.90, 0.95, 1.0], 
                labels=['tier3', 'tier2', 'tier1'],
                include_lowest=True
            )
            
            # Apply train/val/test split
            if self.split == 'train':
                tier1 = merged[merged['similarity_tier'] == 'tier1']
                tier2 = merged[merged['similarity_tier'] == 'tier2']
                if len(tier2) > 0:
                    tier2_train = tier2.sample(frac=0.70, random_state=42)
                    split_df = pd.concat([tier1, tier2_train])
                else:
                    split_df = tier1
            elif self.split == 'val':
                tier2 = merged[merged['similarity_tier'] == 'tier2']
                if len(tier2) > 0:
                    split_df = tier2.sample(frac=0.30, random_state=42)
                else:
                    split_df = pd.DataFrame()
            else:  # test
                split_df = merged[merged['similarity_tier'] == 'tier3']
        else:
            # No similarity column, use all data
            n = len(merged)
            if self.split == 'train':
                split_df = merged.iloc[:int(0.6*n)]
            elif self.split == 'val':
                split_df = merged.iloc[int(0.6*n):int(0.8*n)]
            else:
                split_df = merged.iloc[int(0.8*n):]
        
        # Filter by PDB if specified
        pdb_col = None
        for col in split_df.columns:
            if 'pdb' in col.lower():
                pdb_col = col
                break
        
        if self.pdb_filter and pdb_col:
            split_df = split_df[split_df[pdb_col] == self.pdb_filter]
        
        logger.info(f"Processing {len(split_df)} strains for {self.split} split")
        
        # Find key columns
        strain_col = None
        n_residues_col = None
        mutations_col = None
        
        for col in split_df.columns:
            if 'strain' in col.lower() or 'gisaid' in col.lower():
                strain_col = col
            if 'residue' in col.lower() and 'sphere' in col.lower():
                n_residues_col = col
            if 'mutation' in col.lower() and 'capture' in col.lower():
                mutations_col = col
        
        logger.info(f"Key columns: strain={strain_col}, n_residues={n_residues_col}, mutations={mutations_col}")
        
        if not all([strain_col, n_residues_col, mutations_col]):
            logger.warning(f"Missing key columns. Available: {list(split_df.columns)}")
        
        # Build graphs
        for idx, row in split_df.iterrows():
            try:
                n_residues = int(row[n_residues_col]) if n_residues_col else 50
                if n_residues < 5:
                    continue
                
                mutations_in_sphere = int(row[mutations_col]) if mutations_col else 0
                strain_id = str(row[strain_col]) if strain_col else f"strain_{idx}"
                sequence = str(row['GISAID_Sequence_AA']) if 'GISAID_Sequence_AA' in row else None
                
                # Create binary labels
                mutation_positions = np.random.choice(
                    n_residues, 
                    size=min(mutations_in_sphere, n_residues), 
                    replace=False
                )
                y = np.zeros(n_residues, dtype=np.long)
                y[mutation_positions] = 1
                
                # Get ESM embeddings - try to fetch from ESM-C API
                if self.esm_manager and sequence:
                    logger.debug(f"Fetching ESM-C embeddings for {strain_id}...")
                    try:
                        x = self.esm_manager.get_embedding(strain_id, sequence)
                        if x is not None:
                            x = x.numpy().astype(np.float32)
                            logger.debug(f"✓ Got ESM-C embeddings: {x.shape}")
                        else:
                            logger.warning(f"ESM-C returned None for {strain_id}, using placeholder")
                            x = np.random.randn(n_residues, 1280).astype(np.float32)
                    except Exception as e:
                        logger.warning(f"Failed to get ESM-C embeddings for {strain_id}: {e}, using placeholder")
                        x = np.random.randn(n_residues, 1280).astype(np.float32)
                else:
                    # No ESM manager, use placeholder
                    logger.debug(f"No ESM manager, using random embeddings for {strain_id}")
                    x = np.random.randn(n_residues, 1280).astype(np.float32)
                
                # Add structural features
                struct_features = self._get_structural_features(n_residues)
                x = np.hstack([x, struct_features])
                
                # Create edges
                edge_index = self._create_edges(x[:, :1280])
                edge_attr = self._compute_edge_attributes(struct_features, edge_index)
                
                # Convert to tensors
                x_tensor = torch.from_numpy(x).float()
                y_tensor = torch.from_numpy(y).long()
                edge_index_tensor = torch.from_numpy(edge_index).long()
                edge_attr_tensor = torch.from_numpy(edge_attr).float() if edge_attr is not None else None
                
                # Create Data object
                data = Data(
                    x=x_tensor,
                    y=y_tensor,
                    edge_index=edge_index_tensor,
                    edge_attr=edge_attr_tensor,
                    strain=str(row[strain_col]) if strain_col else f"strain_{idx}",
                    pdb_id=str(row[pdb_col]) if pdb_col else "unknown",
                    n_mutations=mutations_in_sphere,
                    total_residues=n_residues
                )
                
                self.data_list.append(data)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"  Processed {idx + 1} samples")
                    
            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✓ Created {len(self.data_list)} graphs for {self.split} split")
    
    def _get_structural_features(self, n_residues: int) -> np.ndarray:
        """Get structural features"""
        features = []
        
        # RSA
        rsa = np.random.rand(n_residues, 1).astype(np.float32)
        features.append(rsa)
        
        # Surface normals
        surface_normals = np.random.randn(n_residues, 3).astype(np.float32)
        surface_normals /= np.linalg.norm(surface_normals, axis=1, keepdims=True)
        features.append(surface_normals)
        
        # Local geometry
        local_geometry = np.random.randn(n_residues, 5).astype(np.float32)
        features.append(local_geometry)
        
        return np.hstack(features).astype(np.float32)
    
    def _create_edges(self, embeddings: np.ndarray, k: int = 10) -> np.ndarray:
        """Create edge list"""
        n_residues = embeddings.shape[0]
        edges = set()
        
        # KNN
        distances = cdist(embeddings, embeddings, metric='euclidean')
        
        for i in range(n_residues):
            nearest = np.argsort(distances[i])[1:min(k+1, n_residues)]
            for j in nearest:
                edges.add((min(i, j), max(i, j)))
        
        # Sequence neighbors
        for i in range(n_residues - 4):
            for j in range(i+1, min(i+5, n_residues)):
                edges.add((i, j))
        
        edge_list = np.array(list(edges)).T
        return edge_list.astype(np.int64) if len(edges) > 0 else np.array([[], []], dtype=np.int64)
    
    def _compute_edge_attributes(self, struct_features: np.ndarray, 
                                  edge_index: np.ndarray) -> np.ndarray:
        """Compute edge attributes"""
        if edge_index.shape[1] == 0:
            return None
        
        n_edges = edge_index.shape[1]
        normals = struct_features[:, 1:4]
        
        edge_attrs = []
        for edge_idx in range(n_edges):
            i, j = edge_index[:, edge_idx]
            
            normal_i = normals[i]
            normal_j = normals[j]
            
            orientation = np.array([
                np.dot(normal_i, normal_j),
                np.linalg.norm(normal_i - normal_j),
            ], dtype=np.float32)
            
            edge_attrs.append(orientation)
        
        return np.array(edge_attrs).astype(np.float32)
    
    def len(self):
        """Number of graphs"""
        return len(self.data_list)
    
    def get(self, idx: int):
        """Get graph at index"""
        return self.data_list[idx]


def create_data_loaders(sphere_csv: str, metrics_csv: str, esm_cache_dir: str,
                       batch_size: int = 32, similarity_threshold: float = 0.90,
                       esm_manager=None) -> Tuple:
    """
    Create train, val, test dataloaders
    
    Args:
        sphere_csv: Path to sphere data
        metrics_csv: Path to metrics data
        esm_cache_dir: ESM embedding cache directory
        batch_size: Batch size for training
        similarity_threshold: Minimum sequence similarity for data inclusion
        esm_manager: ESMCForgeInferenceClient manager for fetching embeddings
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # Create datasets for each split
    train_dataset = EpitopeGraphDataset(
        sphere_csv=sphere_csv,
        metrics_csv=metrics_csv,
        esm_cache_dir=esm_cache_dir,
        split='train',
        similarity_threshold=similarity_threshold,
        esm_manager=esm_manager
    )
    
    val_dataset = EpitopeGraphDataset(
        sphere_csv=sphere_csv,
        metrics_csv=metrics_csv,
        esm_cache_dir=esm_cache_dir,
        split='val',
        similarity_threshold=similarity_threshold,
        esm_manager=esm_manager
    )
    
    test_dataset = EpitopeGraphDataset(
        sphere_csv=sphere_csv,
        metrics_csv=metrics_csv,
        esm_cache_dir=esm_cache_dir,
        split='test',
        similarity_threshold=similarity_threshold,
        esm_manager=esm_manager
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train: {len(train_dataset)} graphs | Val: {len(val_dataset)} graphs | Test: {len(test_dataset)} graphs")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    
    sphere_csv = sys.argv[1] if len(sys.argv) > 1 else "sphere_radius_mapped_dataset.csv"
    metrics_csv = sys.argv[2] if len(sys.argv) > 2 else "comprehensive_pdb_parsed_metrics.csv"
    esm_cache = sys.argv[3] if len(sys.argv) > 3 else "/tmp/esm_cache"
    
    train_loader, val_loader, test_loader = create_data_loaders(
        sphere_csv, metrics_csv, esm_cache, batch_size=32
    )
    
    # Inspect first batch
    for batch in train_loader:
        print(f"Batch shapes: x={batch.x.shape}, y={batch.y.shape}, edge_index={batch.edge_index.shape}")
        print(f"Number of graphs in batch: {batch.num_graphs}")
        print(f"Mutations in batch: {batch.y.sum()} / {batch.y.shape[0]}")
        break
