"""
Deep Learning Integration for Federated Learning
=================================================

This module provides PyTorch and TensorFlow dataset wrappers and model utilities
for training federated learning models on the processed stock market data.

Supports:
- PyTorch Dataset/DataLoader integration
- TensorFlow tf.data.Dataset creation
- Custom federated learning model wrappers
- Training utilities for federated averaging (FedAvg)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle
import json


# ============================================================================
# PYTORCH INTEGRATION
# ============================================================================

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. PyTorch utilities will not be available.")


if PYTORCH_AVAILABLE:
    class StockMarketDataset(Dataset):
        """
        PyTorch Dataset for stock market data.
        
        Loads preprocessed data from CSV files and provides batching functionality.
        """
        
        def __init__(
            self,
            data_dir: Path,
            client_id: int,
            split: str = 'train',
            target_type: str = 'regression',
            symbols: Optional[List[str]] = None
        ):
            """
            Initialize the dataset.
            
            Args:
                data_dir: Path to federated data directory
                client_id: Client ID
                split: Data split ('train', 'val', 'test')
                target_type: Target type ('regression' or 'classification')
                symbols: Specific symbols to load (None = all)
            """
            self.data_dir = Path(data_dir)
            self.client_id = client_id
            self.split = split
            self.target_type = target_type
            
            # Load data
            self.X = []
            self.y = []
            self._load_data(symbols)
            
            self.X = np.vstack(self.X) if self.X else np.array([])
            self.y = np.hstack(self.y) if self.y else np.array([])
            
            # Convert to tensors
            self.X_tensor = torch.FloatTensor(self.X) if len(self.X) > 0 else None
            self.y_tensor = torch.FloatTensor(self.y) if len(self.y) > 0 else None
        
        def _load_data(self, symbols: Optional[List[str]] = None):
            """Load data from CSV files."""
            client_dir = self.data_dir / f"client_{self.client_id:02d}"
            
            if not client_dir.exists():
                raise FileNotFoundError(f"Client directory not found: {client_dir}")
            
            # Find CSV files for this split
            csv_files = list(client_dir.glob(f"*_{self.split}.csv"))
            
            if symbols:
                csv_files = [f for f in csv_files if any(s in f.name for s in symbols)]
            
            for filepath in csv_files:
                df = pd.read_csv(filepath)
                
                # Select features (exclude Date, Symbol, and targets)
                feature_cols = [c for c in df.columns 
                               if c not in {'Date', 'Symbol', 'Next_Return', 'Direction'}]
                X = df[feature_cols].values.astype(np.float32)
                
                # Select target
                if self.target_type == 'regression':
                    y = df['Next_Return'].values.astype(np.float32)
                else:  # classification
                    y = df['Direction'].values.astype(np.float32)
                
                self.X.append(X)
                self.y.append(y)
        
        def __len__(self) -> int:
            return len(self.X_tensor) if self.X_tensor is not None else 0
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X_tensor[idx], self.y_tensor[idx]
    
    
    class FederatedDataLoaderPyTorch:
        """Utility for creating PyTorch DataLoaders for federated clients."""
        
        @staticmethod
        def get_client_loader(
            data_dir: str,
            client_id: int,
            split: str = 'train',
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            target_type: str = 'regression'
        ) -> DataLoader:
            """
            Create a DataLoader for a specific client.
            
            Args:
                data_dir: Path to federated data directory
                client_id: Client ID
                split: Data split
                batch_size: Batch size
                shuffle: Whether to shuffle data
                num_workers: Number of worker processes
                target_type: Target type
                
            Returns:
                PyTorch DataLoader
            """
            dataset = StockMarketDataset(
                data_dir=Path(data_dir),
                client_id=client_id,
                split=split,
                target_type=target_type
            )
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
        
        @staticmethod
        def get_all_client_loaders(
            data_dir: str,
            num_clients: int,
            split: str = 'train',
            batch_size: int = 32,
            shuffle: bool = True
        ) -> Dict[int, DataLoader]:
            """
            Create DataLoaders for all clients.
            
            Args:
                data_dir: Path to federated data directory
                num_clients: Number of clients
                split: Data split
                batch_size: Batch size
                shuffle: Whether to shuffle data
                
            Returns:
                Dictionary mapping client_id to DataLoader
            """
            loaders = {}
            for client_id in range(num_clients):
                loaders[client_id] = FederatedDataLoaderPyTorch.get_client_loader(
                    data_dir=data_dir,
                    client_id=client_id,
                    split=split,
                    batch_size=batch_size,
                    shuffle=shuffle
                )
            return loaders


# ============================================================================
# TENSORFLOW INTEGRATION
# ============================================================================

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. TensorFlow utilities will not be available.")


if TENSORFLOW_AVAILABLE:
    class FederatedDataLoaderTensorFlow:
        """Utility for creating TensorFlow Datasets for federated clients."""
        
        @staticmethod
        def load_client_data(
            data_dir: str,
            client_id: int,
            split: str = 'train',
            target_type: str = 'regression'
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Load data for a client as numpy arrays.
            
            Args:
                data_dir: Path to federated data directory
                client_id: Client ID
                split: Data split
                target_type: Target type
                
            Returns:
                Tuple of (X, y) numpy arrays
            """
            data_dir = Path(data_dir)
            client_dir = data_dir / f"client_{client_id:02d}"
            
            X_list = []
            y_list = []
            
            # Load all CSV files for this split
            csv_files = list(client_dir.glob(f"*_{split}.csv"))
            
            for filepath in csv_files:
                df = pd.read_csv(filepath)
                
                # Select features
                feature_cols = [c for c in df.columns 
                               if c not in {'Date', 'Symbol', 'Next_Return', 'Direction'}]
                X = df[feature_cols].values.astype(np.float32)
                
                # Select target
                if target_type == 'regression':
                    y = df['Next_Return'].values.astype(np.float32)
                else:  # classification
                    y = df['Direction'].values.astype(np.float32)
                
                X_list.append(X)
                y_list.append(y)
            
            X = np.vstack(X_list)
            y = np.hstack(y_list)
            
            return X, y
        
        @staticmethod
        def get_client_dataset(
            data_dir: str,
            client_id: int,
            split: str = 'train',
            batch_size: int = 32,
            shuffle: bool = True,
            target_type: str = 'regression'
        ) -> tf.data.Dataset:
            """
            Create a tf.data.Dataset for a client.
            
            Args:
                data_dir: Path to federated data directory
                client_id: Client ID
                split: Data split
                batch_size: Batch size
                shuffle: Whether to shuffle data
                target_type: Target type
                
            Returns:
                TensorFlow Dataset
            """
            X, y = FederatedDataLoaderTensorFlow.load_client_data(
                data_dir=data_dir,
                client_id=client_id,
                split=split,
                target_type=target_type
            )
            
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            
            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(X))
            
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        @staticmethod
        def get_all_client_datasets(
            data_dir: str,
            num_clients: int,
            split: str = 'train',
            batch_size: int = 32,
            shuffle: bool = True
        ) -> Dict[int, tf.data.Dataset]:
            """
            Create tf.data.Datasets for all clients.
            
            Args:
                data_dir: Path to federated data directory
                num_clients: Number of clients
                split: Data split
                batch_size: Batch size
                shuffle: Whether to shuffle data
                
            Returns:
                Dictionary mapping client_id to Dataset
            """
            datasets = {}
            for client_id in range(num_clients):
                datasets[client_id] = FederatedDataLoaderTensorFlow.get_client_dataset(
                    data_dir=data_dir,
                    client_id=client_id,
                    split=split,
                    batch_size=batch_size,
                    shuffle=shuffle
                )
            return datasets


# ============================================================================
# FEDERATED LEARNING UTILITIES
# ============================================================================

class FederatedAveragingUtils:
    """Utilities for Federated Averaging (FedAvg) algorithm."""
    
    @staticmethod
    def aggregate_models(model_weights: List[Dict], client_weights: Optional[List[float]] = None) -> Dict:
        """
        Aggregate model weights from multiple clients (FedAvg).
        
        Args:
            model_weights: List of model weight dictionaries from each client
            client_weights: Optional weights for each client (e.g., based on dataset size)
            
        Returns:
            Aggregated weights dictionary
        """
        if client_weights is None:
            # Uniform weighting
            client_weights = [1.0 / len(model_weights)] * len(model_weights)
        else:
            # Normalize weights
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]
        
        aggregated = {}
        
        for param_name in model_weights[0].keys():
            aggregated[param_name] = None
            
            for client_id, weight in enumerate(client_weights):
                if aggregated[param_name] is None:
                    aggregated[param_name] = model_weights[client_id][param_name] * weight
                else:
                    aggregated[param_name] += model_weights[client_id][param_name] * weight
        
        return aggregated
    
    @staticmethod
    def get_client_dataset_sizes(data_dir: str, num_clients: int, split: str = 'train') -> List[int]:
        """
        Get dataset size for each client.
        
        Useful for weighting clients in FedAvg based on data size.
        
        Args:
            data_dir: Path to federated data directory
            num_clients: Number of clients
            split: Data split
            
        Returns:
            List of dataset sizes
        """
        sizes = []
        data_dir = Path(data_dir)
        
        for client_id in range(num_clients):
            client_dir = data_dir / f"client_{client_id:02d}"
            csv_files = list(client_dir.glob(f"*_{split}.csv"))
            
            total_rows = 0
            for filepath in csv_files:
                df = pd.read_csv(filepath)
                total_rows += len(df)
            
            sizes.append(total_rows)
        
        return sizes


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    data_dir = "./federated_data"
    
    # PyTorch Example
    if PYTORCH_AVAILABLE:
        print("PyTorch Example:")
        loader = FederatedDataLoaderPyTorch.get_client_loader(
            data_dir=data_dir,
            client_id=0,
            split='train',
            batch_size=32
        )
        
        print(f"DataLoader created with {len(loader)} batches")
        
        for X_batch, y_batch in loader:
            print(f"Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
            break
    
    # TensorFlow Example
    if TENSORFLOW_AVAILABLE:
        print("\nTensorFlow Example:")
        dataset = FederatedDataLoaderTensorFlow.get_client_dataset(
            data_dir=data_dir,
            client_id=0,
            split='train',
            batch_size=32
        )
        
        for X_batch, y_batch in dataset.take(1):
            print(f"Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
    
    # Federated Averaging Example
    print("\nFederated Averaging Example:")
    sizes = FederatedAveragingUtils.get_client_dataset_sizes(data_dir, num_clients=10)
    print(f"Client dataset sizes: {sizes}")
    print(f"Total data: {sum(sizes)} samples")
