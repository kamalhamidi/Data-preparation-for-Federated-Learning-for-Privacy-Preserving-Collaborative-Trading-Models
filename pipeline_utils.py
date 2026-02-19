"""
Configuration and utilities for federated learning data pipeline.
Includes data loading utilities, feature definitions, and helper functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_GROUPS = {
    'price_features': [
        'Open', 'High', 'Low', 'Close', 'Adj Close'
    ],
    'volume_features': [
        'Volume', 'Volume_Change', 'Volume_Normalized'
    ],
    'momentum_features': [
        'MA_5', 'MA_20', 'Price_MA5_Ratio', 'Price_MA20_Ratio'
    ],
    'volatility_features': [
        'Volatility'
    ],
    'range_features': [
        'Price_Range', 'Close_Open_Ratio'
    ],
    'returns_features': [
        'Returns'
    ]
}

EXCLUDE_COLS = {'Date', 'Symbol', 'Next_Return', 'Direction'}

# Target features for different learning tasks
TARGET_FEATURES = {
    'regression': 'Next_Return',      # Predict next-day return
    'classification': 'Direction'      # Predict price movement direction
}


# ============================================================================
# DATA LOADER UTILITIES
# ============================================================================

class FederatedDataLoader:
    """
    Utility class for loading federated client data.
    
    Supports loading data in multiple formats and provides batch iteration
    for model training.
    """
    
    def __init__(self, client_data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            client_data_dir: Path to the federated data directory
        """
        self.client_data_dir = Path(client_data_dir)
    
    def load_client_data(
        self,
        client_id: int,
        split: str = 'train',
        symbols: List[str] = None,
        as_array: bool = False
    ) -> Dict[str, pd.DataFrame] or Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a specific client.
        
        Args:
            client_id: Client ID (e.g., 0, 1, 2, ...)
            split: Data split ('train', 'val', 'test')
            symbols: Specific symbols to load (None = all)
            as_array: If True, return X, y arrays; if False, return DataFrames
            
        Returns:
            Dictionary of DataFrames or (X, y) arrays
        """
        client_dir = self.client_data_dir / f"client_{client_id:02d}"
        
        if not client_dir.exists():
            raise FileNotFoundError(f"Client directory not found: {client_dir}")
        
        data = {}
        
        # Find all CSV files for this split
        csv_files = list(client_dir.glob(f"*_{split}.csv"))
        
        if symbols:
            csv_files = [f for f in csv_files if any(s in f.name for s in symbols)]
        
        for filepath in csv_files:
            symbol = filepath.name.split('_')[0]
            df = pd.read_csv(filepath)
            data[symbol] = df
        
        if as_array:
            # Convert to numpy arrays
            return self._convert_to_arrays(data)
        
        return data
    
    def _convert_to_arrays(
        self,
        data: Dict[str, pd.DataFrame],
        target_col: str = 'Next_Return'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataFrame dictionary to feature and target arrays.
        
        Args:
            data: Dictionary of DataFrames
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y) numpy arrays
        """
        X_list = []
        y_list = []
        
        for df in data.values():
            feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
            X_list.append(df[feature_cols].values)
            y_list.append(df[target_col].values)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        return X, y
    
    def create_batches(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create mini-batches for training.
        
        Args:
            X: Feature array
            y: Target array
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Returns:
            List of (X_batch, y_batch) tuples
        """
        n = len(X)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, n, batch_size):
            batch_indices = indices[i:i+batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches


# ============================================================================
# STATISTICS AND ANALYSIS UTILITIES
# ============================================================================

class DataStatistics:
    """Compute and visualize dataset statistics."""
    
    @staticmethod
    def compute_client_stats(client_data_dir: str, client_id: int) -> Dict:
        """
        Compute statistics for a client's dataset.
        
        Args:
            client_data_dir: Path to federated data directory
            client_id: Client ID
            
        Returns:
            Dictionary of statistics
        """
        loader = FederatedDataLoader(client_data_dir)
        train_data = loader.load_client_data(client_id, split='train')
        val_data = loader.load_client_data(client_id, split='val')
        test_data = loader.load_client_data(client_id, split='test')
        
        stats = {
            'num_symbols': len(train_data),
            'symbols': list(train_data.keys()),
            'train_rows': sum(len(df) for df in train_data.values()),
            'val_rows': sum(len(df) for df in val_data.values()),
            'test_rows': sum(len(df) for df in test_data.values()),
        }
        
        # Compute feature statistics
        all_data = pd.concat(train_data.values(), ignore_index=True)
        feature_cols = [c for c in all_data.columns if c not in EXCLUDE_COLS]
        
        stats['feature_stats'] = {
            col: {
                'mean': all_data[col].mean(),
                'std': all_data[col].std(),
                'min': all_data[col].min(),
                'max': all_data[col].max(),
                'missing': all_data[col].isna().sum()
            }
            for col in feature_cols
        }
        
        return stats
    
    @staticmethod
    def print_summary(client_data_dir: str, num_clients: int):
        """Print summary statistics for all clients."""
        print("\n" + "="*70)
        print("FEDERATED LEARNING DATASET SUMMARY")
        print("="*70 + "\n")
        
        for client_id in range(num_clients):
            stats = DataStatistics.compute_client_stats(client_data_dir, client_id)
            print(f"Client {client_id:02d}")
            print(f"  Symbols: {stats['num_symbols']} - {', '.join(stats['symbols'][:3])}...")
            print(f"  Train: {stats['train_rows']:,} | Val: {stats['val_rows']:,} | Test: {stats['test_rows']:,}")
            print()


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

class DataQualityValidator:
    """Validate data quality and identify issues."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Validate a single dataframe.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol name for reporting
            
        Returns:
            Dictionary of validation results
        """
        issues = {
            'symbol': symbol,
            'total_rows': len(df),
            'null_cols': {},
            'duplicate_rows': 0,
            'date_issues': 0,
        }
        
        # Check for null values
        null_counts = df.isnull().sum()
        issues['null_cols'] = {
            col: count for col, count in null_counts.items() if count > 0
        }
        
        # Check for duplicates
        issues['duplicate_rows'] = df.duplicated().sum()
        
        # Check date ordering
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date')
            if not df.equals(df_sorted):
                issues['date_issues'] = "Data not properly sorted by date"
        
        return issues
    
    @staticmethod
    def validate_splits(split_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Validate train/val/test splits.
        
        Args:
            split_data: Dictionary of splits
            
        Returns:
            Validation results
        """
        results = {}
        for split_name, df in split_data.items():
            results[split_name] = DataQualityValidator.validate_dataframe(df, split_name)
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Load and analyze federated data
    client_data_dir = "./federated_data"
    
    # Initialize loader
    loader = FederatedDataLoader(client_data_dir)
    
    # Load data for client 0
    print("Loading data for client 0...")
    client_data = loader.load_client_data(client_id=0, split='train')
    print(f"Loaded {len(client_data)} symbols")
    
    # Convert to arrays
    X, y = loader.load_client_data(client_id=0, split='train', as_array=True)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Create batches
    batches = loader.create_batches(X, y, batch_size=64)
    print(f"Created {len(batches)} batches")
    
    # Print dataset summary
    DataStatistics.print_summary(client_data_dir, num_clients=10)
