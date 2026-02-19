"""
Federated Learning Data Pipeline for Stock Market Data
=======================================================

This module implements a complete data pipeline for preparing stock/ETF market data
for federated learning. It handles:
- Data loading and cleaning
- Feature engineering (returns, moving averages, volatility, volume changes)
- Data normalization and scaling
- Non-IID data distribution across federated clients
- Data serialization and storage

Author: Big Data Project Team
Date: 2026
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class StockDataPipeline:
    """
    Complete pipeline for preparing stock market data for federated learning.
    
    Attributes:
        data_dir (Path): Root directory containing stock/ETF data
        output_dir (Path): Directory for saving processed data
        metadata_file (Path): Path to symbols metadata CSV
        etf_dir (Path): Directory containing ETF CSV files
        stock_dir (Path): Directory containing stock CSV files
        num_clients (int): Number of federated clients to create
        test_split (float): Proportion of data for testing
        val_split (float): Proportion of data for validation
    """
    
    def __init__(
        self,
        data_dir: str = "/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/BIg DATA/Project/Stock market dataset",
        output_dir: str = "./federated_data",
        num_clients: int = 10,
        test_split: float = 0.2,
        val_split: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir: Root directory containing stock market dataset
            output_dir: Directory for saving processed data
            num_clients: Number of federated learning clients
            test_split: Proportion of data reserved for testing
            val_split: Proportion of data reserved for validation
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = self.data_dir / "symbols_valid_meta.csv"
        self.etf_dir = self.data_dir / "etfs"
        self.stock_dir = self.data_dir / "stocks"
        self.num_clients = num_clients
        self.test_split = test_split
        self.val_split = val_split
        self.seed = seed
        
        np.random.seed(seed)
        
        # Create output directories
        self._create_output_dirs()
        
        # Storage for processed data
        self.processed_data = {}  # symbol -> dataframe
        self.client_data = {}     # client_id -> list of (symbol, data)
        self.scalers = {}         # feature -> scaler
        
    def _create_output_dirs(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "clients").mkdir(exist_ok=True)
        (self.output_dir / "scalers").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        print(f"✓ Output directories created at {self.output_dir}")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and parse the symbols metadata file.
        
        Returns:
            DataFrame containing symbol metadata with columns: Symbol, ETF, Security Name
        """
        print("\n[STEP 1] Loading metadata...")
        metadata = pd.read_csv(self.metadata_file)
        
        # Extract relevant columns
        metadata = metadata[['Symbol', 'ETF', 'Security Name']].copy()
        
        print(f"✓ Loaded metadata for {len(metadata)} symbols")
        print(f"  - ETFs: {metadata['ETF'].sum()}")
        print(f"  - Stocks: {(~metadata['ETF']).sum()}")
        
        return metadata
    
    def select_symbols(
        self,
        metadata: pd.DataFrame,
        num_symbols: int = 50,
        etf_ratio: float = 0.3,
        force_symbols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select a subset of symbols for processing.
        
        Args:
            metadata: DataFrame with symbol metadata
            num_symbols: Total number of symbols to select
            etf_ratio: Proportion of selected symbols that should be ETFs
            force_symbols: List of specific symbols to always include
            
        Returns:
            List of selected symbol names
        """
        print("\n[STEP 2] Selecting symbols...")
        
        if force_symbols is None:
            force_symbols = []
        
        # Separate ETFs and stocks
        etfs = metadata[metadata['ETF'] == 'Y']['Symbol'].tolist()
        stocks = metadata[metadata['ETF'] != 'Y']['Symbol'].tolist()
        
        # Calculate number of each type to select
        num_etfs = max(1, int(num_symbols * etf_ratio))
        num_stocks = num_symbols - num_etfs
        
        # Ensure forced symbols are included
        forced_etfs = [s for s in force_symbols if s in etfs]
        forced_stocks = [s for s in force_symbols if s in stocks]
        
        # Sample remaining symbols
        selected_etfs = forced_etfs + list(
            np.random.choice(
                [e for e in etfs if e not in forced_etfs],
                size=max(0, num_etfs - len(forced_etfs)),
                replace=False
            )
        )
        selected_stocks = forced_stocks + list(
            np.random.choice(
                [s for s in stocks if s not in forced_stocks],
                size=max(0, num_stocks - len(forced_stocks)),
                replace=False
            )
        )
        
        selected_symbols = selected_etfs + selected_stocks
        print(f"✓ Selected {len(selected_symbols)} symbols")
        print(f"  - ETFs: {len(selected_etfs)}")
        print(f"  - Stocks: {len(selected_stocks)}")
        print(f"  - Sample symbols: {selected_symbols[:5]}")
        
        return selected_symbols
    
    def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load CSV data for a single symbol.
        
        Args:
            symbol: Stock/ETF ticker symbol
            
        Returns:
            DataFrame with stock data or None if file not found
        """
        # Try ETF directory first
        etf_path = self.etf_dir / f"{symbol}.csv"
        if etf_path.exists():
            return pd.read_csv(etf_path)
        
        # Try stock directory
        stock_path = self.stock_dir / f"{symbol}.csv"
        if stock_path.exists():
            return pd.read_csv(stock_path)
        
        return None
    
    def load_and_clean_data(
        self,
        symbols: List[str],
        min_rows: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Load and clean data for selected symbols.
        
        Cleaning includes:
        - Parse date columns
        - Sort by date
        - Remove duplicates
        - Handle missing values
        - Filter by minimum data points
        
        Args:
            symbols: List of symbols to load
            min_rows: Minimum number of rows required for a symbol
            
        Returns:
            Dictionary mapping symbol to cleaned DataFrame
        """
        print("\n[STEP 3] Loading and cleaning data...")
        
        valid_symbols = {}
        failed_symbols = []
        
        for symbol in tqdm(symbols, desc="Loading symbols"):
            try:
                df = self._load_symbol_data(symbol)
                
                if df is None:
                    failed_symbols.append(symbol)
                    continue
                
                # Parse date column
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Sort by date
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['Date'], keep='first')
                
                # Check minimum rows
                if len(df) < min_rows:
                    failed_symbols.append(f"{symbol} (insufficient data: {len(df)} rows)")
                    continue
                
                # Handle missing values - forward fill then backward fill
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                
                # Remove any remaining rows with NaN
                df = df.dropna()
                
                # Reset index
                df = df.reset_index(drop=True)
                
                valid_symbols[symbol] = df
                
            except Exception as e:
                failed_symbols.append(f"{symbol} (error: {str(e)[:30]})")
        
        print(f"✓ Loaded {len(valid_symbols)} valid symbols")
        if failed_symbols:
            print(f"  - {len(failed_symbols)} symbols failed to load")
            print(f"    Examples: {failed_symbols[:3]}")
        
        return valid_symbols
    
    def engineer_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Engineer features for federated learning.
        
        Features engineered:
        - Daily Returns: (Close - Prev Close) / Prev Close
        - Moving Average (5-day): MA5 = Close.rolling(5).mean()
        - Moving Average (20-day): MA20 = Close.rolling(20).mean()
        - Rolling Volatility: std(returns) over 20 days
        - Volume Change %: (Volume - Prev Volume) / Prev Volume
        - Price Movement Direction: {-1, 0, 1} for down, flat, up
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name (for logging)
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = df.copy()
            
            # Daily returns (target for regression)
            df['Returns'] = df['Close'].pct_change() * 100  # percentage returns
            
            # Moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            
            # Price-to-MA ratios (momentum indicators)
            df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
            df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
            
            # Rolling volatility (20-day standard deviation of returns)
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Volume change percentage
            df['Volume_Change'] = df['Volume'].pct_change() * 100
            
            # Volume normalized by mean
            df['Volume_Normalized'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            # Price range (High - Low) / Open
            df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
            
            # Close-to-Open ratio
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Next day return (target for prediction) - shift backward
            df['Next_Return'] = df['Returns'].shift(-1)
            
            # Price movement direction (target for classification)
            # 1: price goes up, 0: stays same/flat, -1: goes down
            df['Direction'] = np.where(df['Next_Return'] > 0.1, 1,
                                      np.where(df['Next_Return'] < -0.1, -1, 0))
            
            # Drop rows with NaN values introduced by feature engineering
            df = df.dropna().reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"  ✗ Feature engineering failed for {symbol}: {e}")
            return None
    
    def process_symbols(self, valid_symbols: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all valid symbols with feature engineering.
        
        Args:
            valid_symbols: Dictionary of valid symbol DataFrames
            
        Returns:
            Dictionary of processed DataFrames with engineered features
        """
        print("\n[STEP 4] Engineering features...")
        
        processed = {}
        for symbol, df in tqdm(valid_symbols.items(), desc="Processing symbols"):
            engineered_df = self.engineer_features(df, symbol)
            if engineered_df is not None and len(engineered_df) > 0:
                processed[symbol] = engineered_df
        
        print(f"✓ Successfully processed {len(processed)} symbols")
        self.processed_data = processed
        
        return processed
    
    def normalize_features(
        self,
        processed_data: Dict[str, pd.DataFrame],
        scaler_type: str = 'standard'
    ) -> Dict[str, pd.DataFrame]:
        """
        Normalize/scale features for federated learning.
        
        Features normalized (excluding Date and target columns):
        - Normalized: Open, High, Low, Close, Adj Close, Volume
        - Engineered: Returns, MA_5, MA_20, Volatility, Volume_Change, etc.
        
        Args:
            processed_data: Dictionary of processed DataFrames
            scaler_type: Type of scaler - 'standard' or 'minmax'
            
        Returns:
            Dictionary of normalized DataFrames
        """
        print("\n[STEP 5] Normalizing features...")
        
        # Select features to normalize (exclude Date and targets)
        exclude_cols = {'Date', 'Returns', 'Next_Return', 'Direction', 'Symbol'}
        
        # Determine features from first non-empty dataframe
        features_to_normalize = []
        for df in processed_data.values():
            features_to_normalize = [col for col in df.columns if col not in exclude_cols]
            break
        
        # Initialize scalers for each feature
        scaler_class = StandardScaler if scaler_type == 'standard' else MinMaxScaler
        scalers = {feature: scaler_class() for feature in features_to_normalize}
        
        # Aggregate data for fitting scalers (using all data)
        print("  - Fitting scalers on aggregated data...")
        aggregated_data = {}
        for feature in features_to_normalize:
            aggregated_data[feature] = []
        
        for df in processed_data.values():
            for feature in features_to_normalize:
                if feature in df.columns:
                    aggregated_data[feature].extend(df[feature].values)
        
        # Fit scalers
        for feature, values in aggregated_data.items():
            if len(values) > 0:
                scalers[feature].fit(np.array(values).reshape(-1, 1))
        
        # Apply scalers to each symbol
        normalized_data = {}
        for symbol, df in tqdm(processed_data.items(), desc="Normalizing data"):
            df_normalized = df.copy()
            
            for feature in features_to_normalize:
                if feature in df_normalized.columns:
                    df_normalized[feature] = scalers[feature].transform(
                        df_normalized[feature].values.reshape(-1, 1)
                    ).flatten()
            
            normalized_data[symbol] = df_normalized
        
        # Save scalers
        self.scalers = scalers
        self._save_scalers(scalers)
        
        print(f"✓ Normalized {len(normalized_data)} datasets")
        print(f"  - Features normalized: {len(features_to_normalize)}")
        print(f"  - Scaler type: {scaler_type}")
        
        return normalized_data
    
    def _save_scalers(self, scalers: Dict):
        """Save scalers to disk for future inference."""
        scaler_path = self.output_dir / "scalers" / "feature_scalers.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"  - Scalers saved to {scaler_path}")
    
    def split_data_for_clients(
        self,
        normalized_data: Dict[str, pd.DataFrame],
        distribution_type: str = 'non_iid'
    ) -> Dict[int, Dict[str, Tuple[pd.DataFrame, str]]]:
        """
        Distribute data across federated clients.
        
        Distribution strategies:
        - 'iid': Each client gets random samples from all symbols (balanced, unrealistic)
        - 'non_iid': Symbols assigned to specific clients (non-IID, more realistic)
        
        Args:
            normalized_data: Dictionary of normalized DataFrames
            distribution_type: Type of distribution ('iid' or 'non_iid')
            
        Returns:
            Dictionary mapping client_id to their assigned data
        """
        print("\n[STEP 6] Distributing data to federated clients...")
        
        symbols = list(normalized_data.keys())
        
        if distribution_type == 'non_iid':
            # Assign symbols to clients (non-IID distribution)
            # Different clients get different sets of symbols
            np.random.shuffle(symbols)
            symbols_per_client = len(symbols) // self.num_clients
            
            client_data = {}
            for client_id in range(self.num_clients):
                start_idx = client_id * symbols_per_client
                if client_id == self.num_clients - 1:
                    # Last client gets remaining symbols
                    end_idx = len(symbols)
                else:
                    end_idx = (client_id + 1) * symbols_per_client
                
                client_symbols = symbols[start_idx:end_idx]
                client_data[client_id] = {
                    symbol: normalized_data[symbol] for symbol in client_symbols
                }
        
        elif distribution_type == 'iid':
            # IID distribution: each client gets samples from all symbols
            client_data = {i: {} for i in range(self.num_clients)}
            for symbol, df in normalized_data.items():
                # Split by rows and distribute to clients
                n_rows = len(df)
                rows_per_client = n_rows // self.num_clients
                
                for client_id in range(self.num_clients):
                    start_idx = client_id * rows_per_client
                    if client_id == self.num_clients - 1:
                        end_idx = n_rows
                    else:
                        end_idx = (client_id + 1) * rows_per_client
                    
                    client_data[client_id][symbol] = df.iloc[start_idx:end_idx].copy()
        
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        # Print distribution statistics
        for client_id, data in client_data.items():
            num_symbols = len(data)
            total_rows = sum(len(df) for df in data.values())
            print(f"  - Client {client_id}: {num_symbols} symbols, {total_rows} data points")
        
        self.client_data = client_data
        return client_data
    
    def create_train_val_test_splits(
        self,
        client_data: Dict[int, Dict[str, pd.DataFrame]]
    ) -> Dict[int, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Create train/validation/test splits for each client.
        
        Splits are created per symbol to maintain temporal integrity.
        
        Args:
            client_data: Dictionary of client data
            
        Returns:
            Dictionary with structure: client_id -> symbol -> {train/val/test -> DataFrame}
        """
        print("\n[STEP 7] Creating train/validation/test splits...")
        
        splits = {}
        
        for client_id, symbols_data in client_data.items():
            splits[client_id] = {}
            
            for symbol, df in symbols_data.items():
                n = len(df)
                
                # Calculate split indices
                test_idx = int(n * (1 - self.test_split))
                val_idx = int(test_idx * (1 - self.val_split))
                
                # Split data (temporal order preserved)
                train_data = df.iloc[:val_idx].copy()
                val_data = df.iloc[val_idx:test_idx].copy()
                test_data = df.iloc[test_idx:].copy()
                
                splits[client_id][symbol] = {
                    'train': train_data,
                    'val': val_data,
                    'test': test_data
                }
        
        print(f"✓ Created splits for {len(splits)} clients")
        
        return splits
    
    def save_client_data(
        self,
        splits: Dict[int, Dict[str, Dict[str, pd.DataFrame]]],
        save_format: str = 'csv'
    ):
        """
        Save processed data for each federated client.
        
        Supported formats:
        - 'csv': Individual CSV files per symbol per split
        - 'parquet': Efficient columnar format
        - 'pickle': Python pickle format
        - 'numpy': NumPy arrays for deep learning
        
        Args:
            splits: Dictionary of train/val/test splits
            save_format: Format for saving ('csv', 'parquet', 'pickle', 'numpy')
        """
        print(f"\n[STEP 8] Saving client data (format: {save_format})...")
        
        for client_id, symbols_data in tqdm(splits.items(), desc="Saving clients"):
            client_dir = self.output_dir / "clients" / f"client_{client_id:02d}"
            client_dir.mkdir(parents=True, exist_ok=True)
            
            for symbol, split_data in symbols_data.items():
                if save_format == 'csv':
                    for split_name, df in split_data.items():
                        filepath = client_dir / f"{symbol}_{split_name}.csv"
                        df.to_csv(filepath, index=False)
                
                elif save_format == 'parquet':
                    for split_name, df in split_data.items():
                        filepath = client_dir / f"{symbol}_{split_name}.parquet"
                        df.to_parquet(filepath, index=False)
                
                elif save_format == 'pickle':
                    filepath = client_dir / f"{symbol}_splits.pkl"
                    with open(filepath, 'wb') as f:
                        pickle.dump(split_data, f)
                
                elif save_format == 'numpy':
                    # Save as numpy arrays for deep learning frameworks
                    for split_name, df in split_data.items():
                        # Separate features and targets
                        feature_cols = [c for c in df.columns if c not in {'Date', 'Symbol'}]
                        X = df[feature_cols].values
                        y_reg = df['Next_Return'].values
                        y_clf = df['Direction'].values
                        
                        filepath_X = client_dir / f"{symbol}_{split_name}_X.npy"
                        filepath_y_reg = client_dir / f"{symbol}_{split_name}_y_reg.npy"
                        filepath_y_clf = client_dir / f"{symbol}_{split_name}_y_clf.npy"
                        
                        np.save(filepath_X, X)
                        np.save(filepath_y_reg, y_reg)
                        np.save(filepath_y_clf, y_clf)
        
        print(f"✓ Saved data for {len(splits)} clients")
    
    def generate_client_metadata(self, splits: Dict[int, Dict[str, Dict[str, pd.DataFrame]]]):
        """
        Generate metadata file documenting client datasets.
        
        Metadata includes:
        - Client information (ID, number of symbols, data statistics)
        - Feature information (name, scaling method, min/max)
        - Data split information (train/val/test sizes)
        
        Args:
            splits: Dictionary of train/val/test splits
        """
        print("\n[STEP 9] Generating metadata...")
        
        metadata = {
            'pipeline_config': {
                'num_clients': self.num_clients,
                'test_split': self.test_split,
                'val_split': self.val_split,
                'seed': self.seed
            },
            'clients': {}
        }
        
        for client_id, symbols_data in splits.items():
            client_info = {
                'num_symbols': len(symbols_data),
                'symbols': list(symbols_data.keys()),
                'splits': {}
            }
            
            # Aggregate statistics per split
            for split_name in ['train', 'val', 'test']:
                total_rows = 0
                for df_split in symbols_data.values():
                    total_rows += len(df_split[split_name])
                
                client_info['splits'][split_name] = {'num_rows': total_rows}
            
            metadata['clients'][f'client_{client_id:02d}'] = client_info
        
        # Save metadata
        metadata_path = self.output_dir / "metadata" / "client_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to {metadata_path}")
    
    def generate_feature_documentation(self):
        """Generate comprehensive documentation of all engineered features."""
        feature_docs = {
            'original_features': {
                'Date': 'Trading date',
                'Open': 'Opening price (normalized)',
                'High': 'Highest price of the day (normalized)',
                'Low': 'Lowest price of the day (normalized)',
                'Close': 'Closing price (normalized)',
                'Adj Close': 'Adjusted closing price (normalized)',
                'Volume': 'Trading volume in shares (normalized)'
            },
            'engineered_features': {
                'Returns': 'Daily returns as percentage: (Close - Prev Close) / Prev Close * 100',
                'MA_5': 'Moving average of closing price over 5 days',
                'MA_20': 'Moving average of closing price over 20 days',
                'Price_MA5_Ratio': 'Ratio of current close to 5-day MA (momentum indicator)',
                'Price_MA20_Ratio': 'Ratio of current close to 20-day MA (momentum indicator)',
                'Volatility': '20-day rolling standard deviation of returns',
                'Volume_Change': 'Daily volume change as percentage',
                'Volume_Normalized': 'Volume normalized by 20-day moving average',
                'Price_Range': 'Daily price range (High - Low) / Open',
                'Close_Open_Ratio': 'Ratio of close to open price'
            },
            'target_features': {
                'Next_Return': 'Next day return (target for regression) - percentage',
                'Direction': 'Next day price movement direction (target for classification): 1 (up), 0 (flat), -1 (down)'
            },
            'notes': {
                'normalization': 'Features are normalized using StandardScaler (zero mean, unit variance)',
                'missing_data': 'Rows with missing values after feature engineering are removed',
                'temporal_order': 'Train/val/test splits preserve temporal order of data'
            }
        }
        
        docs_path = self.output_dir / "metadata" / "feature_documentation.json"
        with open(docs_path, 'w') as f:
            json.dump(feature_docs, f, indent=2)
        
        print(f"✓ Feature documentation saved to {docs_path}")
    
    def generate_pipeline_report(self, processed_symbols: int, total_rows: int):
        """
        Generate a comprehensive pipeline execution report.
        
        Args:
            processed_symbols: Number of successfully processed symbols
            total_rows: Total data points across all clients
        """
        report = {
            'execution_timestamp': pd.Timestamp.now().isoformat(),
            'data_source': str(self.data_dir),
            'output_directory': str(self.output_dir),
            'statistics': {
                'symbols_processed': processed_symbols,
                'total_data_points': total_rows,
                'federated_clients': self.num_clients,
                'train_split': 1 - self.test_split - self.val_split,
                'val_split': self.val_split,
                'test_split': self.test_split,
            },
            'features': {
                'original': 7,
                'engineered': 10,
                'targets': 2
            }
        }
        
        report_path = self.output_dir / "metadata" / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION REPORT")
        print('='*60)
        print(f"Symbols Processed: {processed_symbols}")
        print(f"Total Data Points: {total_rows}")
        print(f"Federated Clients: {self.num_clients}")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def run_pipeline(
        self,
        num_symbols: int = 50,
        etf_ratio: float = 0.3,
        save_format: str = 'csv',
        force_symbols: Optional[List[str]] = None
    ):
        """
        Execute the complete data pipeline.
        
        Args:
            num_symbols: Number of symbols to process
            etf_ratio: Ratio of ETFs vs stocks
            save_format: Format for saving client data
            force_symbols: Specific symbols to include
        """
        print("\n" + "="*60)
        print("FEDERATED LEARNING DATA PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Load metadata
            metadata = self.load_metadata()
            
            # Step 2: Select symbols
            symbols = self.select_symbols(
                metadata,
                num_symbols=num_symbols,
                etf_ratio=etf_ratio,
                force_symbols=force_symbols
            )
            
            # Step 3: Load and clean data
            valid_symbols = self.load_and_clean_data(symbols)
            if not valid_symbols:
                print("✗ No valid symbols loaded. Pipeline aborted.")
                return
            
            # Step 4: Engineer features
            processed_data = self.process_symbols(valid_symbols)
            if not processed_data:
                print("✗ No symbols successfully processed. Pipeline aborted.")
                return
            
            # Step 5: Normalize features
            normalized_data = self.normalize_features(processed_data)
            
            # Step 6: Distribute to clients
            client_data = self.split_data_for_clients(normalized_data)
            
            # Step 7: Create train/val/test splits
            splits = self.create_train_val_test_splits(client_data)
            
            # Step 8: Save client data
            self.save_client_data(splits, save_format=save_format)
            
            # Step 9: Generate metadata
            self.generate_client_metadata(splits)
            self.generate_feature_documentation()
            
            # Calculate total statistics
            total_rows = sum(
                sum(len(df[split]) for split in ['train', 'val', 'test'])
                for client_data_all in splits.values()
                for df in client_data_all.values()
            )
            
            # Generate report
            self.generate_pipeline_report(len(processed_data), total_rows)
            
            print("✓ Pipeline execution completed successfully!")
            print(f"✓ Check {self.output_dir} for processed data and metadata")
            
        except Exception as e:
            print(f"\n✗ Pipeline execution failed: {e}")
            raise


def main():
    """
    Main entry point for the data pipeline.
    
    Example usage:
        pipeline = StockDataPipeline(
            data_dir="/path/to/stock/data",
            output_dir="./federated_data",
            num_clients=10
        )
        pipeline.run_pipeline(
            num_symbols=50,
            etf_ratio=0.3,
            save_format='csv'
        )
    """
    # Initialize pipeline
    pipeline = StockDataPipeline(
        data_dir="/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/BIg DATA/Project/Stock market dataset",
        output_dir="./federated_data",
        num_clients=10,
        test_split=0.2,
        val_split=0.1,
        seed=42
    )
    
    # Run pipeline
    # Select 50 symbols (30% ETFs, 70% stocks)
    pipeline.run_pipeline(
        num_symbols=50,
        etf_ratio=0.3,
        save_format='csv',  # Can also use 'parquet', 'pickle', or 'numpy'
        force_symbols=['AAPL', 'SPY', 'QQQ']  # Optional: force specific symbols
    )


if __name__ == "__main__":
    main()
