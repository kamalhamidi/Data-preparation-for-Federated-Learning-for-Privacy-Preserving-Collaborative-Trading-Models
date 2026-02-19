"""
Complete Example: Federated Learning Pipeline Usage
===================================================

This script demonstrates all key features of the pipeline:
1. Running the pipeline
2. Analyzing output
3. Loading data with utilities
4. PyTorch integration
5. TensorFlow integration
6. Federated Averaging

Run this as: python example_complete.py
"""

import sys
from pathlib import Path


def example_1_run_pipeline():
    """Example 1: Run the complete pipeline"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Running the Complete Pipeline")
    print("="*70 + "\n")
    
    from federated_data_pipeline import StockDataPipeline
    
    # Initialize pipeline
    pipeline = StockDataPipeline(
        data_dir="/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/BIg DATA/Project/Stock market dataset",
        output_dir="./federated_data",
        num_clients=10,
        test_split=0.2,
        val_split=0.1,
        seed=42
    )
    
    # Run with specific configuration
    print("Running pipeline with configuration:")
    print("  - 50 symbols (30% ETFs)")
    print("  - 10 federated clients")
    print("  - CSV output format")
    print("  - Non-IID distribution\n")
    
    pipeline.run_pipeline(
        num_symbols=50,
        etf_ratio=0.3,
        save_format='csv',
        force_symbols=['AAPL', 'SPY', 'QQQ']
    )
    
    print("\n✓ Pipeline completed!")
    print("Data saved to: ./federated_data")


def example_2_load_and_analyze():
    """Example 2: Load and analyze generated data"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Loading and Analyzing Data")
    print("="*70 + "\n")
    
    from pipeline_utils import FederatedDataLoader, DataStatistics
    
    # Initialize loader
    loader = FederatedDataLoader("./federated_data")
    
    # Load training data for client 0
    print("Loading data for client 0 (training split)...")
    client_data = loader.load_client_data(client_id=0, split='train')
    
    print(f"✓ Loaded {len(client_data)} symbols")
    print(f"  Symbols: {list(client_data.keys())}\n")
    
    # Get statistics
    print("Computing statistics...")
    stats = DataStatistics.compute_client_stats("./federated_data", client_id=0)
    
    print(f"Client 0 Statistics:")
    print(f"  Symbols: {stats['num_symbols']}")
    print(f"  Train samples: {stats['train_rows']:,}")
    print(f"  Val samples:   {stats['val_rows']:,}")
    print(f"  Test samples:  {stats['test_rows']:,}")
    
    # Print feature statistics
    print(f"\nFeature Statistics (first symbol):")
    for feature_name in list(stats['feature_stats'].keys())[:5]:
        feat_stat = stats['feature_stats'][feature_name]
        print(f"  {feature_name}:")
        print(f"    Mean: {feat_stat['mean']:.6f}, Std: {feat_stat['std']:.6f}")


def example_3_numpy_arrays():
    """Example 3: Convert to NumPy arrays for ML models"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Converting to NumPy Arrays")
    print("="*70 + "\n")
    
    from pipeline_utils import FederatedDataLoader
    import numpy as np
    
    loader = FederatedDataLoader("./federated_data")
    
    # Load as numpy arrays
    print("Loading as NumPy arrays...")
    X_train, y_train = loader.load_client_data(
        client_id=0,
        split='train',
        as_array=True
    )
    
    print(f"✓ Loaded NumPy arrays")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_train dtype: {X_train.dtype}")
    print(f"  y_train dtype: {y_train.dtype}")
    
    # Create batches
    print(f"\nCreating mini-batches (batch_size=64)...")
    batches = loader.create_batches(X_train, y_train, batch_size=64)
    print(f"✓ Created {len(batches)} batches")
    
    # Inspect first batch
    X_batch, y_batch = batches[0]
    print(f"  First batch: X={X_batch.shape}, y={y_batch.shape}")


def example_4_pytorch_integration():
    """Example 4: PyTorch DataLoader integration"""
    print("\n" + "="*70)
    print("EXAMPLE 4: PyTorch Integration")
    print("="*70 + "\n")
    
    try:
        from dl_integration import FederatedDataLoaderPyTorch
        import torch
        
        # Create DataLoader for client 0
        print("Creating PyTorch DataLoaders...")
        train_loader = FederatedDataLoaderPyTorch.get_client_loader(
            data_dir="./federated_data",
            client_id=0,
            split='train',
            batch_size=32,
            shuffle=True,
            target_type='regression'
        )
        
        val_loader = FederatedDataLoaderPyTorch.get_client_loader(
            data_dir="./federated_data",
            client_id=0,
            split='val',
            batch_size=32,
            shuffle=False,
            target_type='regression'
        )
        
        print(f"✓ Created DataLoaders")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        
        # Inspect first batch
        print(f"\nInspecting first batch...")
        for X_batch, y_batch in train_loader:
            print(f"  X batch: shape={X_batch.shape}, dtype={X_batch.dtype}")
            print(f"  y batch: shape={y_batch.shape}, dtype={y_batch.dtype}")
            print(f"  X range: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
            print(f"  y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
            break
        
        # Example: Create all client loaders
        print(f"\nCreating DataLoaders for all 10 clients...")
        all_loaders = FederatedDataLoaderPyTorch.get_all_client_loaders(
            data_dir="./federated_data",
            num_clients=10,
            split='train',
            batch_size=32,
            shuffle=True
        )
        print(f"✓ Created {len(all_loaders)} DataLoaders")
        
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")


def example_5_tensorflow_integration():
    """Example 5: TensorFlow tf.data.Dataset integration"""
    print("\n" + "="*70)
    print("EXAMPLE 5: TensorFlow Integration")
    print("="*70 + "\n")
    
    try:
        from dl_integration import FederatedDataLoaderTensorFlow
        import tensorflow as tf
        
        # Create Dataset for client 0
        print("Creating TensorFlow Datasets...")
        train_dataset = FederatedDataLoaderTensorFlow.get_client_dataset(
            data_dir="./federated_data",
            client_id=0,
            split='train',
            batch_size=32,
            shuffle=True,
            target_type='regression'
        )
        
        val_dataset = FederatedDataLoaderTensorFlow.get_client_dataset(
            data_dir="./federated_data",
            client_id=0,
            split='val',
            batch_size=32,
            shuffle=False,
            target_type='regression'
        )
        
        print(f"✓ Created Datasets")
        
        # Inspect first batch
        print(f"\nInspecting first batch...")
        for X_batch, y_batch in train_dataset.take(1):
            print(f"  X batch: shape={X_batch.shape}, dtype={X_batch.dtype}")
            print(f"  y batch: shape={y_batch.shape}, dtype={y_batch.dtype}")
        
        # Example: Create all client datasets
        print(f"\nCreating Datasets for all 10 clients...")
        all_datasets = FederatedDataLoaderTensorFlow.get_all_client_datasets(
            data_dir="./federated_data",
            num_clients=10,
            split='train',
            batch_size=32,
            shuffle=True
        )
        print(f"✓ Created {len(all_datasets)} Datasets")
        
    except ImportError:
        print("✗ TensorFlow not installed")
        print("  Install with: pip install tensorflow")


def example_6_federated_averaging():
    """Example 6: Federated Averaging (FedAvg)"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Federated Averaging")
    print("="*70 + "\n")
    
    from dl_integration import FederatedAveragingUtils
    import numpy as np
    
    # Get dataset sizes for weighting
    print("Computing dataset sizes for each client...")
    sizes = FederatedAveragingUtils.get_client_dataset_sizes(
        data_dir="./federated_data",
        num_clients=10,
        split='train'
    )
    
    print(f"Client dataset sizes:")
    for client_id, size in enumerate(sizes):
        print(f"  Client {client_id}: {size:,} samples")
    
    print(f"Total: {sum(sizes):,} samples")
    
    # Example: Aggregate models with size-based weighting
    print(f"\nSimulating FedAvg aggregation...")
    
    # Create dummy model weights
    num_clients = len(sizes)
    model_weights = []
    for _ in range(num_clients):
        weights = {
            'layer1_weight': np.random.randn(10, 20),
            'layer1_bias': np.random.randn(20),
            'layer2_weight': np.random.randn(20, 1),
            'layer2_bias': np.random.randn(1)
        }
        model_weights.append(weights)
    
    # Aggregate with size-based weighting
    aggregated = FederatedAveragingUtils.aggregate_models(
        model_weights=model_weights,
        client_weights=sizes
    )
    
    print(f"✓ Aggregated model weights from {num_clients} clients")
    print(f"  Aggregated parameters: {list(aggregated.keys())}")
    print(f"  Example - layer1_weight shape: {aggregated['layer1_weight'].shape}")


def example_7_data_validation():
    """Example 7: Data quality validation"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Data Quality Validation")
    print("="*70 + "\n")
    
    from pipeline_utils import FederatedDataLoader, DataQualityValidator
    
    loader = FederatedDataLoader("./federated_data")
    validator = DataQualityValidator()
    
    print("Validating data quality for client 0...")
    
    # Load sample data
    client_data = loader.load_client_data(client_id=0, split='train')
    
    # Validate each symbol
    print(f"\nValidation results:")
    for symbol, df in list(client_data.items())[:3]:  # Check first 3 symbols
        validation = validator.validate_dataframe(df, symbol)
        
        print(f"\n  {symbol}:")
        print(f"    Total rows: {validation['total_rows']}")
        print(f"    Null columns: {len(validation['null_cols'])}")
        if validation['null_cols']:
            print(f"      {validation['null_cols']}")
        print(f"    Duplicate rows: {validation['duplicate_rows']}")
        
        if not validation['null_cols'] and validation['duplicate_rows'] == 0:
            print(f"    ✓ Quality check passed")


def example_8_feature_inspection():
    """Example 8: Inspect engineered features"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Feature Engineering Inspection")
    print("="*70 + "\n")
    
    from pipeline_utils import FederatedDataLoader, FEATURE_GROUPS
    import json
    from pathlib import Path
    
    # Load feature documentation
    doc_path = Path("./federated_data/metadata/feature_documentation.json")
    if doc_path.exists():
        with open(doc_path) as f:
            feature_docs = json.load(f)
        
        print("Engineered Features:\n")
        for feature_name, description in feature_docs['engineered_features'].items():
            print(f"  • {feature_name}")
            print(f"    {description}\n")
    
    # Load sample data and show statistics
    loader = FederatedDataLoader("./federated_data")
    client_data = loader.load_client_data(client_id=0, split='train')
    
    if client_data:
        first_symbol = list(client_data.keys())[0]
        df = client_data[first_symbol]
        
        print(f"Sample Data Statistics ({first_symbol}):")
        print(f"  Total rows: {len(df)}")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\n  Feature ranges:")
        
        feature_cols = [c for c in df.columns if c not in 
                       {'Date', 'Symbol', 'Next_Return', 'Direction'}]
        for col in feature_cols[:8]:
            print(f"    {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FEDERATED LEARNING DATA PIPELINE - COMPLETE EXAMPLES")
    print("="*70)
    
    examples = [
        ("Pipeline Execution", example_1_run_pipeline),
        ("Data Loading & Analysis", example_2_load_and_analyze),
        ("NumPy Array Conversion", example_3_numpy_arrays),
        ("PyTorch Integration", example_4_pytorch_integration),
        ("TensorFlow Integration", example_5_tensorflow_integration),
        ("Federated Averaging", example_6_federated_averaging),
        ("Data Validation", example_7_data_validation),
        ("Feature Inspection", example_8_feature_inspection),
    ]
    
    while True:
        print("\n" + "="*70)
        print("SELECT AN EXAMPLE TO RUN:")
        print("="*70)
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print(f"{len(examples) + 1}. Run All Examples")
        print(f"{len(examples) + 2}. Exit")
        
        try:
            choice = input("\nEnter your choice (1-{}): ".format(len(examples) + 2))
            choice = int(choice)
            
            if choice == len(examples) + 2:
                print("\nGoodbye! 👋")
                break
            elif choice == len(examples) + 1:
                for name, func in examples:
                    print(f"\n\n{'#'*70}")
                    print(f"# {name}")
                    print(f"{'#'*70}")
                    try:
                        func()
                    except Exception as e:
                        print(f"✗ Error: {e}")
                break
            elif 1 <= choice <= len(examples):
                name, func = examples[choice - 1]
                print(f"\n\n{'#'*70}")
                print(f"# {name}")
                print(f"{'#'*70}")
                try:
                    func()
                except Exception as e:
                    print(f"✗ Error: {e}")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break


if __name__ == "__main__":
    main()
