#!/usr/bin/env python3
"""
Run Script for Federated Learning Data Pipeline
================================================

This script provides complete examples for running the data pipeline
with different configurations and use cases.

Usage:
    python run_pipeline.py --mode basic
    python run_pipeline.py --mode advanced
    python run_pipeline.py --mode analysis
"""

import argparse
import sys
from pathlib import Path
from federated_data_pipeline import StockDataPipeline
from pipeline_utils import DataStatistics, FederatedDataLoader, DataQualityValidator


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

PIPELINE_CONFIGS = {
    'quick_test': {
        'description': 'Quick test with minimal data',
        'num_symbols': 10,
        'num_clients': 3,
        'etf_ratio': 0.4,
        'save_format': 'csv'
    },
    'small': {
        'description': 'Small pilot dataset',
        'num_symbols': 30,
        'num_clients': 5,
        'etf_ratio': 0.3,
        'save_format': 'csv'
    },
    'medium': {
        'description': 'Medium-sized dataset (default)',
        'num_symbols': 50,
        'num_clients': 10,
        'etf_ratio': 0.3,
        'save_format': 'csv'
    },
    'large': {
        'description': 'Large production dataset',
        'num_symbols': 100,
        'num_clients': 20,
        'etf_ratio': 0.25,
        'save_format': 'parquet'  # More efficient for large data
    },
    'xl': {
        'description': 'Extra-large dataset',
        'num_symbols': 200,
        'num_clients': 50,
        'etf_ratio': 0.2,
        'save_format': 'parquet'
    }
}


# ============================================================================
# BASIC PIPELINE RUN
# ============================================================================

def run_basic_pipeline(config_name='medium', data_dir=None, output_dir='./federated_data'):
    """
    Run pipeline with predefined configuration.
    
    Args:
        config_name: Name of configuration preset
        data_dir: Path to stock data directory
        output_dir: Output directory for processed data
    """
    print("\n" + "="*70)
    print("BASIC PIPELINE EXECUTION")
    print("="*70)
    
    if config_name not in PIPELINE_CONFIGS:
        print(f"✗ Unknown configuration: {config_name}")
        print(f"  Available: {', '.join(PIPELINE_CONFIGS.keys())}")
        return
    
    config = PIPELINE_CONFIGS[config_name]
    print(f"\nConfiguration: {config_name}")
    print(f"Description: {config['description']}")
    print(f"  - Symbols: {config['num_symbols']}")
    print(f"  - Clients: {config['num_clients']}")
    print(f"  - ETF Ratio: {config['etf_ratio']}")
    print(f"  - Format: {config['save_format']}\n")
    
    # Determine data directory
    if data_dir is None:
        data_dir = "/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/BIg DATA/Project/Stock market dataset"
    
    # Initialize pipeline
    pipeline = StockDataPipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        num_clients=config['num_clients'],
        test_split=0.2,
        val_split=0.1,
        seed=42
    )
    
    # Run pipeline
    pipeline.run_pipeline(
        num_symbols=config['num_symbols'],
        etf_ratio=config['etf_ratio'],
        save_format=config['save_format'],
        force_symbols=['AAPL', 'SPY', 'QQQ']  # Force inclusion of major symbols
    )


# ============================================================================
# ADVANCED PIPELINE RUN WITH CUSTOM PARAMETERS
# ============================================================================

def run_advanced_pipeline(
    num_symbols=50,
    num_clients=10,
    etf_ratio=0.3,
    save_format='csv',
    distribution_type='non_iid',
    data_dir=None,
    output_dir='./federated_data'
):
    """
    Run pipeline with fine-grained control over parameters.
    
    Args:
        num_symbols: Number of symbols to process
        num_clients: Number of federated clients
        etf_ratio: Ratio of ETFs to include
        save_format: Output format (csv, parquet, pickle, numpy)
        distribution_type: Data distribution (non_iid, iid)
        data_dir: Input data directory
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("ADVANCED PIPELINE EXECUTION")
    print("="*70)
    
    if data_dir is None:
        data_dir = "/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/BIg DATA/Project/Stock market dataset"
    
    print(f"\nPipeline Configuration:")
    print(f"  - Symbols: {num_symbols}")
    print(f"  - Clients: {num_clients}")
    print(f"  - ETF Ratio: {etf_ratio}")
    print(f"  - Distribution: {distribution_type}")
    print(f"  - Format: {save_format}\n")
    
    # Initialize pipeline
    pipeline = StockDataPipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        num_clients=num_clients,
        test_split=0.2,
        val_split=0.1,
        seed=42
    )
    
    # Step-by-step execution for better control
    print("\n[STEP 1/9] Loading metadata...")
    metadata = pipeline.load_metadata()
    
    print("\n[STEP 2/9] Selecting symbols...")
    symbols = pipeline.select_symbols(
        metadata,
        num_symbols=num_symbols,
        etf_ratio=etf_ratio
    )
    
    print("\n[STEP 3/9] Loading and cleaning data...")
    valid_data = pipeline.load_and_clean_data(symbols, min_rows=500)
    
    if not valid_data:
        print("✗ No valid symbols. Pipeline aborted.")
        return
    
    print("\n[STEP 4/9] Engineering features...")
    processed_data = pipeline.process_symbols(valid_data)
    
    print("\n[STEP 5/9] Normalizing features...")
    normalized_data = pipeline.normalize_features(
        processed_data,
        scaler_type='standard'
    )
    
    print("\n[STEP 6/9] Distributing to clients (distribution_type={})...".format(distribution_type))
    client_data = pipeline.split_data_for_clients(
        normalized_data,
        distribution_type=distribution_type
    )
    
    print("\n[STEP 7/9] Creating train/val/test splits...")
    splits = pipeline.create_train_val_test_splits(client_data)
    
    print("\n[STEP 8/9] Saving client data (format={})...".format(save_format))
    pipeline.save_client_data(splits, save_format=save_format)
    
    print("\n[STEP 9/9] Generating metadata and documentation...")
    pipeline.generate_client_metadata(splits)
    pipeline.generate_feature_documentation()
    
    # Final statistics
    total_rows = sum(
        sum(len(df[split]) for split in ['train', 'val', 'test'])
        for client_data_all in splits.values()
        for df in client_data_all.values()
    )
    
    pipeline.generate_pipeline_report(len(processed_data), total_rows)


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def analyze_pipeline_output(output_dir='./federated_data', num_clients=10):
    """
    Analyze and display statistics about generated federated data.
    
    Args:
        output_dir: Path to federated data directory
        num_clients: Number of clients
    """
    print("\n" + "="*70)
    print("PIPELINE OUTPUT ANALYSIS")
    print("="*70)
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"✗ Output directory not found: {output_dir}")
        return
    
    # Print overall statistics
    print("\n[DATASET SUMMARY]")
    DataStatistics.print_summary(output_dir, num_clients)
    
    # Analyze individual client
    print("\n[DETAILED ANALYSIS - CLIENT 0]")
    try:
        stats = DataStatistics.compute_client_stats(output_dir, client_id=0)
        
        print(f"Symbols: {stats['num_symbols']}")
        print(f"  {', '.join(stats['symbols'][:5])}...")
        print(f"\nData Splits:")
        print(f"  Train: {stats['train_rows']:,} samples")
        print(f"  Val:   {stats['val_rows']:,} samples")
        print(f"  Test:  {stats['test_rows']:,} samples")
        
        print(f"\nFeature Statistics (sample):")
        for feature_name in list(stats['feature_stats'].keys())[:3]:
            feat_stats = stats['feature_stats'][feature_name]
            print(f"  {feature_name}:")
            print(f"    Mean: {feat_stats['mean']:.4f}, Std: {feat_stats['std']:.4f}")
            print(f"    Min:  {feat_stats['min']:.4f}, Max: {feat_stats['max']:.4f}")
        
    except Exception as e:
        print(f"✗ Error analyzing client data: {e}")
    
    # Load sample data
    print("\n[SAMPLE DATA]")
    try:
        loader = FederatedDataLoader(output_dir)
        client_data = loader.load_client_data(client_id=0, split='train')
        
        if client_data:
            first_symbol = list(client_data.keys())[0]
            df = client_data[first_symbol]
            
            print(f"Symbol: {first_symbol}")
            print(f"Shape: {df.shape}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nColumns: {', '.join(df.columns[:7])}...")
    except Exception as e:
        print(f"✗ Error loading sample data: {e}")


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data_quality(output_dir='./federated_data', num_clients=10):
    """
    Validate data quality for all clients.
    
    Args:
        output_dir: Path to federated data directory
        num_clients: Number of clients
    """
    print("\n" + "="*70)
    print("DATA QUALITY VALIDATION")
    print("="*70)
    
    output_path = Path(output_dir)
    loader = FederatedDataLoader(output_dir)
    validator = DataQualityValidator()
    
    issues_found = 0
    
    for client_id in range(min(num_clients, 3)):  # Check first 3 clients
        print(f"\nValidating Client {client_id}:")
        
        try:
            client_data = loader.load_client_data(client_id, split='train')
            
            for symbol, df in list(client_data.items())[:2]:  # Check 2 symbols
                validation_result = validator.validate_dataframe(df, symbol)
                
                print(f"  {symbol}:")
                print(f"    Rows: {validation_result['total_rows']}")
                print(f"    Nulls: {len(validation_result['null_cols'])} columns affected")
                print(f"    Duplicates: {validation_result['duplicate_rows']}")
                
                if validation_result['null_cols'] or validation_result['duplicate_rows']:
                    issues_found += 1
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if issues_found == 0:
        print("\n✓ All validated data passed quality checks!")
    else:
        print(f"\n⚠ Found issues in {issues_found} datasets")


# ============================================================================
# PYTORCH INTEGRATION EXAMPLE
# ============================================================================

def pytorch_integration_example(output_dir='./federated_data'):
    """
    Example: Load data with PyTorch and create DataLoaders.
    
    Args:
        output_dir: Path to federated data directory
    """
    print("\n" + "="*70)
    print("PYTORCH INTEGRATION EXAMPLE")
    print("="*70)
    
    try:
        from dl_integration import FederatedDataLoaderPyTorch
        import torch
        
        print("\n[Creating PyTorch DataLoaders]")
        
        # Get loaders for all clients
        train_loaders = FederatedDataLoaderPyTorch.get_all_client_loaders(
            data_dir=output_dir,
            num_clients=10,
            split='train',
            batch_size=32,
            shuffle=True
        )
        
        print(f"✓ Created {len(train_loaders)} DataLoaders")
        
        # Inspect first batch
        print("\n[Inspecting First Batch]")
        for client_id, loader in list(train_loaders.items())[:1]:
            for X_batch, y_batch in loader:
                print(f"Client {client_id}:")
                print(f"  Feature batch shape: {X_batch.shape}")
                print(f"  Target batch shape: {y_batch.shape}")
                print(f"  Feature dtype: {X_batch.dtype}")
                print(f"  Target range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
                break
        
        print("\n✓ PyTorch integration successful!")
        
    except ImportError:
        print("✗ PyTorch not installed. Install with: pip install torch")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# TENSORFLOW INTEGRATION EXAMPLE
# ============================================================================

def tensorflow_integration_example(output_dir='./federated_data'):
    """
    Example: Load data with TensorFlow and create tf.data.Datasets.
    
    Args:
        output_dir: Path to federated data directory
    """
    print("\n" + "="*70)
    print("TENSORFLOW INTEGRATION EXAMPLE")
    print("="*70)
    
    try:
        from dl_integration import FederatedDataLoaderTensorFlow
        import tensorflow as tf
        
        print("\n[Creating TensorFlow Datasets]")
        
        # Get datasets for all clients
        train_datasets = FederatedDataLoaderTensorFlow.get_all_client_datasets(
            data_dir=output_dir,
            num_clients=10,
            split='train',
            batch_size=32,
            shuffle=True
        )
        
        print(f"✓ Created {len(train_datasets)} tf.data.Datasets")
        
        # Inspect first batch
        print("\n[Inspecting First Batch]")
        for client_id, dataset in list(train_datasets.items())[:1]:
            for X_batch, y_batch in dataset.take(1):
                print(f"Client {client_id}:")
                print(f"  Feature batch shape: {X_batch.shape}")
                print(f"  Target batch shape: {y_batch.shape}")
                print(f"  Feature dtype: {X_batch.dtype}")
                break
        
        print("\n✓ TensorFlow integration successful!")
        
    except ImportError:
        print("✗ TensorFlow not installed. Install with: pip install tensorflow")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with command-line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Federated Learning Data Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default medium configuration
  python run_pipeline.py --mode basic
  
  # Run with custom parameters
  python run_pipeline.py --mode advanced --num-symbols 100 --num-clients 20
  
  # Analyze existing output
  python run_pipeline.py --mode analysis
  
  # Validate data quality
  python run_pipeline.py --mode validate
  
  # Test PyTorch integration
  python run_pipeline.py --mode pytorch
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['basic', 'advanced', 'analysis', 'validate', 'pytorch', 'tensorflow', 'all'],
        default='basic',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        choices=list(PIPELINE_CONFIGS.keys()),
        default='medium',
        help='Configuration preset (basic mode only)'
    )
    
    parser.add_argument(
        '--num-symbols',
        type=int,
        default=50,
        help='Number of symbols to process (advanced mode)'
    )
    
    parser.add_argument(
        '--num-clients',
        type=int,
        default=10,
        help='Number of federated clients'
    )
    
    parser.add_argument(
        '--etf-ratio',
        type=float,
        default=0.3,
        help='Ratio of ETFs vs stocks'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'parquet', 'pickle', 'numpy'],
        default='csv',
        help='Output format'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./federated_data',
        help='Output directory'
    )
    
    parser.add_argument(
        '--data-dir',
        default=None,
        help='Stock data directory (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Execute requested mode
    if args.mode == 'basic':
        run_basic_pipeline(
            config_name=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'advanced':
        run_advanced_pipeline(
            num_symbols=args.num_symbols,
            num_clients=args.num_clients,
            etf_ratio=args.etf_ratio,
            save_format=args.format,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'analysis':
        analyze_pipeline_output(args.output_dir, args.num_clients)
    
    elif args.mode == 'validate':
        validate_data_quality(args.output_dir, args.num_clients)
    
    elif args.mode == 'pytorch':
        pytorch_integration_example(args.output_dir)
    
    elif args.mode == 'tensorflow':
        tensorflow_integration_example(args.output_dir)
    
    elif args.mode == 'all':
        print("\n" + "="*70)
        print("RUNNING ALL MODES")
        print("="*70)
        
        run_basic_pipeline(config_name='small', data_dir=args.data_dir, output_dir=args.output_dir)
        analyze_pipeline_output(args.output_dir, 5)
        validate_data_quality(args.output_dir, 5)
        pytorch_integration_example(args.output_dir)
        tensorflow_integration_example(args.output_dir)


if __name__ == "__main__":
    main()
