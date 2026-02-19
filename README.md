# Federated Learning Data Pipeline for Stock Market Data

Complete Python pipeline for preparing stock/ETF market data for federated learning applications.

## Overview

This project implements a production-ready data pipeline that transforms raw CSV stock market data into federated-learning-ready datasets. It handles all preprocessing, feature engineering, normalization, and data distribution steps required for training privacy-preserving collaborative models.

## Features

### 1. **Data Loading & Cleaning**
- Load CSV files from ETF and Stock directories
- Handle missing values (forward fill + backward fill)
- Remove duplicates and sort by date
- Validate data quality and filter by minimum requirements

### 2. **Feature Engineering** (10+ engineered features)
- **Price Features**: Open, High, Low, Close, Adj Close (normalized)
- **Returns**: Daily returns as percentage
- **Moving Averages**: 5-day and 20-day moving averages
- **Momentum Indicators**: Price-to-MA ratios
- **Volatility**: 20-day rolling standard deviation
- **Volume Features**: Volume change percentage, normalized volume
- **Range Features**: Daily price range, Close-to-Open ratio
- **Targets**: Next-day return (regression) and price direction (classification)

### 3. **Data Normalization**
- StandardScaler (zero mean, unit variance) for all features
- Scalers fitted on aggregated data for consistency
- Scalers saved for future inference

### 4. **Federated Data Distribution**
- Non-IID distribution: Different clients get different symbols (realistic)
- IID distribution: Each client gets mixed samples from all symbols
- Temporal order preserved for time-series modeling

### 5. **Train/Val/Test Splits**
- Temporal splits (preserving chronological order)
- Configurable split ratios
- Applied per-symbol for each client

### 6. **Multiple Export Formats**
- **CSV**: Human-readable, easy to inspect
- **Parquet**: Efficient columnar storage
- **Pickle**: Python serialization
- **NumPy**: Arrays for deep learning frameworks

### 7. **Deep Learning Integration**
- PyTorch Dataset/DataLoader wrappers
- TensorFlow tf.data.Dataset support
- Federated Averaging utilities
- Batch generation and prefetching

## Project Structure

```
.
├── federated_data_pipeline.py       # Main pipeline implementation
├── pipeline_utils.py                # Utilities and helpers
├── dl_integration.py                # PyTorch/TensorFlow integration
├── run_pipeline.py                  # Execution script
├── README.md                        # This file
└── federated_data/                  # Output directory (created after running)
    ├── clients/
    │   ├── client_00/
    │   │   ├── AAPL_train.csv
    │   │   ├── AAPL_val.csv
    │   │   ├── AAPL_test.csv
    │   │   └── ...
    │   ├── client_01/
    │   └── ...
    ├── scalers/
    │   └── feature_scalers.pkl      # Fitted scalers for normalization
    └── metadata/
        ├── client_metadata.json     # Client data statistics
        ├── feature_documentation.json # Feature definitions
        └── pipeline_report.json     # Pipeline execution report
```

## Installation

### Prerequisites
- Python 3.8+
- pandas
- numpy
- scikit-learn
- tqdm

### Optional Dependencies
- PyTorch (for deep learning integration)
- TensorFlow (for deep learning integration)

### Setup

```bash
# Clone or download the pipeline files
cd /path/to/project

# Install required packages
pip install pandas numpy scikit-learn tqdm

# Optional: Install deep learning frameworks
pip install torch                    # PyTorch
pip install tensorflow               # TensorFlow
```

## Quick Start

### Basic Usage

```python
from federated_data_pipeline import StockDataPipeline

# Initialize pipeline
pipeline = StockDataPipeline(
    data_dir="./Stock market dataset",
    output_dir="./federated_data",
    num_clients=10,
    test_split=0.2,
    val_split=0.1,
    seed=42
)

# Run complete pipeline
pipeline.run_pipeline(
    num_symbols=50,           # Select 50 symbols
    etf_ratio=0.3,           # 30% ETFs, 70% stocks
    save_format='csv',       # Save as CSV files
    force_symbols=['AAPL', 'SPY', 'QQQ']  # Include specific symbols
)
```

### Advanced Configuration

```python
# Fine-grained control
pipeline = StockDataPipeline(
    data_dir="/path/to/stock/data",
    output_dir="./federated_data",
    num_clients=20,          # 20 federated clients
    test_split=0.15,         # 15% for testing
    val_split=0.15,          # 15% for validation
    seed=123
)

# Load and clean data
metadata = pipeline.load_metadata()
symbols = pipeline.select_symbols(
    metadata,
    num_symbols=100,
    etf_ratio=0.4
)

# Process data
valid_data = pipeline.load_and_clean_data(symbols, min_rows=400)
processed_data = pipeline.process_symbols(valid_data)
normalized_data = pipeline.normalize_features(processed_data, scaler_type='standard')

# Distribute to clients
client_data = pipeline.split_data_for_clients(
    normalized_data,
    distribution_type='non_iid'
)

# Create splits
splits = pipeline.create_train_val_test_splits(client_data)

# Save data
pipeline.save_client_data(splits, save_format='parquet')
pipeline.generate_client_metadata(splits)
```

## Using the Data with PyTorch

```python
from dl_integration import FederatedDataLoaderPyTorch
from torch.optim import Adam

# Load data for client 0
train_loader = FederatedDataLoaderPyTorch.get_client_loader(
    data_dir="./federated_data",
    client_id=0,
    split='train',
    batch_size=32,
    shuffle=True,
    target_type='regression'
)

# Train a model
model = YourModel()  # Define your model
optimizer = Adam(model.parameters())

for epoch in range(10):
    for X_batch, y_batch in train_loader:
        # Forward pass
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Using the Data with TensorFlow

```python
from dl_integration import FederatedDataLoaderTensorFlow

# Load data for all clients
train_datasets = FederatedDataLoaderTensorFlow.get_all_client_datasets(
    data_dir="./federated_data",
    num_clients=10,
    split='train',
    batch_size=32,
    shuffle=True
)

# Train federated model
for client_id, dataset in train_datasets.items():
    for X_batch, y_batch in dataset:
        # Train on client's local data
        model.fit(X_batch, y_batch, epochs=1, verbose=0)
    
    # Aggregate weights across clients
    # (Implement FedAvg here)
```

## Data Analysis and Quality Checks

```python
from pipeline_utils import DataStatistics, DataQualityValidator

# Print dataset summary
DataStatistics.print_summary(
    client_data_dir="./federated_data",
    num_clients=10
)

# Compute client statistics
stats = DataStatistics.compute_client_stats(
    client_data_dir="./federated_data",
    client_id=0
)

# Validate data quality
validator = DataQualityValidator()
issues = validator.validate_dataframe(df, symbol="AAPL")
```

## Feature Documentation

All engineered features are documented in `federated_data/metadata/feature_documentation.json`:

### Original Features (from raw data)
- **Date**: Trading date
- **Open, High, Low, Close, Adj Close**: Price levels (normalized)
- **Volume**: Trading volume in shares (normalized)

### Engineered Features
- **Returns**: Daily percentage returns (%)
- **MA_5, MA_20**: 5-day and 20-day moving averages
- **Price_MA5_Ratio, Price_MA20_Ratio**: Momentum indicators
- **Volatility**: 20-day rolling standard deviation of returns
- **Volume_Change**: Daily volume change (%)
- **Volume_Normalized**: Volume relative to 20-day average
- **Price_Range**: Daily price range as fraction of opening price
- **Close_Open_Ratio**: Close-to-open price ratio

### Target Features
- **Next_Return**: Next-day return (regression target)
- **Direction**: Price movement direction: 1 (up), 0 (flat), -1 (down)

## Output Structure

After running the pipeline, the `federated_data/` directory contains:

```
federated_data/
├── clients/
│   ├── client_00/
│   │   ├── AAPL_train.csv      # Training data for AAPL
│   │   ├── AAPL_val.csv        # Validation data
│   │   ├── AAPL_test.csv       # Test data
│   │   ├── SPY_train.csv
│   │   └── ...
│   ├── client_01/
│   └── ...
│
├── scalers/
│   └── feature_scalers.pkl     # StandardScaler objects
│
└── metadata/
    ├── client_metadata.json           # Size and symbol info per client
    ├── feature_documentation.json     # Feature definitions
    ├── pipeline_report.json           # Pipeline statistics
    └── pipeline_config.json           # Configuration used
```

## Configuration Parameters

### Pipeline Initialization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | - | Root directory with stock/ETF data |
| `output_dir` | `./federated_data` | Output directory for processed data |
| `num_clients` | 10 | Number of federated clients |
| `test_split` | 0.2 | Proportion for testing (20%) |
| `val_split` | 0.1 | Proportion for validation (10%) |
| `seed` | 42 | Random seed for reproducibility |

### Run Pipeline Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_symbols` | 50 | Total symbols to process |
| `etf_ratio` | 0.3 | Proportion of ETFs vs stocks |
| `save_format` | 'csv' | Format: csv, parquet, pickle, numpy |
| `force_symbols` | None | Specific symbols to include |

## Performance Considerations

### Memory Usage
- Loading all data at once: ~2-3GB for 50 symbols
- Per-client storage: ~100-300MB (depending on number of symbols)
- Scalers storage: ~1MB

### Processing Time
- Data loading: ~5-10 minutes for 50 symbols
- Feature engineering: ~2-5 minutes
- Normalization: ~1-2 minutes
- Total pipeline: ~20-30 minutes for 50 symbols

### Optimization Tips
1. **Use Parquet format** for faster I/O
2. **Reduce num_symbols** if memory is limited
3. **Increase num_clients** to distribute data
4. **Use multiprocessing** for data loading in PyTorch (set `num_workers > 0`)

## Federated Learning Integration

### Non-IID Data Distribution
The pipeline creates realistic non-IID (non-independent-identically-distributed) data:
- Each client has a different set of symbols
- Different clients have different data distributions
- Reflects real federated scenarios where data is heterogeneous

### Federated Averaging (FedAvg)
Example implementation:

```python
from dl_integration import FederatedAveragingUtils

# Get dataset sizes for weighting
sizes = FederatedAveragingUtils.get_client_dataset_sizes(
    data_dir="./federated_data",
    num_clients=10
)

# Aggregate models with size-based weighting
aggregated_weights = FederatedAveragingUtils.aggregate_models(
    model_weights=[client_weights for client in clients],
    client_weights=sizes
)
```

## Common Issues and Solutions

### Issue: "File not found" errors
**Solution**: Verify the `data_dir` path points to the Stock market dataset directory

### Issue: Memory errors with large datasets
**Solution**: Reduce `num_symbols` or increase `num_clients` to distribute data

### Issue: Data quality warnings
**Solution**: Adjust `min_rows` parameter or check source data

### Issue: Feature has too many NaN values
**Solution**: Pipeline automatically removes these rows during feature engineering

## Advanced Usage

### Custom Feature Engineering
Extend the `engineer_features()` method:

```python
def engineer_custom_features(self, df):
    # Add your custom features
    df['Custom_Feature'] = df['Close'] / df['MA_20']
    return df
```

### Custom Data Distribution
Implement custom client assignment logic in `split_data_for_clients()`:

```python
# Example: Assign symbols based on volatility
high_vol = symbols[:len(symbols)//2]  # High volatility symbols
low_vol = symbols[len(symbols)//2:]   # Low volatility symbols

client_data[0] = {s: data[s] for s in high_vol}
client_data[1] = {s: data[s] for s in low_vol}
```

## Contributing

To contribute improvements:
1. Test with different datasets
2. Add more feature engineering options
3. Improve distributed processing
4. Add more output formats

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this pipeline in your research, please cite:

```
@software{federated_stock_pipeline,
    title = {Federated Learning Data Pipeline for Stock Market Data},
    author = {Big Data Project Team},
    year = {2026},
    url = {https://github.com/your-repo}
}
```

## Support

For issues, questions, or suggestions:
1. Check the documentation and examples
2. Review the pipeline report in `metadata/pipeline_report.json`
3. Validate data quality using `DataQualityValidator`

---

**Last Updated**: February 2026
