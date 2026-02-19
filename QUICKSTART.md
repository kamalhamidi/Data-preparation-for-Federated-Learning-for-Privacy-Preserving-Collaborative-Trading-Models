# Quick Start Guide - Federated Learning Data Pipeline

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn tqdm
```

Optional (for deep learning):
```bash
pip install torch                    # For PyTorch
pip install tensorflow               # For TensorFlow
```

### 2. Run the Pipeline

#### **Simplest: One Command**

```bash
python run_pipeline.py
```

This runs with the default "medium" configuration:
- 50 symbols (30% ETFs, 70% stocks)
- 10 federated clients
- CSV output format
- Non-IID distribution

#### **Custom Configuration**

```bash
# Quick test (10 symbols, 3 clients)
python run_pipeline.py --mode basic --config quick_test

# Large scale (100 symbols, 20 clients)
python run_pipeline.py --mode advanced --num-symbols 100 --num-clients 20 --format parquet

# Extra large (200 symbols, 50 clients)
python run_pipeline.py --mode basic --config xl
```

### 3. Check Output

```bash
# List generated client data
ls -la federated_data/clients/

# View metadata
cat federated_data/metadata/pipeline_report.json

# Analyze data
python run_pipeline.py --mode analysis
```

---

## Python Usage

### Load Data in Your Code

```python
from federated_data_pipeline import StockDataPipeline

# Initialize
pipeline = StockDataPipeline(
    num_clients=10,
    output_dir="./federated_data"
)

# Run complete pipeline
pipeline.run_pipeline(num_symbols=50, etf_ratio=0.3)
```

### Load Data for Training

```python
from pipeline_utils import FederatedDataLoader

loader = FederatedDataLoader("./federated_data")

# Load training data for client 0
client_data = loader.load_client_data(client_id=0, split='train')

# Convert to numpy arrays
X, y = loader.load_client_data(client_id=0, split='train', as_array=True)

# Create batches
batches = loader.create_batches(X, y, batch_size=32)
```

### PyTorch Integration

```python
from dl_integration import FederatedDataLoaderPyTorch

# Create DataLoader
train_loader = FederatedDataLoaderPyTorch.get_client_loader(
    data_dir="./federated_data",
    client_id=0,
    split='train',
    batch_size=32
)

# Train your model
for X_batch, y_batch in train_loader:
    # Your training code here
    predictions = model(X_batch)
    loss = loss_fn(predictions, y_batch)
```

### TensorFlow Integration

```python
from dl_integration import FederatedDataLoaderTensorFlow

# Create Dataset
train_dataset = FederatedDataLoaderTensorFlow.get_client_dataset(
    data_dir="./federated_data",
    client_id=0,
    split='train',
    batch_size=32
)

# Train your model
model.fit(train_dataset, epochs=10)
```

---

## Data Structure After Pipeline

```
federated_data/
├── clients/
│   ├── client_00/
│   │   ├── AAPL_train.csv
│   │   ├── AAPL_val.csv
│   │   ├── AAPL_test.csv
│   │   ├── SPY_train.csv
│   │   ├── SPY_val.csv
│   │   ├── SPY_test.csv
│   │   └── ...
│   ├── client_01/
│   └── ...
├── scalers/
│   └── feature_scalers.pkl
└── metadata/
    ├── client_metadata.json
    ├── feature_documentation.json
    └── pipeline_report.json
```

---

## Data Format

Each CSV file contains:

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Trading date |
| Open | float | Opening price (normalized) |
| High | float | High price (normalized) |
| Low | float | Low price (normalized) |
| Close | float | Close price (normalized) |
| Adj Close | float | Adjusted close (normalized) |
| Volume | float | Trading volume (normalized) |
| Returns | float | Daily returns (%) |
| MA_5 | float | 5-day moving average |
| MA_20 | float | 20-day moving average |
| Price_MA5_Ratio | float | Price to 5-day MA ratio |
| Price_MA20_Ratio | float | Price to 20-day MA ratio |
| Volatility | float | 20-day rolling volatility |
| Volume_Change | float | Volume change (%) |
| Volume_Normalized | float | Normalized volume |
| Price_Range | float | High-Low / Open |
| Close_Open_Ratio | float | Close / Open |
| Next_Return | float | **TARGET (Regression)** |
| Direction | int | **TARGET (Classification)**: 1, 0, -1 |

---

## Common Commands

```bash
# Run with small config (quick test)
python run_pipeline.py --mode basic --config quick_test

# Run advanced with custom parameters
python run_pipeline.py --mode advanced \
  --num-symbols 75 \
  --num-clients 15 \
  --etf-ratio 0.25 \
  --format parquet

# Analyze existing data
python run_pipeline.py --mode analysis

# Validate data quality
python run_pipeline.py --mode validate

# Test PyTorch integration
python run_pipeline.py --mode pytorch

# Test TensorFlow integration
python run_pipeline.py --mode tensorflow

# Run everything
python run_pipeline.py --mode all
```

---

## Configuration Presets

### Quick Test
```
num_symbols: 10
num_clients: 3
etf_ratio: 0.4
Time: ~2 minutes
```

### Small
```
num_symbols: 30
num_clients: 5
etf_ratio: 0.3
Time: ~5 minutes
```

### Medium (Default)
```
num_symbols: 50
num_clients: 10
etf_ratio: 0.3
Time: ~10 minutes
```

### Large
```
num_symbols: 100
num_clients: 20
etf_ratio: 0.25
Time: ~25 minutes
```

### XL
```
num_symbols: 200
num_clients: 50
etf_ratio: 0.2
Time: ~60 minutes
```

---

## Features Engineered

### Price Features
- Open, High, Low, Close, Adj Close (normalized)

### Volume Features
- Volume (normalized)
- Volume_Change (%)
- Volume_Normalized (relative to 20-day mean)

### Momentum Features
- MA_5: 5-day moving average
- MA_20: 20-day moving average
- Price_MA5_Ratio: Momentum indicator
- Price_MA20_Ratio: Momentum indicator

### Volatility Features
- Volatility: 20-day rolling std of returns

### Range Features
- Price_Range: Daily high-low range
- Close_Open_Ratio: Close-to-open ratio

### Returns
- Returns: Daily percentage returns

### Targets
- Next_Return: Next day return (regression)
- Direction: Price movement direction (classification)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:**
```bash
pip install pandas numpy scikit-learn tqdm
```

### Issue: "FileNotFoundError: Stock market dataset not found"
**Solution:** Specify the correct data directory:
```bash
python run_pipeline.py --data-dir "/path/to/Stock market dataset"
```

### Issue: Out of memory
**Solution:** Use smaller config or more clients:
```bash
python run_pipeline.py --mode basic --config quick_test --num-clients 5
```

### Issue: Slow processing
**Solution:** Use Parquet format instead of CSV:
```bash
python run_pipeline.py --format parquet
```

---

## Next Steps

1. **Understand the data:**
   ```bash
   python run_pipeline.py --mode analysis
   ```

2. **Validate quality:**
   ```bash
   python run_pipeline.py --mode validate
   ```

3. **Load for training (PyTorch example):**
   ```python
   from dl_integration import FederatedDataLoaderPyTorch
   
   loader = FederatedDataLoaderPyTorch.get_client_loader(
       data_dir="./federated_data",
       client_id=0,
       split='train',
       batch_size=32
   )
   
   for X, y in loader:
       # Train your federated learning model
       pass
   ```

4. **Implement federated averaging:**
   ```python
   from dl_integration import FederatedAveragingUtils
   
   sizes = FederatedAveragingUtils.get_client_dataset_sizes(
       data_dir="./federated_data",
       num_clients=10
   )
   
   aggregated = FederatedAveragingUtils.aggregate_models(
       model_weights=[...],
       client_weights=sizes
   )
   ```

---

## For More Information

- See **README.md** for comprehensive documentation
- Check **federated_data/metadata/pipeline_report.json** after running
- Review feature documentation in **federated_data/metadata/feature_documentation.json**

---

**Happy federated learning!** 🚀
