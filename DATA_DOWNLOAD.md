# 📥 Data Download Instructions

This document contains links to download the stock market dataset required for the pipeline.

---

## Dataset Information

**Name**: Stock Market Dataset (Kaggle)  
**Size**: ~2 GB (uncompressed)  
**Contents**: 
- ~2,150 ETF CSV files
- ~5,800 Stock CSV files
- Metadata file (symbols_valid_meta.csv)

---

## Download Link

### Option 1: Kaggle API (Recommended)
```bash
# Install kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d "your-dataset-name"

# Extract
unzip your-dataset-name.zip -d "./Stock market dataset"
```

### Option 2: Direct Download
[Download from Kaggle](https://www.kaggle.com/)
```
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/data
```

---

## Setup Instructions

### Step 1: Download the data
Use one of the options above to download the dataset.

### Step 2: Extract the files
```bash
unzip dataset.zip -d "./Stock market dataset"
```

### Step 3: Verify structure
```bash
ls -la "Stock market dataset/"
# Should show:
# - etfs/  (folder with ~2150 CSV files)
# - stocks/  (folder with ~5800 CSV files)
# - symbols_valid_meta.csv
```

### Step 4: Run the pipeline
```bash
python run_pipeline.py
```

---

## Storage Requirements

- **Download**: ~500 MB (compressed)
- **Extract**: ~2 GB (uncompressed)
- **Pipeline Output**: ~1-3 GB (depends on config)
- **Total**: ~5-6 GB disk space recommended

---

## Troubleshooting

### Error: "Stock market dataset not found"
- Ensure the dataset is extracted to the correct location
- Run from the project directory

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: Out of memory
- Use smaller config: `python run_pipeline.py --mode basic --config quick_test`
- Or reduce symbols: `python run_pipeline.py --num-symbols 30`

---

## Data Structure

Once downloaded, your directory should look like:
```
Stock market dataset/
├── etfs/
│   ├── AAAU.csv
│   ├── AADR.csv
│   └── ... (2150+ files)
├── stocks/
│   ├── A.csv
│   ├── AA.csv
│   └── ... (5800+ files)
└── symbols_valid_meta.csv
```

---

## Next Steps

1. Download the data using one of the methods above
2. Extract it to `./Stock market dataset/`
3. Run: `python run_pipeline.py`
4. Check output in `./federated_data/`

---

## Questions?

Refer to:
- [START_HERE.md](START_HERE.md) - Quick overview
- [QUICKSTART.md](QUICKSTART.md) - Setup guide
- [README.md](README.md) - Full documentation

**Happy learning!** 🚀
