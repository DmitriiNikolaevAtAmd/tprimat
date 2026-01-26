# Data Loading Fix - Summary

## Problem Identified

All frameworks except NeMo were using **pure random tokens** instead of real data, making convergence impossible:

- **DeepSpeed, FSDP, Transformers, Megatron**: Generated random token IDs with `torch.randint()`
- **NeMo**: Loaded real indexed dataset when available
- **Result**: Only NeMo showed convergence in training loss

## Solution Implemented

### 1. Created Indexed Dataset Loader (`indexed_dataset.py`)
- Reads Megatron-format binary datasets (`.bin` + `.idx` files)
- Compatible with NeMo's PreTrainingDataModule format
- Efficient random access to pre-tokenized sequences

### 2. Updated All Training Scripts

**Files Modified:**
- `train_nvd_deep.py` - DeepSpeed
- `train_nvd_fsdp.py` - FSDP
- `train_nvd_tran.py` - Transformers
- `train_nvd_mega.py` - Megatron

**Changes:**
- Each `PretrainingDataset` class now attempts to load indexed dataset
- Falls back to synthetic data if real data unavailable
- Handles padding/truncation to sequence length
- Clear logging of data source (real vs synthetic)

### 3. Added Data Verification Script (`check_data_loading.py`)

Run this to check if real data is available:
```bash
python3 check_data_loading.py
```

## Current Status

**Real Data Status:** ✗ NOT FOUND

The dataset files don't exist at `/data/llama_dataset_text_document.{bin,idx}`, so all frameworks currently use synthetic data.

## Expected Behavior After Retraining

### With Real Data (once available):
- ✓ All frameworks load real text data
- ✓ Loss converges for all frameworks
- ✓ Fair performance comparison
- ✓ Logs show: "✓ Loaded real indexed dataset"

### Without Real Data (current):
- All frameworks use synthetic random tokens
- Loss stays flat (~13-14) for all frameworks
- Still useful for throughput benchmarking
- Logs show: "⚠ Could not load real data" → "Falling back to synthetic data"

## How to Add Real Data

### Option 1: Use Existing Dataset (if available)
If you have indexed dataset files somewhere:
```bash
cp /path/to/dataset.bin /data/llama_dataset_text_document.bin
cp /path/to/dataset.idx /data/llama_dataset_text_document.idx
```

### Option 2: Create New Dataset

1. **Prepare text data** (e.g., `raw_text.txt` or `.json`)

2. **Install Megatron-LM:**
```bash
git clone https://github.com/NVIDIA/Megatron-LM
cd Megatron-LM
```

3. **Preprocess data:**
```bash
python tools/preprocess_data.py \
    --input raw_text.txt \
    --output-prefix /data/llama_dataset \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --workers 8 \
    --append-eod
```

4. **Verify:**
```bash
python3 check_data_loading.py
```

### Option 3: Use Small Test Dataset

For quick testing, create a small dataset:
```python
# Create simple test data
with open('test.txt', 'w') as f:
    f.write("The quick brown fox jumps over the lazy dog. " * 1000)

# Then preprocess as in Option 2
```

## Testing the Fix

1. **Check data loading:**
```bash
python3 check_data_loading.py
```

2. **Retrain one framework to test:**
```bash
./train_nvd_fsdp_llama.sh
```

3. **Check training logs for:**
```
✓ Loaded real indexed dataset from /data/llama_dataset_text_document
  Dataset contains XXXXX sequences
```

4. **Verify convergence in results:**
```bash
python3 nvd_compare.py
```

## Code Changes Summary

### Before (Broken):
```python
def __getitem__(self, idx):
    # Always returns random tokens
    input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_length,))
    return {'input_ids': input_ids, 'labels': input_ids.clone()}
```

### After (Fixed):
```python
def __getitem__(self, idx):
    if self.real_data_available and self.indexed_dataset is not None:
        # Load real tokenized text
        dataset_idx = idx % len(self.indexed_dataset)
        tokens = self.indexed_dataset[dataset_idx]
        # Pad/truncate to seq_length
        input_ids = self._process_tokens(tokens)
    else:
        # Fallback to synthetic
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_length,))
    
    return {'input_ids': input_ids, 'labels': input_ids.clone()}
```

## Next Steps

1. ✓ **Completed**: Fixed all dataset classes
2. ✓ **Completed**: Added indexed dataset loader
3. ✓ **Completed**: Added verification script
4. ⏳ **Pending**: Obtain or create real dataset files
5. ⏳ **Pending**: Retrain all frameworks with real data
6. ⏳ **Pending**: Verify convergence across all frameworks

## Notes

- The fix is backward compatible - works with or without real data
- Synthetic data mode still useful for pure throughput benchmarking
- All frameworks now have **identical** data loading behavior
- With real data, expect similar convergence patterns across frameworks
- Performance differences will reflect framework efficiency, not data quality
