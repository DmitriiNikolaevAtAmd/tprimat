# Script Variants Guide

This project uses two variants of training scripts for maximum flexibility:

## Variant Types

### 1. **Standard Scripts** (no suffix)
**Location:** `train_*.py`, `train_*.sh`

**Characteristics:**
- ✅ Simple, standalone implementations
- ✅ Hardcoded default values
- ✅ Quick to run, minimal setup
- ✅ Good for quick testing

**Examples:**
```bash
train_nvd_nemo_llama.py       # Standalone NeMo Llama training
train_amd_prim_llama.sh       # Standalone Primus Llama training
```

**Usage:**
```bash
# Just run directly
python3 -u train_nvd_nemo_llama.py
./train_amd_prim_llama.sh
```

---

### 2. **Config-Based Scripts** (`_env` suffix)
**Location:** `train_*_env.py`, `train_*_env.sh`

**Characteristics:**
- ✅ Load settings from `config.yaml`
- ✅ Respect environment variables
- ✅ Highly configurable
- ✅ Good for systematic benchmarking

**Examples:**
```bash
train_nvd_nemo_llama_env.py   # Config-based NeMo Llama
train_amd_prim_llama_env.sh   # Config-based Primus Llama
```

**Usage:**
```bash
# Loads config.yaml automatically
python3 -u train_nvd_nemo_llama_env.py

# Or with custom env vars
PARALLEL=tensor_parallel python3 -u train_nvd_nemo_llama_env.py
```

---

## When to Use Which?

### Use **Standard Scripts** when:
- Quick testing or debugging
- Running once with default settings
- Simplicity is preferred
- Don't need config management

### Use **Config-Based Scripts** (`_env`) when:
- Running systematic benchmarks
- Testing multiple configurations
- Need reproducible results
- Want centralized configuration

---

## Configuration Hierarchy (for `_env` scripts)

Config-based scripts follow this priority:

1. **Environment variables** (highest priority)
   ```bash
   PARALLEL=data_parallel ./train_amd_prim_llama_env.sh
   ```

2. **config.yaml** (middle priority)
   ```yaml
   methodology: "minimal_communication"
   ```

3. **Script defaults** (lowest priority)
   - Fallback values if nothing else is set

---

## Available `_env` Scripts

### NVIDIA:
```
train_nvd_nemo_llama_env.py
train_nvd_nemo_qwen_env.py
```

### AMD Primus:
```
train_amd_prim_llama_env.sh
train_amd_prim_qwen_env.sh
```

---

## Examples

### Standard Usage (Simple):
```bash
# Quick test with defaults
python3 -u train_nvd_nemo_llama.py
```

### Config-Based Usage (Advanced):
```bash
# Load from config.yaml
python3 -u train_nvd_nemo_llama_env.py

# Override with environment
PARALLEL=tensor_parallel python3 -u train_nvd_nemo_llama_env.py

# Use different config file
CONFIG_FILE=custom.yaml ./train_amd_prim_llama_env.sh
```

---

## Master Orchestrators

The master scripts (`train_amd.sh`, `train_nvd.sh`) use the **standard** (non-`_env`) versions by default for simplicity.

To use config-based versions instead, you can create custom orchestrator scripts or run them individually.

---

## Summary

| Feature | Standard | Config-Based (`_env`) |
|---------|----------|----------------------|
| **Simplicity** | ✅ Simple | ⚠️ More complex |
| **Configurability** | ❌ Limited | ✅ Highly configurable |
| **Setup required** | ❌ None | ✅ config.yaml |
| **Best for** | Quick tests | Benchmarking |
| **Default in** | Master scripts | Manual use |

**Both approaches are valid** - choose based on your needs!
