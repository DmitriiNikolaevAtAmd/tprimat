# TPrimat Naming Convention

## Script Naming Structure

All training scripts follow a consistent naming pattern to indicate platform compatibility:

```
run_<platform>_<framework>_<model>.{sh,py}
```

### Platform Prefixes

| Prefix | Platform | Example |
|--------|----------|---------|
| `nvd_` | NVIDIA-specific | `train_nvd_nemo_llama.py` |
| `amd_` | AMD-specific | `train_amd_nemo_llama.sh` |
| (none) | Platform-agnostic | `train_all_tran.py` |

### Complete Script Categories

#### NVIDIA Scripts (`nvd_` prefix)

**Master Scripts:**
- `train_nvd.sh` - Run all NVIDIA frameworks
- `train_nvd_docker.sh` - NVIDIA Docker launcher

**Framework Scripts:**
- `train_nvd_deep.py` - DeepSpeed ZeRO-3
- `train_nvd_mega.py` - Megatron-LM (Python)
- `train_nvd_mega.sh` - Megatron-LM (Shell wrapper)
- `train_nvd_nemo_llama.py` - NVIDIA NeMo

#### AMD Scripts (`amd_` prefix)

**Master Scripts:**
- `train_amd.sh` - Run all AMD frameworks
- `train_amd.sh` - AMD with config loading
- `train_amd_docker.sh` - AMD Docker launcher

**Framework Scripts (per model):**
- `train_amd_nemo_llama.sh` / `train_amd_nemo_qwen.sh` - NeMo with ROCm
- `train_amd_tran_llama.sh` / `train_amd_tran_qwen.sh` - Transformers
- `train_amd_deep_llama.sh` / `train_amd_deep_qwen.sh` - DeepSpeed
- `train_amd_fsdp_llama.sh` / `train_amd_fsdp_qwen.sh` - PyTorch FSDP
- `train_amd_mgds_llama.sh` / `train_amd_mgds_qwen.sh` - Megatron-DeepSpeed

**Primus (AMD-optimized):**
- `train_prim.sh` - Run all Primus training
- `train_amd_prim_llama.sh` / `train_amd_prim_qwen.sh` - Primus per model
- `train_prim.sh` - Primus plain wrapper

#### Platform-Agnostic Scripts (no prefix)

These scripts work on both NVIDIA and AMD GPUs:

- `train_all_tran.py` - HuggingFace Transformers (detects platform)
- `train_all_fsdp.py` - PyTorch FSDP (detects platform)
- `train_all_mgds.py` - Megatron-DeepSpeed (detects platform)

### Output File Naming

Output files follow the pattern:

```
train_<framework>_<model>.json
```

**Examples:**
- `train_nemo_llama.json` - NeMo + Llama 3.1 8B
- `train_deep_qwen.json` - DeepSpeed + Qwen 2.5 7B
- `train_amd_prim_llama.json` - Primus + Llama 3.1 8B

**Framework Identifiers:**

| Framework | Identifier |
|-----------|-----------|
| NeMo | `nemo` |
| Megatron-LM | `mega` |
| DeepSpeed | `deep` |
| Transformers | `tran` |
| FSDP | `fsdp` |
| Primus | `prim` |
| Megatron-DeepSpeed | `mgds` |

### Why This Convention?

1. **Clear Platform Identification**: Instantly see which platform a script targets
2. **Consistent Sorting**: Scripts group by platform when sorted alphabetically
3. **Easy Filtering**: Simple glob patterns work everywhere
   - `run_nvd_*.py` - All NVIDIA Python scripts
   - `run_amd_*.sh` - All AMD shell scripts
   - `run_*.py` - All plain Python implementations
4. **Self-Documenting**: File names tell you what they do

### Usage Examples

```bash
# Run all NVIDIA frameworks
./train_nvd.sh

# Run all AMD frameworks
./train_amd.sh

# Run specific NVIDIA framework
python3 train_nvd_nemo_llama.py

# Run specific AMD framework
./train_amd_nemo_llama.sh

# Run platform-agnostic (auto-detects)
python3 train_all_tran.py llama
```

### Migration from Old Naming

| Old Name | New Name | Reason |
|----------|----------|--------|
| `train_mega.py` | `train_nvd_mega.py` | NVIDIA-specific |
| `train_mega.sh` | `train_nvd_mega.sh` | NVIDIA-specific |
| `train_nemo.py` | `train_nvd_nemo_llama.py` | NVIDIA-specific |
| `train_all_tran.py` | (unchanged) | Platform-agnostic |
| `train_deep.py` | `train_nvd_deep.py` | Already had prefix |

### Quick Reference

**"I want to run on NVIDIA H100:"**
- Use scripts with `nvd_` prefix
- Or platform-agnostic scripts (auto-detect)

**"I want to run on AMD MI300X:"**
- Use scripts with `amd_` prefix
- Or platform-agnostic scripts (auto-detect)
- Or Primus scripts (`prim_` prefix) for best performance

**"I want to run on both platforms:"**
- Use platform-agnostic scripts (no prefix)
- They auto-detect and optimize for your platform
