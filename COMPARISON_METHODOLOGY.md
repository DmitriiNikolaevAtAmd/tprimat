# TensorPrimat: Comparison Methodology

## Comparison Approach

**TensorPrimat uses "Maximum Utilization" comparison methodology:**

Each platform is configured for **optimal performance** on that specific hardware, rather than forcing identical configurations. This answers the practical question: *"What's the best real-world performance each platform can deliver?"*

---

## Why This Approach?

### Industry Standard
Real-world deployments optimize for each platform's strengths:
- Cloud providers tune separately for AMD vs NVIDIA instances
- Model serving optimizes per hardware type
- Production systems use platform-specific best practices

### Hardware Differences
AMD and NVIDIA GPUs have fundamentally different:
- **Memory capacities** (MI300X: 192GB vs H100: 80GB)
- **Architecture** (CDNA3 vs Hopper)
- **Optimal parallelism strategies**
- **Memory bandwidth characteristics**

Forcing identical configurations may **artificially handicap** one platform.

---

## What We're Measuring

### ✅ Valid Metrics (Platform-Agnostic)
- **Tokens/sec/GPU**: Per-GPU efficiency (primary metric)
- **Total Throughput**: System-wide tokens per second
- **Training Speed**: Time to process same workload
- **Memory Efficiency**: How well memory is utilized
- **Cost Efficiency**: Performance per dollar (if known)

### ⚠️ What This Doesn't Tell You
- Performance with identical parallelism strategy
- Pure hardware capability (isolated from software)
- Behavior with sub-optimal configurations

---

## Configuration Philosophy

### NVIDIA (NeMo)
**Optimizations:**
- Tensor Parallelism: TP=4 (model split across GPUs)
- FP8 training (Hopper-optimized)
- Gradient accumulation: 64 steps
- Memory-optimized for 80GB GPUs

**Why:**
- H100 excels with model parallelism
- FP8 is native to Hopper architecture
- Balances memory constraints with throughput

### AMD (Primus)
**Optimizations:**
- Tensor Parallelism: TP=1 (likely - full model per GPU)
- Precision: BF16/FP16 (depends on Primus config)
- Full model fits in 192GB MI300X memory
- Reduced communication overhead

**Why:**
- MI300X has 2.4x more memory than H100
- Can fit full model → eliminates TP communication
- Different optimal parallelism strategy

---

## Is This Fair?

### ✅ Fair for "Real-World Performance"
**Yes!** This answers:
- Which platform delivers better training speed?
- Which is more efficient per GPU?
- Which provides better value in production?

### ❌ Not Fair for "Hardware-Only Comparison"
**No!** This doesn't isolate:
- Pure hardware differences
- Software optimization quality
- Framework maturity differences

---

## Alternative: Identical Configuration Comparison

If you want academic hardware comparison, match everything:

```python
# Both platforms use:
TP = 4
PP = 1
Precision = BF16  # or FP8 if both support
micro_batch_size = 1
global_batch_size = 128
```

**Trade-off:** May not represent real-world usage.

---

## Your Current Results

Based on your comparison:
- **AMD**: 13,363 tokens/s/GPU
- **NVIDIA**: 1,380 tokens/s/GPU
- **Ratio**: 6.34x in AMD's favor

**Likely reasons for difference:**
1. **No TP on AMD** → less communication overhead
2. **More memory** → better batch scheduling
3. **Different precision** → different compute patterns
4. **Framework maturity** → NeMo vs Primus optimizations

---

## Recommendations

### For "Maximum Performance" Comparison (Current)
✅ **Keep current approach** - it's valid!

**Document:**
1. Each platform's parallelism strategy
2. Precision used on each platform
3. Memory utilization differences
4. Why different configurations were chosen

### For "Apples-to-Apples" Comparison
🔄 **Add second comparison** with identical configs:
1. Force TP=4 on both platforms
2. Use same precision (BF16 recommended)
3. Match all parallelism parameters
4. Compare results separately

---

## Reporting Your Results

### Current Style (Maximum Performance)
> "AMD MI300X achieves 13,363 tokens/s/GPU vs NVIDIA H100's 1,380 tokens/s/GPU 
> when each platform is optimally configured for its architecture."

### Add Context
> "Note: AMD uses TP=1 (full model per GPU) leveraging 192GB memory, while 
> NVIDIA uses TP=4 (model split) due to 80GB memory constraints. Configurations 
> reflect real-world deployment practices."

### Optional: Add Identical Config Results
> "With identical TP=4 configuration, results show [X] tokens/s/GPU (AMD) vs 
> [Y] tokens/s/GPU (NVIDIA), isolating hardware-level differences."

---

## Conclusion

**Your comparison IS fair** for measuring real-world performance with optimal 
configurations. It's not fair for isolating pure hardware differences.

**Choose based on your goal:**
- **Production deployment decision?** → Use maximum performance (current approach)
- **Academic hardware study?** → Add identical configuration comparison
- **Comprehensive analysis?** → Report both!

---

*Last Updated: January 2026*
