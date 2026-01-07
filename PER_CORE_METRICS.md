# Per-GPU-Core Throughput Metrics

## Overview

The benchmarking system now includes **throughput per GPU core** metrics, which normalize performance across different GPU architectures. This is essential for fair comparison between AMD and NVIDIA GPUs that have different core counts.

## What Was Added

### 1. GPU Core Count Detection

The system automatically detects GPU core counts for common models:

#### NVIDIA GPUs (CUDA Cores)
- **H100 SXM5**: 16,896 cores (80GB/94GB)
- **H100 PCIe**: 14,592 cores
- **A100**: 6,912 cores (40GB/80GB)
- **V100**: 5,120 cores
- **A40/A30**: 10,752 cores
- **A10**: 9,216 cores
- **RTX 4090**: 16,384 cores
- **RTX 3090**: 10,496 cores
- **RTX 3080**: 8,704 cores

#### AMD GPUs (Stream Processors)
- **MI300X/A**: 19,456 SPs per GCD
- **MI250X**: 28,160 SPs total (2 GCDs × 14,080)
- **MI250**: 26,624 SPs total (2 GCDs × 13,312)
- **MI210**: 13,312 SPs
- **MI100**: 7,680 SPs

**Note**: For MI250 series, the count includes both GCDs (Graphics Compute Dies) since training utilizes both dies.

### 2. Per-Core Throughput Calculation

```
Throughput per Core = Total Throughput (steps/s) / Number of GPU Cores
```

This metric shows how efficiently each GPU core is being utilized.

### 3. Enhanced Output

#### During Training
```
============================================================
BENCHMARK COMPLETE - Platform: CUDA
============================================================
Total Steps: 10
Total Time: 12.45s
Avg Step Time: 1.245s
Throughput: 0.803 steps/s
GPU Cores: 16,896                                    ← NEW
Throughput/Core: 0.000048 steps/s/core              ← NEW
Avg Memory: 45.67GB
Peak Memory: 45.89GB
============================================================
```

#### Comparison Output
```
============================================================
AMD vs NVIDIA GPU COMPARISON
============================================================

NVIDIA GPU (NVIDIA H100 SXM5 80GB):
  GPU Cores:       16,896                            ← NEW
  Avg Step Time:   1.245s
  Throughput:      0.803 steps/s
  Throughput/Core: 0.000048 steps/s/core            ← NEW
  Peak Memory:     45.89GB

AMD GPU (AMD Instinct MI250X):
  GPU Cores:       28,160                            ← NEW
  Avg Step Time:   1.567s
  Throughput:      0.638 steps/s
  Throughput/Core: 0.000023 steps/s/core            ← NEW
  Peak Memory:     48.23GB

Result:
  NVIDIA is 1.26x faster
  Throughput ratio (NVIDIA/AMD): 1.26x
  Per-core efficiency (NVIDIA/AMD): 2.10x           ← NEW
============================================================
```

### 4. Enhanced Visualizations

The comparison plot now includes:
- **2x3 grid** (instead of 2x2) when per-core data is available
- **New chart**: Throughput per GPU Core (bar chart)
- **New chart**: GPU Core Count comparison (bar chart)

Layout:
```
┌─────────────────┬─────────────────┬─────────────────┐
│ Avg Step Time   │  Throughput     │ Throughput/Core │
│   (bar chart)   │  (bar chart)    │  (bar chart)    │
├─────────────────┼─────────────────┼─────────────────┤
│ Memory Usage    │ Step Time Dist  │  Core Count     │
│ (grouped bars)  │  (line plot)    │  (bar chart)    │
└─────────────────┴─────────────────┴─────────────────┘
```

### 5. Enhanced Reports

The markdown report now includes:

#### Hardware Configuration
```markdown
### NVIDIA GPU
- **Device**: NVIDIA H100 SXM5 80GB
- **GPU Cores**: 16,896                              ← NEW
- **Total Memory**: 80.00 GB
- **CUDA Version**: 12.1
```

#### Throughput Table
```markdown
| Platform | Steps/Second | Steps/Second/Core |
|----------|--------------|-------------------|
| NVIDIA   | 0.803        | 0.000048          |
| AMD      | 0.638        | 0.000023          |
```

#### Per-Core Efficiency Section (NEW)
```markdown
### Per-Core Efficiency
- **NVIDIA per-core throughput**: 0.000048 steps/s/core
- **AMD per-core throughput**: 0.000023 steps/s/core
- **Per-core ratio (NVIDIA/AMD)**: 2.10x
- **More efficient per core**: NVIDIA
```

## Why This Matters

### 1. Fair Comparison

Raw throughput can be misleading when comparing GPUs with different core counts. For example:

- **GPU A**: 0.803 steps/s with 16,896 cores
- **GPU B**: 0.638 steps/s with 28,160 cores

Per-core metrics reveal:
- **GPU A**: 0.000048 steps/s/core (more efficient)
- **GPU B**: 0.000023 steps/s/core

GPU A is actually **2.1x more efficient per core**, even though its absolute throughput advantage is only 1.26x.

### 2. Architecture Insights

Per-core metrics help identify:
- **Software optimization**: How well code utilizes available cores
- **Architecture efficiency**: Core-for-core performance differences
- **Scaling**: How workloads scale with core count

### 3. Cost Analysis

When considering cost per performance:
```
Cost per Core Efficiency = GPU Price / (Throughput per Core × Core Count)
```

## Example Comparison

### H100 vs MI250X for Llama 3.1 8B Training

#### Raw Performance
```
H100:    0.803 steps/s (1.26x faster)
MI250X:  0.638 steps/s
```

#### Per-Core Performance
```
H100:    0.000048 steps/s/core (16,896 cores)
MI250X:  0.000023 steps/s/core (28,160 cores)
Per-core ratio: 2.10x (H100 more efficient)
```

#### Insights
1. H100 is 1.26x faster in absolute terms
2. H100 is 2.10x more efficient per core
3. MI250X has 1.67x more cores but lower per-core efficiency
4. Training workload may be limited by memory bandwidth or other factors, not just core count

## Technical Details

### Core Count Detection

The system uses a lookup table matching GPU model names to core counts. If the GPU is not in the table, it attempts to use PyTorch device properties:

```python
# For NVIDIA: multiply SM count by cores per SM (default 128)
if hasattr(device_props, 'multi_processor_count'):
    cores = device_props.multi_processor_count * 128
```

### Handling Unknown GPUs

If core count cannot be determined:
- Core count is set to 0
- Per-core metrics are not calculated or displayed
- System still works normally for absolute performance metrics

### AMD Multi-Die GPUs

For AMD MI250/MI250X (which have 2 GCDs):
- Total core count includes both dies
- This is correct since training utilizes both dies

## Usage

No changes needed! The new metrics are automatically collected and displayed:

```bash
# Same commands as before
./run_benchmark.sh llama          # On both platforms
python3 compare_results.py        # Compare with new metrics
```

## Updated Files

1. **`benchmark_utils.py`** - Added core count detection and per-core calculation
2. **`compare_results.py`** - Enhanced plots and reports with per-core metrics

## Interpreting Results

### Per-Core Throughput

**Higher is better**. Typical values:
- **Modern GPUs (H100, MI300)**: 0.00004 - 0.00007 steps/s/core
- **Previous gen (A100, MI250)**: 0.00003 - 0.00005 steps/s/core
- **Older GPUs (V100, MI100)**: 0.00002 - 0.00004 steps/s/core

*Note: Actual values depend heavily on the model, batch size, and parallelism strategy.*

### Per-Core Efficiency Ratio

- **> 1.0**: NVIDIA more efficient per core
- **< 1.0**: AMD more efficient per core
- **≈ 1.0**: Similar per-core efficiency

### When Per-Core Metrics Are Most Useful

1. **Comparing different GPU architectures**
2. **Evaluating cost-effectiveness**
3. **Understanding software optimization**
4. **Planning hardware upgrades**
5. **Analyzing scaling behavior**

### When Absolute Throughput Matters More

1. **Time-to-solution**: How fast can I finish training?
2. **Meeting deadlines**: Need X steps in Y hours
3. **Fixed hardware**: Can't change GPU type

## Example Scenarios

### Scenario 1: Budget Optimization

```
Option A (H100):
- Price: $30,000
- Throughput: 0.803 steps/s
- Cores: 16,896
- Per-core: 0.000048 steps/s/core
- Cost per core: $1.78

Option B (MI250X):
- Price: $15,000
- Throughput: 0.638 steps/s
- Cores: 28,160
- Per-core: 0.000023 steps/s/core
- Cost per core: $0.53

Analysis:
- H100: Better absolute performance, higher per-core efficiency
- MI250X: Better cost per core, more cores
- Decision depends on budget vs time-to-solution requirements
```

### Scenario 2: Software Optimization

```
Before optimization:
- H100: 0.000030 steps/s/core (53% of expected)
- MI250X: 0.000015 steps/s/core (65% of expected)

After optimization:
- H100: 0.000048 steps/s/core (85% of expected)
- MI250X: 0.000023 steps/s/core (82% of expected)

Insight: H100 benefits more from optimization (60% improvement vs 53%)
```

## Limitations

1. **Core count is approximate**: Different core types (CUDA vs Stream Processors) aren't directly comparable
2. **Not all cores are equal**: Architecture differences mean 1 CUDA core ≠ 1 Stream Processor
3. **Workload dependent**: Per-core efficiency varies significantly with workload type
4. **Memory bound workloads**: May not scale with core count

## Best Practices

1. ✅ Use per-core metrics for architecture comparisons
2. ✅ Use absolute throughput for time-to-solution
3. ✅ Consider both metrics together
4. ✅ Run multiple benchmarks for statistical significance
5. ✅ Document your specific workload characteristics

## Future Enhancements

Potential additions:
- [ ] Memory bandwidth per core
- [ ] FLOPS per core
- [ ] Power efficiency per core
- [ ] Core utilization percentage
- [ ] Automatic core count detection for more GPUs

---

**Added**: January 5, 2026  
**Requires**: No additional dependencies  
**Works with**: All existing benchmarking features  

