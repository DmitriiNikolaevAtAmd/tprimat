# Recommended Comparison Metrics for TensorPrimat

## Current Metrics ✅

You already have:
- ✅ **Tokens/sec/GPU** - Per-GPU efficiency (excellent primary metric)
- ✅ **Total Throughput** - System-wide tokens/sec
- ✅ **Step Time** - Training speed (avg, min, max, std dev)
- ✅ **Memory Usage** - Avg and peak memory per GPU
- ✅ **GPU Count** - Cluster size

---

## Highly Recommended Additions

### 1. Cost-Normalized Metrics 💰

**Why:** Real-world deployment decisions depend on cost, not just raw performance.

**Metrics to add:**
```python
# Tokens per Dollar-Hour
tokens_per_dollar_hour = (tokens_per_second * 3600) / cost_per_hour

# Cost to Train 1 Trillion Tokens
cost_per_trillion_tokens = (1_000_000_000_000 / tokens_per_second) * (cost_per_hour / 3600)

# Training Cost Comparison
# e.g., "AMD costs $X to train Llama 3.1 8B to completion"
```

**Cloud pricing (approximate):**
- NVIDIA H100 (8 GPUs): ~$32/hr on major clouds
- AMD MI300X (8 GPUs): ~$24/hr (varies by provider)

**Impact:** Shows true value, not just performance.

---

### 2. Model FLOPs Utilization (MFU) 📊

**Why:** Industry-standard metric for comparing training efficiency.

**Formula:**
```python
# Theoretical peak FLOPs
peak_flops_h100 = 989e12  # 989 TFLOPs (FP8)
peak_flops_mi300x = 653e12  # 653 TFLOPs (FP16)

# Model FLOPs per token (for Llama 8B)
model_flops_per_token = 6 * num_parameters  # 6 = 2 (matmul) * 3 (fwd + bwd + grad)
# For Llama 8B: 6 * 8e9 = 48e9 FLOPs per token

# Achieved FLOPs
achieved_flops = tokens_per_second * model_flops_per_token * num_gpus

# MFU
mfu = achieved_flops / (peak_flops * num_gpus)
```

**Typical values:**
- Good: 30-40% MFU
- Excellent: 40-55% MFU
- State-of-art: 55-65% MFU

**Impact:** Shows how well you're utilizing the hardware.

---

### 3. Memory Efficiency Metrics 💾

**Why:** Different memory capacities are a key differentiator.

**Metrics to add:**
```python
# Memory utilization percentage
memory_utilization = (memory_used / total_memory) * 100

# Memory per token processed
memory_per_token = memory_used / (batch_size * seq_length)

# Batch size efficiency
# "How much larger could the batch be?"
potential_batch_size = (total_memory * 0.9) / (memory_per_token * seq_length)
```

**Impact:** Shows headroom for larger models/batches.

---

### 4. Power Efficiency ⚡

**Why:** Data centers care about power costs and carbon footprint.

**Metrics to add:**
```python
# Typical TDP (Thermal Design Power)
tdp_h100 = 700  # watts
tdp_mi300x = 750  # watts

# Tokens per watt-hour
tokens_per_watt_hour = tokens_per_second * 3600 / (tdp * num_gpus)

# Energy cost to train 1T tokens
energy_cost = (1e12 / tokens_per_second) * (tdp * num_gpus / 1000) * electricity_rate
# electricity_rate ≈ $0.10/kWh (US average)

# Carbon footprint (optional)
co2_per_kwh = 0.5  # kg CO2 per kWh (varies by region)
carbon_footprint = energy_cost * co2_per_kwh
```

**Impact:** Shows environmental and operational cost differences.

---

### 5. Communication Efficiency 🔄

**Why:** Critical for understanding parallelism overhead.

**Metrics to add:**
```python
# Communication overhead (inferred from TP)
# For TP=1: minimal communication
# For TP=4: significant all-reduce operations

# Scaling efficiency
ideal_throughput = tokens_per_gpu * num_gpus
actual_throughput = measured_tokens_per_second
scaling_efficiency = (actual_throughput / ideal_throughput) * 100

# Effective bandwidth
# For TP=4: need to measure inter-GPU traffic
# NVLink: 900 GB/s bidirectional
# Infinity Fabric: 896 GB/s bidirectional
```

**Impact:** Explains performance differences from parallelism.

---

### 6. Practical Training Metrics 🎯

**Why:** Users want to know "how long to train my model?"

**Metrics to add:**
```python
# Time to train 1 Trillion tokens
time_to_1T_tokens_hours = (1e12 / tokens_per_second) / 3600

# Full training estimates (Llama 3.1 8B ≈ 15T tokens)
time_to_full_training_days = (15e12 / tokens_per_second) / (3600 * 24)

# Samples per second
samples_per_second = tokens_per_second / seq_length

# Epochs per day (for fixed dataset)
dataset_tokens = 100e9  # example: 100B token dataset
epochs_per_day = (tokens_per_second * 86400) / dataset_tokens
```

**Impact:** Helps with project planning and deadlines.

---

### 7. Stability and Reliability Metrics 📈

**Why:** Production systems need consistent performance.

**Metrics to add:**
```python
# Coefficient of variation (lower is better)
cv_step_time = std_dev_step_time / mean_step_time

# Performance stability score
stability_score = 1.0 / (1.0 + cv_step_time)

# Outlier analysis
# Detect and count steps > 2 std dev from mean
outliers = sum(1 for t in step_times if abs(t - mean) > 2 * std_dev)
outlier_percentage = (outliers / len(step_times)) * 100
```

**Impact:** Shows reliability for long training runs.

---

## Priority Ranking

### Must-Have (Add First) 🌟
1. **Cost per Trillion Tokens** - Most actionable metric
2. **MFU (Model FLOPs Utilization)** - Industry standard
3. **Memory Utilization %** - Shows headroom
4. **Time to Train 1T Tokens** - Practical planning

### Nice-to-Have (Add Second) ⭐
5. **Tokens per Dollar-Hour** - ROI metric
6. **Power Efficiency** - Environmental/cost
7. **Scaling Efficiency** - Explains parallelism overhead

### Optional (Add If Needed) ✨
8. **Carbon Footprint** - For ESG reporting
9. **Stability Score** - For production systems
10. **Peak vs Sustained Performance** - Burst vs steady-state

---

## Implementation Priority

### Phase 1: Basic Economic Metrics
```python
# Easy to implement, high value
- tokens_per_dollar_hour
- cost_to_train_1T_tokens
- time_to_train_1T_tokens
- memory_utilization_percent
```

### Phase 2: Technical Efficiency
```python
# Requires model knowledge
- mfu (model_flops_utilization)
- scaling_efficiency
- tokens_per_watt
```

### Phase 3: Advanced Analysis
```python
# Requires additional measurement
- communication_overhead
- carbon_footprint
- stability_scores
```

---

## Visualization Recommendations

### Add to Comparison Plot

**New Charts:**
1. **Cost Efficiency**: Bar chart of $/1T tokens
2. **MFU**: Bar chart showing % of theoretical peak
3. **Memory Efficiency**: Stacked bar (used vs available)
4. **Time to Train**: Horizontal bar chart (hours to 1T tokens)

**Enhanced Summary Box:**
```
Performance Summary
══════════════════════════════════════════
Efficiency Metrics:
  MFU: NVIDIA 42%, AMD 38%
  Memory Used: NVIDIA 27%, AMD 61%
  
Cost Metrics:
  $/1T tokens: NVIDIA $45, AMD $28
  Training Time (1T): NVIDIA 25hrs, AMD 18hrs
  
Winner by Metric:
  • Raw Performance: AMD 6.34x faster
  • Cost Efficiency: AMD 1.6x cheaper
  • MFU: NVIDIA 1.1x better
  • Overall: AMD (best value)
══════════════════════════════════════════
```

---

## Code Implementation Helpers

See `enhanced_metrics.py` for:
- Automatic MFU calculation
- Cost metric computation
- Enhanced comparison report
- Updated visualization

---

## Example Output

```markdown
## Extended Comparison

### Cost Analysis
| Metric | NVIDIA H100 | AMD MI300X | Winner |
|--------|-------------|------------|--------|
| Cost per Hour (8 GPUs) | $32 | $24 | AMD |
| Tokens/$ | 345 | 4,454 | AMD 12.9x |
| Cost to 1T tokens | $81 | $6.24 | AMD 13x |
| Days to train Llama 8B | 17.4 | 2.7 | AMD 6.4x |

### Technical Efficiency
| Metric | NVIDIA H100 | AMD MI300X | Winner |
|--------|-------------|------------|--------|
| MFU | 42% | 38% | NVIDIA |
| Memory Utilization | 27% | 61% | AMD |
| Scaling Efficiency | 87% | 94% | AMD |
| Power per Token | 1.8 J/token | 0.4 J/token | AMD 4.5x |
```

---

*Last Updated: January 2026*
