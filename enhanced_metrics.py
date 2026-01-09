#!/usr/bin/env python3
"""
Enhanced Metrics Module for TensorPrimat

Provides additional comparison metrics:
- Cost efficiency (tokens per dollar)
- Model FLOPs Utilization (MFU)
- Power efficiency
- Memory efficiency
- Training time estimates
"""

from typing import Dict, Tuple


# Hardware specifications
GPU_SPECS = {
    "h100": {
        "peak_tflops_fp8": 989,
        "peak_tflops_fp16": 494,
        "tdp_watts": 700,
        "memory_gb": 80,
        "cost_per_hour_8gpu": 32.0,  # Approximate cloud pricing
        "interconnect": "NVLink 4.0 (900 GB/s)",
    },
    "mi300x": {
        "peak_tflops_fp8": 653,  # FP8 (if supported)
        "peak_tflops_fp16": 653,
        "tdp_watts": 750,
        "memory_gb": 192,
        "cost_per_hour_8gpu": 24.0,  # Approximate cloud pricing
        "interconnect": "Infinity Fabric (896 GB/s)",
    }
}

# Model specifications (FLOPs per token)
MODEL_PARAMS = {
    "llama": {
        "8b": 8e9,
        "70b": 70e9,
    },
    "qwen": {
        "7b": 7.6e9,
    }
}


def calculate_model_flops_per_token(num_parameters: float) -> float:
    """
    Calculate FLOPs required per token for a transformer model.
    
    Formula: 6 * N where N is number of parameters
    6 = 2 (forward matmul) * 3 (forward + backward + optimizer)
    
    Args:
        num_parameters: Number of model parameters
        
    Returns:
        FLOPs per token
    """
    return 6 * num_parameters


def calculate_mfu(
    tokens_per_second: float,
    num_gpus: int,
    num_parameters: float,
    peak_tflops: float
) -> Tuple[float, float]:
    """
    Calculate Model FLOPs Utilization (MFU).
    
    MFU = (Achieved FLOPs) / (Peak FLOPs)
    
    Args:
        tokens_per_second: Training throughput
        num_gpus: Number of GPUs
        num_parameters: Model parameter count
        peak_tflops: Peak TFLOPs per GPU
        
    Returns:
        (mfu_percentage, achieved_tflops)
    """
    # FLOPs per token
    flops_per_token = calculate_model_flops_per_token(num_parameters)
    
    # Total achieved FLOPs
    achieved_flops = tokens_per_second * flops_per_token * num_gpus
    achieved_tflops = achieved_flops / 1e12
    
    # Peak system FLOPs
    peak_flops = peak_tflops * num_gpus * 1e12
    
    # MFU
    mfu = (achieved_flops / peak_flops) * 100
    
    return mfu, achieved_tflops


def calculate_cost_metrics(
    tokens_per_second: float,
    cost_per_hour: float
) -> Dict[str, float]:
    """
    Calculate cost-related metrics.
    
    Args:
        tokens_per_second: Training throughput
        cost_per_hour: Cost per hour for the cluster
        
    Returns:
        Dictionary with cost metrics
    """
    # Tokens per dollar
    tokens_per_dollar = (tokens_per_second * 3600) / cost_per_hour
    
    # Cost to train 1 trillion tokens
    cost_per_trillion = (1e12 / tokens_per_second) * (cost_per_hour / 3600)
    
    # Hours to train 1 trillion tokens
    hours_per_trillion = 1e12 / (tokens_per_second * 3600)
    
    return {
        "tokens_per_dollar": tokens_per_dollar,
        "cost_per_trillion_tokens": cost_per_trillion,
        "hours_per_trillion_tokens": hours_per_trillion,
        "cost_per_hour": cost_per_hour,
    }


def calculate_power_metrics(
    tokens_per_second: float,
    num_gpus: int,
    tdp_watts: float,
    electricity_cost_per_kwh: float = 0.10
) -> Dict[str, float]:
    """
    Calculate power and energy efficiency metrics.
    
    Args:
        tokens_per_second: Training throughput
        num_gpus: Number of GPUs
        tdp_watts: Thermal Design Power per GPU
        electricity_cost_per_kwh: Electricity cost ($/kWh)
        
    Returns:
        Dictionary with power metrics
    """
    total_power_kw = (tdp_watts * num_gpus) / 1000
    
    # Tokens per watt-hour
    tokens_per_watt_hour = tokens_per_second * 3600 / (tdp_watts * num_gpus)
    
    # Energy to train 1T tokens (kWh)
    kwh_per_trillion = (1e12 / tokens_per_second) * total_power_kw / 3600
    
    # Energy cost for 1T tokens
    energy_cost_per_trillion = kwh_per_trillion * electricity_cost_per_kwh
    
    # CO2 emissions (kg) - US average: 0.5 kg CO2/kWh
    co2_kg_per_trillion = kwh_per_trillion * 0.5
    
    return {
        "tokens_per_watt_hour": tokens_per_watt_hour,
        "kwh_per_trillion_tokens": kwh_per_trillion,
        "energy_cost_per_trillion": energy_cost_per_trillion,
        "co2_kg_per_trillion_tokens": co2_kg_per_trillion,
        "total_power_kw": total_power_kw,
    }


def calculate_memory_metrics(
    memory_used_gb: float,
    total_memory_gb: float,
    batch_size: int,
    seq_length: int
) -> Dict[str, float]:
    """
    Calculate memory efficiency metrics.
    
    Args:
        memory_used_gb: Memory actually used
        total_memory_gb: Total available memory
        batch_size: Batch size per GPU
        seq_length: Sequence length
        
    Returns:
        Dictionary with memory metrics
    """
    # Memory utilization
    memory_utilization = (memory_used_gb / total_memory_gb) * 100
    
    # Memory per token
    tokens_per_batch = batch_size * seq_length
    memory_per_million_tokens = (memory_used_gb / tokens_per_batch) * 1e6
    
    # Headroom
    available_headroom_gb = total_memory_gb - memory_used_gb
    headroom_percentage = (available_headroom_gb / total_memory_gb) * 100
    
    # Potential batch size with 90% memory usage
    potential_batch_size = int((total_memory_gb * 0.9) / (memory_used_gb / batch_size))
    
    return {
        "memory_utilization_percent": memory_utilization,
        "memory_per_million_tokens_gb": memory_per_million_tokens,
        "available_headroom_gb": available_headroom_gb,
        "headroom_percent": headroom_percentage,
        "potential_batch_size": potential_batch_size,
    }


def calculate_training_estimates(
    tokens_per_second: float,
    model_name: str = "llama",
    model_size: str = "8b"
) -> Dict[str, float]:
    """
    Calculate practical training time estimates.
    
    Args:
        tokens_per_second: Training throughput
        model_name: Model name (llama, qwen)
        model_size: Model size (8b, 70b, 7b, etc.)
        
    Returns:
        Dictionary with training estimates
    """
    # Standard training token counts
    training_tokens = {
        "1T": 1e12,
        "llama_8b_full": 15e12,  # Llama 3.1 8B trained on ~15T tokens
        "llama_70b_full": 15e12,
    }
    
    results = {}
    
    for name, tokens in training_tokens.items():
        hours = tokens / (tokens_per_second * 3600)
        days = hours / 24
        
        results[f"{name}_hours"] = hours
        results[f"{name}_days"] = days
    
    # Samples per second
    results["samples_per_second"] = tokens_per_second / 2048  # Assuming 2048 seq len
    
    return results


def calculate_scaling_efficiency(
    tokens_per_gpu: float,
    num_gpus: int,
    total_throughput: float
) -> Dict[str, float]:
    """
    Calculate scaling efficiency metrics.
    
    Args:
        tokens_per_gpu: Per-GPU throughput
        num_gpus: Number of GPUs
        total_throughput: Total system throughput
        
    Returns:
        Dictionary with scaling metrics
    """
    # Ideal throughput (perfect linear scaling)
    ideal_throughput = tokens_per_gpu * num_gpus
    
    # Scaling efficiency
    scaling_efficiency = (total_throughput / ideal_throughput) * 100
    
    # Communication overhead (percentage lost to communication)
    communication_overhead = 100 - scaling_efficiency
    
    return {
        "ideal_throughput": ideal_throughput,
        "scaling_efficiency_percent": scaling_efficiency,
        "communication_overhead_percent": communication_overhead,
    }


def get_enhanced_metrics(
    benchmark_data: Dict,
    gpu_type: str = "h100",
    model_name: str = "llama",
    model_size: str = "8b"
) -> Dict:
    """
    Calculate all enhanced metrics for a benchmark result.
    
    Args:
        benchmark_data: Benchmark JSON data
        gpu_type: GPU type (h100, mi300x)
        model_name: Model name
        model_size: Model size
        
    Returns:
        Dictionary with all enhanced metrics
    """
    # Extract basic metrics
    perf = benchmark_data['performance_metrics']
    gpu_info = benchmark_data['gpu_info']
    config = benchmark_data['training_config']
    
    tokens_per_second = perf.get('tokens_per_second', 0)
    tokens_per_gpu = perf.get('tokens_per_second_per_gpu', 0)
    num_gpus = gpu_info.get('device_count', config.get('num_gpus', 8))
    if isinstance(num_gpus, str):
        num_gpus = 8  # Default
    
    # Get GPU specs
    specs = GPU_SPECS.get(gpu_type.lower(), GPU_SPECS["h100"])
    
    # Get model params
    num_parameters = MODEL_PARAMS.get(model_name, {}).get(model_size, 8e9)
    
    # Calculate all metrics
    enhanced = {}
    
    # MFU
    software_stack = gpu_info.get('software_stack', 'cuda')
    precision_key = 'peak_tflops_fp8' if 'fp8' in str(benchmark_data).lower() else 'peak_tflops_fp16'
    peak_tflops = specs[precision_key]
    
    mfu, achieved_tflops = calculate_mfu(tokens_per_second, num_gpus, num_parameters, peak_tflops)
    enhanced['mfu_percent'] = mfu
    enhanced['achieved_tflops'] = achieved_tflops
    enhanced['peak_tflops'] = peak_tflops * num_gpus
    
    # Cost metrics
    cost_metrics = calculate_cost_metrics(tokens_per_second, specs['cost_per_hour_8gpu'])
    enhanced.update(cost_metrics)
    
    # Power metrics
    power_metrics = calculate_power_metrics(tokens_per_second, num_gpus, specs['tdp_watts'])
    enhanced.update(power_metrics)
    
    # Memory metrics
    memory_metrics_data = benchmark_data.get('memory_metrics', {})
    if memory_metrics_data and isinstance(gpu_info.get('total_memory_gb'), (int, float)):
        memory_used = memory_metrics_data.get('avg_memory_allocated_gb', 0)
        total_memory = gpu_info['total_memory_gb']
        batch_size = config.get('global_batch_size', 128) // num_gpus
        seq_length = config.get('sequence_length', 2048)
        
        memory_metrics = calculate_memory_metrics(memory_used, total_memory, batch_size, seq_length)
        enhanced.update(memory_metrics)
    
    # Training estimates
    training_estimates = calculate_training_estimates(tokens_per_second, model_name, model_size)
    enhanced.update(training_estimates)
    
    # Scaling efficiency
    if tokens_per_gpu > 0:
        scaling_metrics = calculate_scaling_efficiency(tokens_per_gpu, num_gpus, tokens_per_second)
        enhanced.update(scaling_metrics)
    
    return enhanced


if __name__ == "__main__":
    # Example usage
    print("Enhanced Metrics Calculator")
    print("=" * 60)
    
    # Example for NVIDIA H100
    tokens_per_sec = 11045
    num_gpus = 8
    
    print(f"\nExample: NVIDIA H100 (8 GPUs)")
    print(f"Throughput: {tokens_per_sec:,} tokens/sec")
    
    # MFU
    mfu, achieved_tflops = calculate_mfu(tokens_per_sec, num_gpus, 8e9, 989)
    print(f"\nMFU: {mfu:.1f}%")
    print(f"Achieved: {achieved_tflops:.1f} TFLOPs")
    
    # Cost
    cost_metrics = calculate_cost_metrics(tokens_per_sec, 32.0)
    print(f"\nCost to train 1T tokens: ${cost_metrics['cost_per_trillion_tokens']:.2f}")
    print(f"Time to train 1T tokens: {cost_metrics['hours_per_trillion_tokens']:.1f} hours")
    
    # Power
    power_metrics = calculate_power_metrics(tokens_per_sec, num_gpus, 700)
    print(f"\nEnergy: {power_metrics['kwh_per_trillion_tokens']:.1f} kWh per 1T tokens")
    print(f"CO2: {power_metrics['co2_kg_per_trillion_tokens']:.1f} kg per 1T tokens")
