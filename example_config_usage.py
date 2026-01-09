#!/usr/bin/env python3
"""
Example: How to use config.yaml

This demonstrates how to integrate the unified configuration
into your benchmark scripts.

Requirements:
    pip install pyyaml

Note: Ensure pyyaml is installed in your environment first:
    pip install pyyaml  # or
    pip install -r requirements.txt
"""

# Example 1: Using config_loader module
def example_with_config_loader():
    """Example using the config_loader module."""
    from config_loader import load_config
    
    # Load configuration
    config = load_config()
    
    print("=" * 70)
    print("EXAMPLE 1: Using config_loader module")
    print("=" * 70)
    
    # Access experiment settings
    print(f"\nExperiment: {config.experiment.name}")
    print(f"Version: {config.experiment.version}")
    print(f"Methodology: {config.experiment.methodology}")
    
    # Get model list
    print(f"\nAvailable models: {config.get_models_list()}")
    
    # Get parallelism for specific model and platform
    llama_nvidia = config.get_parallelism("llama", "nvidia")
    print(f"\nLlama on NVIDIA parallelism:")
    print(f"  TP: {llama_nvidia['tensor_model_parallel_size']}")
    print(f"  PP: {llama_nvidia['pipeline_model_parallel_size']}")
    print(f"  DP: {llama_nvidia['data_parallel_size']}")
    
    # Get training configuration
    print(f"\nTraining settings:")
    print(f"  Global Batch Size: {config.training.data.global_batch_size}")
    print(f"  Micro Batch Size: {config.training.data.micro_batch_size}")
    print(f"  Sequence Length: {config.training.data.seq_length}")
    print(f"  Max Steps: {config.training.duration.max_steps}")
    
    # Get platform optimizations
    nvidia_opts = config.get_platform_optimizations("nvidia")
    print(f"\nNVIDIA optimizations:")
    print(f"  Precision: {nvidia_opts.get('precision')}")
    print(f"  FP8 Hybrid: {nvidia_opts.get('fp8_hybrid')}")
    
    # Get output paths
    print(f"\nOutput paths:")
    print(f"  Directory: {config.get_output_dir()}")
    print(f"  Llama log: {config.get_log_filename('llama')}")
    print(f"  Benchmark: {config.get_benchmark_filename('cuda', 'llama')}")


# Example 2: Direct YAML parsing (if you don't want to use config_loader)
def example_direct_yaml():
    """Example using direct YAML parsing."""
    import yaml
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Direct YAML parsing")
    print("=" * 70)
    
    # Load YAML file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Access nested values
    print(f"\nExperiment name: {config['experiment']['name']}")
    
    # Get parallelism config
    methodology = config['experiment']['methodology']
    llama_amd = config['parallelism'][methodology]['llama']['amd']
    
    print(f"\nLlama on AMD (using {methodology}):")
    print(f"  TP: {llama_amd['tensor_model_parallel_size']}")
    print(f"  PP: {llama_amd['pipeline_model_parallel_size']}")
    print(f"  DP: {llama_amd['data_parallel_size']}")


# Example 3: Integration into pretrain script
def example_integration_pretrain():
    """Example showing how to integrate into pretrain_llama.py."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Integration into training script")
    print("=" * 70)
    
    print("""
# Add to pretrain_llama.py or pretrain_qwen.py:

from config_loader import load_config

def run_pretrain():
    # Load configuration
    config = load_config()
    
    # Get model-specific settings
    model_name = "llama"  # or "qwen"
    platform = "nvidia"   # auto-detect in real code
    
    # Get parallelism configuration
    parallelism = config.get_parallelism(model_name, platform)
    
    # Get training configuration  
    training_config = config.get_training_config()
    
    # Get platform optimizations
    optimizations = config.get_platform_optimizations(platform)
    
    # Initialize recipe with config values
    recipe = llm.llama31_8b.pretrain_recipe(
        name=f"{model_name}_pretrain",
        dir="/checkpoints",
        num_nodes=1,
        num_gpus_per_node=config.hardware.platforms[platform]['num_gpus'],
    )
    
    # Apply parallelism from config
    recipe.trainer.strategy.tensor_model_parallel_size = parallelism['tensor_model_parallel_size']
    recipe.trainer.strategy.pipeline_model_parallel_size = parallelism['pipeline_model_parallel_size']
    
    # Apply data config
    recipe.data.micro_batch_size = training_config['data']['micro_batch_size']
    recipe.data.global_batch_size = training_config['data']['global_batch_size']
    recipe.data.seq_length = training_config['data']['seq_length']
    
    # Apply training duration
    recipe.trainer.max_steps = training_config['duration']['max_steps']
    
    # Apply platform-specific optimizations
    if optimizations.get('fp8_hybrid'):
        recipe.model.config.fp8 = "hybrid"
        recipe.model.config.fp8_param = optimizations.get('fp8_param', False)
    
    # ... rest of your training code
    """)


# Example 4: Using config for comparison
def example_comparison():
    """Example showing how to use config for benchmark comparison."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Using config for comparison")
    print("=" * 70)
    
    print("""
# Add to compare_results.py:

from config_loader import load_config

def compare_platforms():
    config = load_config()
    
    # Get cloud costs for cost analysis
    nvidia_cost = config.get_cloud_cost('nvidia')
    amd_cost = config.get_cloud_cost('amd')
    
    # Get hardware specs for MFU calculation
    nvidia_specs = config.get_hardware_specs('nvidia')
    amd_specs = config.get_hardware_specs('amd')
    
    # Use specs for calculations
    nvidia_peak_tflops = nvidia_specs['peak_tflops_fp8']
    amd_peak_tflops = amd_specs['peak_tflops_bf16']
    
    # Calculate MFU
    mfu = (achieved_flops / peak_tflops) * 100
    
    # Get comparison settings
    comparison_config = config.comparison
    metrics_to_plot = comparison_config.metrics_to_compare
    
    # Generate plots based on config
    output_file = comparison_config.plots.filename
    # ... plotting code
    """)


# Example 5: Switching methodologies
def example_switching_methodology():
    """Example showing how to switch between comparison methodologies."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Switching methodologies")
    print("=" * 70)
    
    print("""
You can switch between comparison methodologies by:

1. Editing config.yaml:
   experiment:
     methodology: "identical_config"  # or "maximum_performance"

2. Or programmatically:
   config = load_config()
   
   # Get maximum performance config
   llama_nvidia_max = config.get_parallelism("llama", "nvidia", "maximum_performance")
   
   # Get identical config
   llama_nvidia_fair = config.get_parallelism("llama", "nvidia", "identical_config")
   
   print(f"Max perf TP: {llama_nvidia_max['tensor_model_parallel_size']}")
   print(f"Fair TP: {llama_nvidia_fair['tensor_model_parallel_size']}")
    """)


# Example 6: Configuration validation
def example_validation():
    """Example showing configuration validation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Configuration validation")
    print("=" * 70)
    
    print("""
from config_loader import load_config

def validate_config():
    config = load_config()
    
    # Validate parallelism
    parallelism = config.get_parallelism("llama", "nvidia")
    is_valid = config.validate_parallelism(
        tp=parallelism['tensor_model_parallel_size'],
        pp=parallelism['pipeline_model_parallel_size'],
        dp=parallelism['data_parallel_size'],
        num_gpus=8
    )
    
    if not is_valid:
        print("ERROR: Parallelism configuration is invalid!")
        print("Constraint: TP × PP × DP must equal num_gpus")
    
    # Calculate gradient accumulation
    grad_accum = config.calculate_gradient_accumulation_steps(
        global_batch_size=128,
        micro_batch_size=1,
        data_parallel_size=parallelism['data_parallel_size']
    )
    
    expected = parallelism['gradient_accumulation_steps']
    assert grad_accum == expected, f"Mismatch: {grad_accum} != {expected}"
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("TensorPrimat Configuration Examples")
    print("=" * 70)
    print("\nThese examples show how to use config.yaml")
    print("in your benchmark and training scripts.")
    print("\n⚠️  Note: Install pyyaml first: pip install pyyaml")
    
    try:
        # Only run if yaml is available
        import yaml
        example_with_config_loader()
        example_direct_yaml()
    except ImportError:
        print("\n⚠️  pyyaml not installed. Showing code examples only:")
    
    # These just print example code
    example_integration_pretrain()
    example_comparison()
    example_switching_methodology()
    example_validation()
    
    print("\n" + "=" * 70)
    print("✅ Examples complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Install pyyaml: pip install pyyaml")
    print("2. Try: python3 config_loader.py")
    print("3. Integrate into your training scripts")
    print("4. Run benchmarks with unified config")
    print("=" * 70)


if __name__ == "__main__":
    main()
