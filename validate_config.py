#!/usr/bin/env python3
"""
Validate TPrimat Configuration

Checks if parallelism configurations are valid for the models.
Particularly important for tensor parallelism constraints.
"""

from config_loader import load_config
import sys

# Model specifications
MODEL_SPECS = {
    'llama': {
        'name': 'Llama 3.1 8B',
        'num_attention_heads': 32,
        'num_gpus_min': 8,  # Minimum GPUs needed
    },
    'qwen': {
        'name': 'Qwen 2.5 7B',
        'num_attention_heads': 28,
        'num_gpus_min': 8,
    },
}

def validate_tensor_parallelism(model: str, tp: int) -> tuple[bool, str]:
    """
    Validate that TP is compatible with model's attention heads.
    
    Returns:
        (is_valid, message)
    """
    specs = MODEL_SPECS[model]
    num_heads = specs['num_attention_heads']
    
    if num_heads % tp != 0:
        return False, f"[X] {specs['name']}: {num_heads} heads cannot be divided by TP={tp} (result: {num_heads/tp:.2f})"
    else:
        heads_per_gpu = num_heads // tp
        return True, f"[OK] {specs['name']}: TP={tp} → {heads_per_gpu} heads per GPU"

def validate_parallelism_product(tp: int, pp: int, dp: int, num_gpus: int) -> tuple[bool, str]:
    """
    Validate that TP * PP * DP = num_gpus.
    
    Returns:
        (is_valid, message)
    """
    product = tp * pp * dp
    if product != num_gpus:
        return False, f"[X] TP({tp}) × PP({pp}) × DP({dp}) = {product} ≠ {num_gpus} GPUs"
    else:
        return True, f"[OK] TP({tp}) × PP({pp}) × DP({dp}) = {product} GPUs"

def validate_configuration(config, methodology: str, model: str, platform: str) -> list[tuple[bool, str]]:
    """
    Validate a specific configuration.
    
    Returns:
        List of (is_valid, message) tuples
    """
    results = []
    
    # Get parallelism settings
    try:
        parallelism = config.get_parallelism(model, platform, methodology=methodology)
    except ValueError as e:
        results.append((False, f"[X] Configuration not found: {e}"))
        return results
    
    tp = parallelism['tensor_model_parallel_size']
    pp = parallelism['pipeline_model_parallel_size']
    dp = parallelism['data_parallel_size']
    
    # Get hardware config
    hw_config = config.get_hardware_config(platform)
    num_gpus = hw_config['num_gpus']
    
    # Validate TP constraints
    is_valid, msg = validate_tensor_parallelism(model, tp)
    results.append((is_valid, msg))
    
    # Validate parallelism product
    is_valid, msg = validate_parallelism_product(tp, pp, dp, num_gpus)
    results.append((is_valid, msg))
    
    return results

def main():
    print("=" * 80)
    print("TPrimat Configuration Validator")
    print("=" * 80)
    print()
    
    # Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"[X] Failed to load config: {e}")
        sys.exit(1)
    
    # Get all methodologies
    methodologies = list(config.parallelism.keys())
    models = list(MODEL_SPECS.keys())
    platforms = ['nvidia', 'amd']
    
    all_valid = True
    
    for methodology in methodologies:
        print(f"\n{'─' * 80}")
        print(f"Configuration: {methodology}")
        print(f"{'─' * 80}")
        
        for model in models:
            for platform in platforms:
                header = f"{model.upper()} on {platform.upper()}"
                print(f"\n  {header}:")
                print(f"  {'-' * len(header)}")
                
                results = validate_configuration(config, methodology, model, platform)
                
                for is_valid, msg in results:
                    print(f"  {msg}")
                    if not is_valid:
                        all_valid = False
    
    print(f"\n{'=' * 80}")
    
    if all_valid:
        print("[OK] All configurations are valid!")
        sys.exit(0)
    else:
        print("[X] Some configurations have issues. Please fix config.yaml.")
        print()
        print("[i] Tip: For tensor parallelism, use TP values that divide both:")
        print(f"   - Llama: {MODEL_SPECS['llama']['num_attention_heads']} heads → Valid TP: 1, 2, 4, 8, 16, 32")
        print(f"   - Qwen:  {MODEL_SPECS['qwen']['num_attention_heads']} heads → Valid TP: 1, 2, 4, 7, 14, 28")
        print(f"   - Common valid TP values: 1, 2, 4")
        sys.exit(1)

if __name__ == "__main__":
    main()
