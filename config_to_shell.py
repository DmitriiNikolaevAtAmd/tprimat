#!/usr/bin/env python3
"""
Helper script to export config.yaml values as shell variables.

Usage:
    source <(python3 config_to_shell.py)
    # or
    eval "$(python3 config_to_shell.py)"
"""

import sys
from config_loader import load_config

def main():
    try:
        config = load_config()
        
        # Basic configuration
        print(f"export CONFIG_OUTPUT_DIR='{config.get_output_dir()}'")
        print(f"export CONFIG_MAX_STEPS='{config.training.duration.max_steps}'")
        print(f"export CONFIG_TRAIN_ITERS='{config.training.duration.train_iters}'")
        print(f"export CONFIG_GLOBAL_BATCH_SIZE='{config.training.data.global_batch_size}'")
        print(f"export CONFIG_MICRO_BATCH_SIZE='{config.training.data.micro_batch_size}'")
        print(f"export CONFIG_SEQ_LENGTH='{config.training.data.seq_length}'")
        
        # Optimizer configuration
        print(f"export CONFIG_LEARNING_RATE='{config.training.optimizer.learning_rate}'")
        print(f"export CONFIG_MIN_LEARNING_RATE='{config.training.optimizer.learning_rate * 0.1}'")
        print(f"export CONFIG_WARMUP_STEPS='{config.training.optimizer.warmup_steps}'")
        print(f"export CONFIG_WEIGHT_DECAY='{config.training.optimizer.weight_decay}'")
        
        # Paths
        print(f"export CONFIG_PRIMUS_PATH='{config.get_primus_path()}'")
        
        # Models
        for model in config.get_models_list():
            model_config = config.get_model_config(model)
            primus_config_path = model_config.get('primus_config', '')
            print(f"export CONFIG_{model.upper()}_PRIMUS_CONFIG='{primus_config_path}'")
            print(f"export CONFIG_{model.upper()}_LOG_FILE='{config.get_log_filename(model)}'")
        
        # Hardware
        for platform in config.get_platforms_list():
            hw = config.get_hardware_config(platform)
            print(f"export CONFIG_{platform.upper()}_NUM_GPUS='{hw['num_gpus']}'")
        
        # Methodology
        print(f"export CONFIG_METHODOLOGY='{config.get_methodology()}'")
        
    except Exception as e:
        print(f"# Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
