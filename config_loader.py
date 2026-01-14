#!/usr/bin/env python3
"""
Configuration Loader for TensorPrimat
Loads and provides access to experiment_config.yaml settings
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class Config:
    """
    Loads and provides access to unified experiment configuration.
    
    Usage:
        config = Config()
        
        # Get parallelism settings for current platform and model
        llama_parallelism = config.get_parallelism("llama", "nvidia")
        
        # Get training parameters
        batch_size = config.training.data.global_batch_size
        
        # Get platform-specific optimizations
        nvidia_opts = config.get_platform_optimizations("nvidia")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to experiment_config.yaml. 
                        If None, looks in current directory.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config.yaml"
            )
        
        self.config_path = config_path
        self._config = self._load_config()
        
        # Create convenience accessors
        self.experiment = DotDict(self._config.get("experiment", {}))
        self.hardware = DotDict(self._config.get("hardware", {}))
        self.models = DotDict(self._config.get("models", {}))
        self.training = DotDict(self._config.get("training", {}))
        self.parallelism = DotDict(self._config.get("parallelism", {}))
        self.platform_optimizations = DotDict(self._config.get("platform_optimizations", {}))
        self.benchmarking = DotDict(self._config.get("benchmarking", {}))
        self.comparison = DotDict(self._config.get("comparison", {}))
        self.paths = DotDict(self._config.get("paths", {}))
        self.logging_config = DotDict(self._config.get("logging", {}))
        self.validation = DotDict(self._config.get("validation", {}))
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables in paths
        config = self._expand_env_vars(config)
        
        return config
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in strings."""
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(v) for v in obj]
        elif isinstance(obj, str):
            # Expand ${VAR:-default} syntax
            import re
            pattern = r'\$\{([^:}]+)(?::-)([^}]+)?\}'
            
            def replace_env_var(match):
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.environ.get(var_name, default)
            
            return re.sub(pattern, replace_env_var, obj)
        else:
            return obj
    
    def get_methodology(self) -> str:
        """Get current comparison methodology."""
        return self.experiment.methodology
    
    def get_parallelism(
        self,
        model: str,
        platform: str,
        methodology: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Get parallelism configuration for model and platform.
        
        Args:
            model: Model name (llama, qwen)
            platform: Platform name (nvidia, amd)
            methodology: Override methodology (maximum_performance or identical_config)
        
        Returns:
            Dictionary with TP, PP, DP, and gradient accumulation steps
        """
        if methodology is None:
            methodology = self.get_methodology()
        
        try:
            config = self.parallelism[methodology][model][platform]
            return dict(config)
        except KeyError as e:
            raise ValueError(
                f"No parallelism config for model={model}, platform={platform}, "
                f"methodology={methodology}: {e}"
            )
    
    def get_platform_optimizations(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific optimizations."""
        try:
            return dict(self.platform_optimizations[platform])
        except KeyError:
            return {}
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get model configuration."""
        try:
            return dict(self.models[model])
        except KeyError:
            raise ValueError(f"Unknown model: {model}")
    
    def get_hardware_config(self, platform: str) -> Dict[str, Any]:
        """Get hardware configuration for platform."""
        try:
            return dict(self.hardware.platforms[platform])
        except KeyError:
            raise ValueError(f"Unknown platform: {platform}")
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return dict(self.training)
    
    def get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmarking configuration."""
        return dict(self.benchmarking)
    
    def get_output_dir(self) -> str:
        """Get output directory path (can be overridden by OUTPUT_DIR env var)."""
        return os.environ.get('OUTPUT_DIR', self.benchmarking.output.directory)
    
    def get_log_filename(self, model: str) -> str:
        """Get log filename for model."""
        return self.benchmarking.output.log_format.format(model=model)
    
    def get_benchmark_filename(self, software_stack: str, model: str) -> str:
        """Get benchmark output filename."""
        return self.benchmarking.output.filename_format.format(
            software_stack=software_stack,
            model=model
        )
    
    def get_cloud_cost(self, platform: str) -> float:
        """Get cloud cost per hour for platform (8 GPUs)."""
        cost_key = f"{platform}_{'h100' if platform == 'nvidia' else 'mi300x'}_8gpu_per_hour"
        return self.benchmarking.enhanced_metrics.cloud_costs.get(cost_key, 0.0)
    
    def get_hardware_specs(self, platform: str) -> Dict[str, Any]:
        """Get hardware specifications for MFU calculation."""
        if platform == "nvidia" or platform == "nvd":
            return dict(self.benchmarking.enhanced_metrics.hardware_specs.nvidia_h100)
        elif platform == "amd":
            return dict(self.benchmarking.enhanced_metrics.hardware_specs.amd_mi300x)
        else:
            raise ValueError(f"Unknown platform: {platform}")
    
    def get_primus_path(self) -> str:
        """Get Primus installation path."""
        return self.paths.primus.installation
    
    def get_primus_config_path(self, model: str) -> str:
        """Get full path to Primus config file for model."""
        primus_path = self.get_primus_path()
        model_config = self.get_model_config(model)
        config_file = model_config.get("primus_config", "")
        return os.path.join(primus_path, config_file)
    
    def validate_parallelism(
        self,
        tp: int,
        pp: int,
        dp: int,
        num_gpus: int
    ) -> bool:
        """
        Validate that parallelism configuration is consistent.
        
        Args:
            tp: Tensor parallel size
            pp: Pipeline parallel size
            dp: Data parallel size
            num_gpus: Total number of GPUs
        
        Returns:
            True if valid, False otherwise
        """
        return tp * pp * dp == num_gpus
    
    def calculate_gradient_accumulation_steps(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int
    ) -> int:
        """
        Calculate gradient accumulation steps.
        
        Args:
            global_batch_size: Global batch size
            micro_batch_size: Micro batch size
            data_parallel_size: Data parallel size
        
        Returns:
            Number of gradient accumulation steps
        """
        return global_batch_size // (micro_batch_size * data_parallel_size)
    
    def get_models_list(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_platforms_list(self) -> List[str]:
        """Get list of available platforms."""
        return list(self.hardware.platforms.keys())
    
    def print_config_summary(self, model: str, platform: str):
        """Print configuration summary for model and platform."""
        print("=" * 60)
        print(f"Configuration Summary: {model.upper()} on {platform.upper()}")
        print("=" * 60)
        
        # Model info
        model_config = self.get_model_config(model)
        print(f"\nModel: {model_config.get('full_name', model)}")
        print(f"Parameters: {model_config.get('num_parameters', 'N/A')}")
        
        # Hardware info
        hw_config = self.get_hardware_config(platform)
        print(f"\nHardware: {hw_config.get('gpu_model', 'N/A')}")
        print(f"GPUs: {hw_config.get('num_gpus', 'N/A')}")
        print(f"Memory/GPU: {hw_config.get('memory_per_gpu_gb', 'N/A')} GB")
        
        # Training config
        print(f"\nTraining Configuration:")
        print(f"  Global Batch Size: {self.training.data.global_batch_size}")
        print(f"  Micro Batch Size: {self.training.data.micro_batch_size}")
        print(f"  Sequence Length: {self.training.data.seq_length}")
        print(f"  Max Steps: {self.training.duration.max_steps}")
        
        # Parallelism
        parallelism = self.get_parallelism(model, platform)
        print(f"\nParallelism Strategy ({self.get_methodology()}):")
        print(f"  Tensor Parallel (TP): {parallelism['tensor_model_parallel_size']}")
        print(f"  Pipeline Parallel (PP): {parallelism['pipeline_model_parallel_size']}")
        print(f"  Data Parallel (DP): {parallelism['data_parallel_size']}")
        print(f"  Gradient Accumulation: {parallelism['gradient_accumulation_steps']}")
        
        # Platform optimizations
        opts = self.get_platform_optimizations(platform)
        print(f"\nPlatform Optimizations:")
        print(f"  Precision: {opts.get('precision', 'N/A')}")
        print(f"  FP8 Hybrid: {opts.get('fp8_hybrid', False)}")
        print(f"  Activation Checkpointing: {opts.get('activation_checkpointing', False)}")
        
        print("=" * 60)


class DotDict(dict):
    """
    Dictionary with dot notation access.
    
    Usage:
        d = DotDict({'a': {'b': 1}})
        print(d.a.b)  # 1
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
    
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
    
    def get(self, key, default=None):
        """Get with default value."""
        try:
            return self[key]
        except KeyError:
            return default


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load experiment configuration.
    
    Args:
        config_path: Path to config file (optional)
    
    Returns:
        Config instance
    """
    return Config(config_path)


if __name__ == "__main__":
    """Example usage and testing."""
    import sys
    
    try:
        config = load_config()
        
        print("TensorPrimat Configuration Loader")
        print("=" * 60)
        print(f"Config file: {config.config_path}")
        print(f"Experiment: {config.experiment.name}")
        print(f"Methodology: {config.get_methodology()}")
        print()
        
        # Print summary for each model and platform
        for model in config.get_models_list():
            for platform in config.get_platforms_list():
                config.print_config_summary(model, platform)
                print()
        
        # Show available methodologies
        print("=" * 60)
        print("Available Methodologies:")
        print("=" * 60)
        print("1. maximum_performance - Each platform uses optimal settings")
        print("2. identical_config - Both platforms use identical settings")
        print()
        
        # Example: Get specific configuration
        print("=" * 60)
        print("Example: Getting LLAMA configuration for NVIDIA")
        print("=" * 60)
        parallelism = config.get_parallelism("llama", "nvidia")
        print(f"Parallelism config: {parallelism}")
        print()
        
        print("✅ Configuration loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
