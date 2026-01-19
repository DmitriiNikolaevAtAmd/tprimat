#!/usr/bin/env python3
"""
Primus Pipeline Runner with alive-progress status bar.
Runs all models (Llama, Qwen) sequentially with visual progress tracking.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from alive_progress import alive_bar


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color


def run_model(model: str, script_dir: str, output_dir: str) -> bool:
    """
    Run Primus training for a single model.
    
    Args:
        model: Model name (llama, qwen)
        script_dir: Directory containing the scripts
        output_dir: Output directory for results
        
    Returns:
        True if successful, False otherwise
    """
    script = os.path.join(script_dir, f"run_primus_{model}.sh")
    
    if not os.path.isfile(script):
        print(f"{Colors.RED}❌ Script not found: {script}{Colors.NC}")
        return False
    
    env = os.environ.copy()
    env['OUTPUT_DIR'] = output_dir
    
    result = subprocess.run(
        ["bash", script],
        env=env,
        cwd=script_dir
    )
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run Primus training pipeline with progress tracking"
    )
    parser.add_argument(
        '--parallel',
        choices=['maximum_performance', 'truly_identical', 'memory_optimized',
                 'minimal_communication', 'balanced'],
        default='truly_identical',
        help='Parallelism strategy (default: truly_identical)'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['llama', 'qwen'],
        help='Models to train (default: llama qwen)'
    )
    parser.add_argument(
        '--cooldown',
        type=int,
        default=30,
        help='Cooldown seconds between models (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Set parallelism strategy environment variable
    os.environ['PARALLEL'] = args.parallel
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models = args.models
    output_dir = args.output_dir
    
    # Make output_dir absolute if relative
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Print header
    print(f"{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.CYAN}║{Colors.NC}        {Colors.MAGENTA}Primus Training Pipeline{Colors.NC}                        {Colors.CYAN}║{Colors.NC}")
    print(f"{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.NC}")
    print()
    print(f"{Colors.BLUE}📋 Training Plan:{Colors.NC}")
    for i, model in enumerate(models, 1):
        model_names = {'llama': 'Llama 3.1 8B', 'qwen': 'Qwen 2.5 7B'}
        print(f"   {i}. {model_names.get(model, model)}")
    print()
    print(f"{Colors.BLUE}🔧 Parallelism:{Colors.NC} {args.parallel}")
    print(f"{Colors.BLUE}📁 Output:{Colors.NC} {output_dir}")
    print()
    
    successful = []
    failed = []
    
    with alive_bar(len(models), title="Primus Pipeline", bar="smooth", 
                   spinner="dots_waves", elapsed_end=True) as bar:
        for i, model in enumerate(models):
            bar.text(f"→ Training {model}")
            
            print(f"\n{Colors.CYAN}{'═'*60}{Colors.NC}")
            print(f"{Colors.BLUE}Starting:{Colors.NC} {Colors.GREEN}{model}{Colors.NC}")
            print(f"{Colors.CYAN}{'═'*60}{Colors.NC}")
            print()
            
            if run_model(model, script_dir, output_dir):
                successful.append(model)
                print(f"{Colors.GREEN}✅ {model} completed successfully{Colors.NC}")
            else:
                failed.append(model)
                print(f"{Colors.RED}❌ {model} failed (continuing with next model...){Colors.NC}")
            
            bar()
            
            # Cooldown between models (not after last one)
            if i < len(models) - 1:
                print()
                print(f"{Colors.YELLOW}⏳ Cooling down for {args.cooldown} seconds...{Colors.NC}")
                time.sleep(args.cooldown)
    
    # Print summary
    print()
    print(f"{Colors.CYAN}{'═'*60}{Colors.NC}")
    print(f"{Colors.BLUE}SUMMARY{Colors.NC}")
    print(f"{Colors.CYAN}{'═'*60}{Colors.NC}")
    print()
    
    if successful:
        print(f"{Colors.GREEN}✅ Successful ({len(successful)}): {', '.join(successful)}{Colors.NC}")
        print()
        print("Results saved to:")
        for model in successful:
            print(f"   📄 {output_dir}/benchmark_rocm_{model}.json")
        print()
    
    if failed:
        print(f"{Colors.RED}❌ Failed ({len(failed)}): {', '.join(failed)}{Colors.NC}")
        print()
    
    if successful:
        print(f"{Colors.CYAN}{'═'*60}{Colors.NC}")
        print(f"{Colors.BLUE}🎯 Next Steps:{Colors.NC}")
        print(f"{Colors.CYAN}{'═'*60}{Colors.NC}")
        print()
        print("1. Run on NVIDIA system (if not done):")
        print("   ./benchmark.py")
        print()
        print("2. Compare results:")
        print("   python3 compare.py")
        print()
    
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
