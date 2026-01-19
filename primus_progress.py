#!/usr/bin/env python3
"""
Progress wrapper for Primus training.
Monitors log file and displays alive-progress bar for training iterations.

Usage:
    python3 primus_progress.py --log-file /path/to/log --total-iters 50 &
    # Then start your training which writes to log-file
    
Or use with subprocess:
    python3 primus_progress.py --command "bash train.sh" --total-iters 50
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Optional

from alive_progress import alive_bar


def parse_iteration_from_line(line: str) -> Optional[int]:
    """
    Parse current iteration number from a Primus/Megatron log line.
    
    Handles formats like:
    - "iteration 10/50"
    - "iter: 10"
    - "[step 10]"
    - "training step 10"
    """
    patterns = [
        r'iteration\s+(\d+)',
        r'iter[:\s]+(\d+)',
        r'\[step\s+(\d+)\]',
        r'training\s+step\s+(\d+)',
        r'step\s+(\d+)[/\s]',
        r'global_step[:\s]+(\d+)',
    ]
    
    line_lower = line.lower()
    for pattern in patterns:
        match = re.search(pattern, line_lower)
        if match:
            return int(match.group(1))
    return None


def monitor_log_file(log_file: str, total_iters: int, poll_interval: float = 0.5):
    """
    Monitor a log file and show progress bar based on iterations.
    
    Args:
        log_file: Path to the log file to monitor
        total_iters: Total number of training iterations expected
        poll_interval: How often to check the log file (seconds)
    """
    log_path = Path(log_file)
    last_position = 0
    current_iter = 0
    last_iter = 0
    
    # Wait for log file to exist
    print(f"⏳ Waiting for log file: {log_file}")
    while not log_path.exists():
        time.sleep(poll_interval)
    
    print(f"📄 Monitoring: {log_file}")
    print()
    
    with alive_bar(total_iters, title="Primus Training", bar="smooth", 
                   spinner="dots_waves", elapsed_end=True) as bar:
        while current_iter < total_iters:
            try:
                with open(log_path, 'r', errors='ignore') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                    
                    for line in new_lines:
                        iter_num = parse_iteration_from_line(line)
                        if iter_num is not None and iter_num > current_iter:
                            # Update progress
                            steps_done = iter_num - current_iter
                            current_iter = iter_num
                            bar.text(f"→ Step {current_iter}/{total_iters}")
                            
                            # Update bar for each step completed
                            for _ in range(steps_done):
                                if last_iter < total_iters:
                                    bar()
                                    last_iter += 1
                            
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"⚠️  Error reading log: {e}")
            
            time.sleep(poll_interval)
    
    print()
    print("✅ Training complete!")


def run_with_progress(command: str, log_file: str, total_iters: int):
    """
    Run a command while showing progress bar.
    
    Args:
        command: Shell command to execute
        log_file: Path where the command writes its log
        total_iters: Total number of iterations expected
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start the monitor thread
    monitor_thread = Thread(
        target=monitor_log_file,
        args=(log_file, total_iters),
        daemon=True
    )
    monitor_thread.start()
    
    # Run the command
    print(f"🚀 Running: {command}")
    print(f"📝 Log: {log_file}")
    print()
    
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT
        )
        process.wait()
    
    # Give monitor time to catch up
    time.sleep(1)
    
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Progress bar wrapper for Primus training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--log-file',
        required=True,
        help='Path to the log file to monitor (or write to if using --command)'
    )
    
    parser.add_argument(
        '--total-iters',
        type=int,
        required=True,
        help='Total number of training iterations'
    )
    
    parser.add_argument(
        '--command',
        default=None,
        help='Command to run (optional - if not provided, just monitors log file)'
    )
    
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=0.5,
        help='How often to check log file in seconds (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    if args.command:
        # Run command with progress
        exit_code = run_with_progress(args.command, args.log_file, args.total_iters)
        sys.exit(exit_code)
    else:
        # Just monitor existing/incoming log file
        monitor_log_file(args.log_file, args.total_iters, args.poll_interval)


if __name__ == "__main__":
    main()
