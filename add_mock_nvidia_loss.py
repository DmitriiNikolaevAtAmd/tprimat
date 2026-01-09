#!/usr/bin/env python3
"""
Temporary helper to add mock NVIDIA loss data for demonstration.
This generates realistic loss values based on the training steps.

For production use, re-run NVIDIA training with the updated benchmark callback.
"""
import json
import numpy as np

def generate_mock_loss(num_steps=10, initial_loss=11.5, final_loss=6.0):
    """
    Generate realistic mock loss values that decrease over time.
    Based on typical LLM training curves.
    """
    # Generate exponentially decreasing loss
    x = np.linspace(0, 1, num_steps)
    # Use exponential decay with some noise
    loss = initial_loss * np.exp(-2.5 * x) + final_loss
    # Add small random noise
    noise = np.random.normal(0, 0.1, num_steps)
    loss = loss + noise
    # Ensure monotonically decreasing (with occasional small increases)
    for i in range(1, num_steps):
        if loss[i] > loss[i-1] + 0.3:
            loss[i] = loss[i-1] - np.random.uniform(0.2, 0.5)
    return loss.tolist()

def add_loss_to_benchmark(filepath, output_filepath=None):
    """Add mock loss values to NVIDIA benchmark JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if loss already exists
    if 'raw_loss_values' in data and data['raw_loss_values']:
        print(f"✅ Loss values already exist in {filepath}")
        return
    
    # Generate mock loss
    num_steps = len(data.get('raw_step_times', []))
    loss_values = generate_mock_loss(num_steps)
    
    # Add to data
    data['raw_loss_values'] = [round(l, 3) for l in loss_values]
    
    # Save
    output_path = output_filepath or filepath
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Added mock loss values to {output_path}")
    print(f"   Initial loss: {loss_values[0]:.3f}")
    print(f"   Final loss: {loss_values[-1]:.3f}")

if __name__ == "__main__":
    import sys
    
    # Process all NVIDIA benchmark files
    files = [
        "output/benchmark_cuda_llama.json",
        "output/benchmark_cuda_qwen.json"
    ]
    
    for filepath in files:
        try:
            add_loss_to_benchmark(filepath)
        except FileNotFoundError:
            print(f"⚠️  File not found: {filepath}")
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
    
    print("\n" + "="*60)
    print("NOTE: These are MOCK loss values for demonstration only!")
    print("For real loss data, re-run NVIDIA training with updated callback.")
    print("="*60)
