from nemo.collections import llm
import torch

llm.import_ckpt(
    model=llm.Qwen2Model(llm.Qwen2Config7B()), 
    source='hf://Qwen/Qwen2.5-7B',
    output_path='/checkpoints/qwen2_5_7b.nemo'
)
