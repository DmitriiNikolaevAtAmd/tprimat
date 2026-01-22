from nemo.collections import llm
import torch

llm.import_ckpt(
    model=llm.LlamaModel(llm.Llama3Config8B()), 
    source='hf://meta-llama/Llama-3.1-8B',
    output_path='/checkpoints/llama3_1_8b.nemo'
)
