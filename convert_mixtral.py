from nemo.collections import llm
import torch

llm.import_ckpt(
    model=llm.MixtralModel(llm.Mixtral8x7BConfig()), 
    source='hf://mistralai/Mixtral-8x7B-v0.1',
    output_path='/checkpoints/mixtral_8x7b.nemo',
    overwrite=True
)
