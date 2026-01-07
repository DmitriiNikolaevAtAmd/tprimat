from nemo.collections import llm
import torch

llm.import_ckpt(
    model=llm.MistralModel(llm.MistralConfig7B()), 
    source='hf://mistralai/Mistral-7B-v0.1',
    output_path='/checkpoints/mistral_7b.nemo',
    overwrite=True
)

