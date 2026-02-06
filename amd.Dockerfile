FROM rocm/primus:v25.11

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    ranger \
    zip \
    fish \
    tmux \
    iproute2 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# PyTorch memory settings
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# AMD ROCm specific settings
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HSA_ENABLE_SDMA=1
ENV HSA_FORCE_FINE_GRAIN_PCIE=1

# HuggingFace settings
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# Communication settings
ENV RCCL_DEBUG=ERROR
ENV NCCL_DEBUG=ERROR
ENV GLOO_LOG_LEVEL=ERROR
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME=ens51np0
ENV RCCL_MSCCL_ENABLE=0

# Disable NVIDIA-specific features for NeMo on AMD
ENV USE_APEX=NO
ENV TRANSFORMERS_NO_APEX=1
ENV NVTE_FRAMEWORK=pytorch

RUN mkdir -p /workspace/code
WORKDIR /workspace/code

# Install core requirements first (these should always work)
COPY amd-requirements.txt /workspace/code/
RUN pip install --no-cache-dir \
    lightning>=2.0.0 \
    transformers>=4.36.0 \
    accelerate>=0.30.0 \
    datasets>=2.14.0 \
    matplotlib>=3.5.0 \
    numpy>=1.21.0 \
    pyyaml>=6.0 \
    scipy \
    sentencepiece \
    protobuf \
    pydantic \
    packaging \
    tensorboard>=2.10.0 \
    wandb

# Install DeepSpeed (ROCm compatible version)
RUN pip install --no-cache-dir deepspeed>=0.12.0 || \
    echo "DeepSpeed installation failed - will use native PyTorch"

# Install megatron-core first (required by NeMo)
RUN pip install --no-cache-dir megatron-core>=0.5.0 || \
    echo "megatron-core installation failed"

# Install NeMo toolkit for NLP tasks
# Note: Some NVIDIA-specific features (FP8, TransformerEngine) won't work on AMD
RUN pip install --no-cache-dir 'nemo-toolkit[nlp]>=2.0.0' || \
    pip install --no-cache-dir nemo-toolkit || \
    echo "NeMo toolkit installation failed - nemo training scripts won't work"

# Install nemo_run for recipe-based training
RUN pip install --no-cache-dir nemo_run || \
    echo "nemo_run installation failed - nemo training scripts won't work"

COPY . /workspace/code/

SHELL ["/bin/fish", "-c"]
CMD ["/bin/fish"]
