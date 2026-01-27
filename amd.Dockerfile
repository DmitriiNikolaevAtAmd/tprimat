FROM rocm/primus:v25.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    neovim \
    ranger \
    zip \
    fish \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Python settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ROCm/HSA settings
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HSA_ENABLE_SDMA=1
ENV HSA_FORCE_FINE_GRAIN_PCIE=1
ENV TRACELENS_ENABLED=0
ENV PROFILE_ITERS=""

# RCCL settings (disable verbose logging for performance)
ENV RCCL_DEBUG=WARN
ENV NCCL_DEBUG=WARN
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_IB_DISABLE=0
ENV RCCL_MSCCL_ENABLE=0

# HuggingFace token - pass at runtime: docker run -e HF_TOKEN=xxx
# Required for gated models like Llama 3.1
ENV HF_TOKEN=""
ENV HUGGINGFACE_HUB_TOKEN=""
ENV HF_HOME="/workspace/cache/huggingface"

# Create directories
RUN mkdir -p /workspace/tprimat /workspace/cache/huggingface
WORKDIR /workspace/tprimat

# Install Python dependencies
COPY amd-requirements.txt /workspace/tprimat/
RUN pip install --no-cache-dir -r amd-requirements.txt

# Copy project files
COPY . /workspace/tprimat/

# Make scripts executable
RUN chmod +x /workspace/tprimat/*.sh 2>/dev/null || true

SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
