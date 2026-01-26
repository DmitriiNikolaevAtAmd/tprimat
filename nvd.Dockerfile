FROM nvcr.io/nvidia/nemo:25.04

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    ranger \
    zip \
    tmux \
    fish \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=PHB

RUN mkdir -p /workspace/tprimat
WORKDIR /workspace/tprimat
COPY nvd-requirements.txt /workspace/tprimat/
RUN pip install --no-cache-dir -r nvd-requirements.txt
RUN pip install --no-cache-dir --force-reinstall \
    "packaging<26.0" \
    "cryptography>=43.0.0,<46"
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python3 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" && \
    python3 -c "import nemo; print(f'NeMo: {nemo.__version__}')"

SHELL ["/usr/bin/fish", "-c"]
CMD ["/usr/bin/fish"]
