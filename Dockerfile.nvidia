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

SHELL ["/usr/bin/fish", "-c"]
CMD ["/usr/bin/fish"]
