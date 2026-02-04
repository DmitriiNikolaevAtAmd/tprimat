FROM nvcr.io/nvidia/nemo:25.04

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    ranger \
    zip \
    tmux \
    fish \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y apex || true

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=PHB
ENV USE_APEX=NO
ENV TRANSFORMERS_NO_APEX=1

RUN mkdir -p /workspace/code
WORKDIR /workspace/code
COPY nvd-requirements.txt /workspace/code/
RUN pip install --no-cache-dir -r nvd-requirements.txt

SHELL ["/usr/bin/fish", "-c"]
CMD ["/usr/bin/fish"]
