FROM rocm/primus:v25.11

RUN apt-get update && apt-get install -y \
    git \
    neovim \
    ranger \
    zip \
    fish \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HSA_ENABLE_SDMA=1
ENV HSA_FORCE_FINE_GRAIN_PCIE=1
ENV TRACELENS_ENABLED=0
ENV PROFILE_ITERS=""

ENV RCCL_DEBUG=WARN
ENV NCCL_DEBUG=WARN
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_IB_DISABLE=0
# Remove NCCL_SOCKET_IFNAME - let RCCL auto-detect the network interface
# If you get "no socket interface found", check container network with: ip addr
ENV RCCL_MSCCL_ENABLE=0

RUN mkdir -p /workspace/tprimat
WORKDIR /workspace/tprimat
COPY amd-requirements.txt /workspace/tprimat/
RUN pip install --no-cache-dir -r amd-requirements.txt

SHELL ["/usr/bin/fish", "-c"]
CMD ["/usr/bin/fish"]
