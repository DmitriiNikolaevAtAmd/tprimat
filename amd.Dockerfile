FROM rocm/primus:v25.11

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

ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HSA_ENABLE_SDMA=1
ENV HSA_FORCE_FINE_GRAIN_PCIE=1
ENV TRACELENS_ENABLED=0

ENV RCCL_DEBUG=INFO
ENV NCCL_DEBUG=INFO
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_IB_DISABLE=0

RUN mkdir -p /workspace/tprimat
WORKDIR /workspace/tprimat
COPY amd-requirements.txt /workspace/tprimat/
RUN pip install --no-cache-dir "nemo_toolkit[all]" nemo_run
RUN pip install --no-cache-dir -r amd-requirements.txt
RUN pip install --no-cache-dir --force-reinstall \
    "packaging<26.0" \
    "cryptography>=43.0.0,<46"
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python3 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" && \
    python3 -c "import nemo; print(f'NeMo: {nemo.__version__}')"

SHELL ["/usr/bin/fish", "-c"]
CMD ["/usr/bin/fish"]
