FROM rocm/primus:v25.11

RUN apt-get update && apt-get install -y \
    neovim \
    ranger \
    zip \
    && rm -rf /var/lib/apt/lists/*

ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HSA_ENABLE_SDMA=1
ENV TRACELENS_ENABLED=1
ENV PROFILE_ITERS="0,50"

WORKDIR /workspace
RUN cd Primus
RUN git pull
RUN cd /workspace/tprimat

CMD ["/bin/bash"]
