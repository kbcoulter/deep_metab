FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CUDA=cu121

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Accept Conda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n graphormer-rt python=3.9.18 -y && \
    /opt/conda/envs/graphormer-rt/bin/pip install --upgrade "pip<24.1"

ENV CONDA_DEFAULT_ENV=graphormer-rt
ENV PATH=/opt/conda/envs/graphormer-rt/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /workspace
ARG CACHEBUST=1

# Clone Graphormer-RT repository
RUN rm -rf Graphormer-RT && \
    git clone https://github.com/HopkinsLaboratory/Graphormer-RT.git

WORKDIR /workspace/Graphormer-RT/fairseq

# Upgrade pip before package installs
RUN pip install --upgrade "pip<24.1"

# Install Graphormer-RT package
RUN pip install --editable ./


# Graph / ML / chemistry dependencies
RUN pip install --editable ./ && \
    pip install dgl==1.1.3+${CUDA} -f https://data.dgl.ai/wheels/${CUDA}/repo.html && \
    pip install torch-geometric==2.4.0 && \
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html && \
    pip install ogb==1.3.2 && \
    pip install rdkit-pypi==2022.9.5 && \
    pip install matplotlib && \
    pip install numpy==1.23 && \
    pip install dgllife==0.3.2 && \
    pip install mordred==1.2.0 && \
    pip install torchaudio==2.1.0 && \
    pip install rdkit==2023.9.1 && \
    pip install rdkit-pypi==2022.9.5

WORKDIR /workspace/Graphormer-RT

# Reinstall fairseq from pinned Git commit
RUN python -m pip uninstall -y fairseq || true && \
    python -m pip install --no-cache-dir --no-build-isolation \
    "fairseq @ git+https://github.com/facebookresearch/fairseq.git@98ebe4f1ada75d006717d84f9d603519d8ff5579"

CMD ["/bin/bash"]