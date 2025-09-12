# Jetson JP6.x base with CUDA/cuDNN/TensorRT
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# --- OS deps (+ OpenBLAS runtime; + libgfortran) ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget curl aria2 ca-certificates \
    python3-pip python3-dev \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev \
    python3-opencv \
    libopenblas0 libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

# (Optional) normalize CUDA path (JP6.2 ships CUDA 12.6)
RUN if [ -d /usr/local/cuda-12.6 ]; then ln -sfn /usr/local/cuda-12.6 /usr/local/cuda; fi

# --- Install cuSPARSELt (required by Jetson PyTorch >=24.06) ---
# Use NVIDIA's Tegra local repo for Ubuntu 22.04 (ARM64)
WORKDIR /tmp
RUN set -eux; \
    wget -nv https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb && \
    dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb && \
    cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libcusparselt0 libcusparselt-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Python base (pin NumPy for Jetson/ComfyUI compatibility) ---
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir numpy==1.26.4 wheel setuptools

# --- Training / data tooling (install BEFORE torch; no xformers/bnb) ---
RUN pip3 install --no-cache-dir \
    safetensors==0.4.4 \
    pillow==10.4.0 \
    tqdm==4.66.5 \
    imagehash==4.3.1 \
    requests==2.32.3 \
    beautifulsoup4==4.12.3 \
    accelerate==0.33.0 \
    transformers==4.44.2 \
    huggingface-hub==0.24.6 \
    sentencepiece==0.2.0 \
    timm==1.0.9 \
    einops==0.8.0 \
    lmdb==1.5.1 \
    diffusers==0.30.2 \
    tensorboard==2.17.1 \
    toml==0.10.2 \
    rich==13.7.1 \
    pyyaml==6.0.2

# --- CUDA-enabled PyTorch 2.5.0 for Jetson (install LAST; force, no-deps) ---
ARG TORCH_WHL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
RUN pip3 uninstall -y torch || true && \
    pip3 cache purge && \
    pip3 install --no-cache-dir --force-reinstall --no-deps ${TORCH_WHL_URL}

# Install additional Python packages needed
RUN pip3 install --no-cache-dir imagesize voluptuous

# Install build tools for torchvision and other dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    libjpeg-dev zlib1g-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev && \
    rm -rf /var/lib/apt/lists/*

# Build and install torchvision 0.20.1 from source to match Torch 2.5.0
RUN pip3 uninstall -y torchvision && \
    pip3 install --no-cache-dir --no-build-isolation --no-deps \
        git+https://github.com/pytorch/vision@v0.20.1

# --- Kohya sd-scripts (LoRA trainer) ---
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git

# Caches
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers

WORKDIR /workspace

