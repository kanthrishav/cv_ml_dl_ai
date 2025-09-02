# NVIDIA L4T JetPack container (exists: r36.4.0)
# Works on JP 6.1/6.2 hosts; r36.4.0 is the latest NGC tag today.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential cmake \
    libgl1 libglib2.0-0 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Isolated venv (no host Python touched)
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Pin Jetson wheels repo (stable mirror) + constraints
# (The *.io mirror is the backup many folks use when *.dev is flaky.)
ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
ENV PIP_CONSTRAINT=/opt/constraints.txt
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY constraints.txt /opt/constraints.txt

# Tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install pinned numpy first (prevents surprise rebuilds)
RUN python -m pip install "numpy==1.26.4"

# Install CUDA-enabled torch/torchvision that match JP 6.2 (CUDA 12.6)
# These versions are specifically referenced for JetPack 6.2 / CU126 by PyTorch-TensorRT docs.
RUN python -m pip install "torch==2.8.0" "torchvision==0.23.0"

# Clone AUTOMATIC1111
WORKDIR /opt
RUN git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
WORKDIR /opt/stable-diffusion-webui

# Use the pinned requirements list shipped with A1111
RUN python -m pip install -r requirements_versions.txt

# Preinstall Model Manager extension (manage models from the UI)
RUN mkdir -p /opt/stable-diffusion-webui/extensions && \
    git clone --depth 1 https://github.com/GustavSO/sd-webui-model-manager \
      /opt/stable-diffusion-webui/extensions/sd-webui-model-manager

# Favor SDPA over xFormers on Jetson (fewer ARM wheel headaches)
# Point WebUI at your ComfyUI models via read-only bind at /comfy
ENV COMMANDLINE_ARGS="--listen --port 7860 --api --opt-sdp-attention \
  --enable-insecure-extension-access \
  --ckpt-dir /comfy/checkpoints --vae-dir /comfy/vae \
  --lora-dir /comfy/loras --embeddings-dir /comfy/embeddings \
  --controlnet-dir /comfy/controlnet --hypernetwork-dir /comfy/hypernetworks"

CMD bash -lc 'cd /opt/stable-diffusion-webui && \
  echo "Torch: $(python -c \"import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\")" && \
  python launch.py ${COMMANDLINE_ARGS}'
