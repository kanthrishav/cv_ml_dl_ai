# COMFYUI

## Overview

ComfyUI is a powerful, open-source, node-based interface for Stable Diffusion models that allows users to create complex workflows visually by connecting different nodes representing specific functions, such as loading models, inputting text, or creating images. This free, local tool offers high flexibility and fine-grained control over the AI image and video generation process without coding, making it popular for creating detailed and customized content.

Example of converting a random street into an Indian street

<img width="1917" height="975" alt="image" src="https://github.com/user-attachments/assets/c5e08a6a-fa3c-4ae3-b54d-cf215fda54df" />



Example of converting a random empty living room into an Indian, a Parisian and a Japanese living room

<img width="1917" height="975" alt="image" src="https://github.com/user-attachments/assets/128bdb5a-fc98-4a00-b74a-045aee4ff844" />

<table><tr><td><img width="400" height="260" alt="indian" src="https://github.com/user-attachments/assets/e1eb60c8-729f-42e7-9f7e-707a4dbda6e8" /></td><td><img width="400" height="260" alt="japanese" src="https://github.com/user-attachments/assets/d1e0fbb1-4f45-4ed5-9f6f-ccbec0c368f8" /></td><td><img width="400" height="260" alt="parisian" src="https://github.com/user-attachments/assets/f0d04e03-5385-4aec-9741-e52cc2704087" /></td></tr></table>


https://github.com/user-attachments/assets/d3562c5a-5be4-4555-8d1e-7814312debb1



### Key Features and Benefits
1. Node-Based System: Users build "workflows" by connecting nodes, which are like building blocks that perform specific tasks in the AI generation process. 
2. Flexibility and Control: ComfyUI provides granular control over the image generation process, allowing users to fine-tune parameters and create detailed workflows beyond what's possible with simpler interfaces. 
3. Local and Free: It operates locally on a user's computer, eliminating subscription fees and online connection requirements. 
4. Open-Source & Community-Driven: As an open-source project, it benefits from a dedicated community that creates and shares custom nodes and complex workflows, which can be easily loaded and used by others. 
5. Visual Workflow Management: The interface is highly visual, representing the entire process in a flowchart-like manner, making it easier to understand and manage. 
6. Versatility: It supports various AI models and can be used for tasks like text-to-image, image-to-image, upscaling, inpainting, and more. 

### How it Works
1. Nodes: Each task in the workflow, like loading a specific AI model or defining a text prompt, is represented by a node. 
2. Connecting Nodes: Users connect these nodes with lines that show how information flows from one step to the next, creating a visual pipeline. 
3. Executing Workflows: The system executes only the parts of the workflow that need to change, making it efficient. 
4. Sharing: Users can save and share their completed workflows as PNG files, which can then be loaded by others by dragging them onto the ComfyUI application. 

*******************************************************************************************************************************************************************

## Setup

### Create the SSD layout (all data lives under /ssd/comfyui)

    sudo mkdir -p /ssd/comfyui/{compose,models,custom_nodes,input,output,cache,pip-cache,pip}
    sudo chown -R "$USER":"$USER" /ssd/comfyui
    chmod -R u+rwX,g+rwX /ssd/comfyui

### Pin the Python stack inside the container via pip config + constraints
This ensures custom-node installs cannot upgrade torch/numpy.

    # Pip config: prefer PyPI; keep Jetson wheels as an extra index; cache on SSD
    cat > /ssd/comfyui/pip/pip.conf <<'PIP'
    [global]
    index-url = https://pypi.org/simple
    extra-index-url = https://pypi.jetson-ai-lab.dev/jp6/cu126
    timeout = 60
    disable-pip-version-check = true
    PIP
    
    # Constraints: pin critical runtime
    cat > /ssd/comfyui/pip/constraints.txt <<'PIN'
    # Hard pins that must never change
    numpy==1.26.4
    torch==2.5.0
    torchvision==0.20.0
    xformers==0.0.30
    PIN

The NVIDIA/Jetson ComfyUI image already comes with torch 2.5.0 (CUDA) and matching torchvision/xformers. These pins stop later installs from “helpfully” upgrading them.

### Bring up the container
Run the docker-compose file

    cd /ssd/comfyui/compose
    docker compose up -d

Verify it's healthy and listening

    docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}'
    docker logs --tail=120 comfyui

### Sanity-check the pinned runtime (CUDA torch + numpy 1.26.4)

    docker exec -it comfyui python3 - <<'PY'
    import torch, numpy as np, os
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("numpy:", np.__version__)
    print("constraints:", os.environ.get("PIP_CONSTRAINT"))
    PY

### Install ComfyUI-Manager (the only custom node) inside the container
Done inside the container so host Python stays untouched. We also install Manager’s lightweight deps; the pins prevent any torch/numpy changes.

    docker exec -it comfyui bash -lc '
    set -e
    cd /opt/ComfyUI/custom_nodes
    if [ ! -d ComfyUI-Manager ]; then
      git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager.git
    fi
    # Minimal, safe deps for the Manager (will respect constraints)
    pip install --no-cache-dir GitPython toml matrix-nio uv
    '
Restart once so the Manager UI appears cleanly:

    docker restart comfyui

Check logs show Manager loaded (no errors), and that the server is started:

    docker logs --tail=200 comfyui

Open the UI: http://<jetson-ip>:8188
You should see the Manager tab to install models/custom nodes from the UI itself.

### Quality-of-life commands

Start/Stop :

    docker start comfyui
    docker stop comfyui

Live logs : 

    docker logs -f comfyui

Destroy the container (keep all your SSD data) : 

    docker rm -f comfyui

Bring it back up (same config) : 

    cd /ssd/comfyui/compose
    docker compose up -d

Change runtime flags (e.g., add --extra-model-paths /opt/ComfyUI/other_models or tweak --lowvram → remove it) – edit the COMFYUI_ARGS in /ssd/comfyui/compose/docker-compose.yml, then:

    docker compose up -d --force-recreate

Single “kill everything ComfyUI” cleanup (container + image; keeps your SSD folders):

    docker rm -f comfyui 2>/dev/null || true
    docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' \
     | awk '/comfyui/ {print $2}' | xargs -r docker rmi -f
