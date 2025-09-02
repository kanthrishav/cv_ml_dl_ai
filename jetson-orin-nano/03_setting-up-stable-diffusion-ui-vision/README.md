# Stable Diffusion UI

## Overview

<img width="1917" height="973" alt="image" src="https://github.com/user-attachments/assets/bf9d8635-bba0-47f0-bbaf-f4f15b688de6" />

<img width="1915" height="977" alt="image" src="https://github.com/user-attachments/assets/62d71f43-7e48-40c6-b9e4-3398cc0d1c80" />

<img width="1918" height="978" alt="image" src="https://github.com/user-attachments/assets/a1394b4a-b35c-4743-a2ca-b01fd8688c3a" />



A Stable Diffusion UI is a user-friendly, browser-based interface for the Stable Diffusion AI model, which allows users to generate and edit images without needing to use code. The most well-known of these interfaces is the AUTOMATIC1111 Stable Diffusion WebUI. 

### Key features of the Stable Diffusion WebUI
 - Text-to-Image (txt2img): Generate an image from a descriptive text prompt.
 - Image-to-Image (img2img): Transform an existing image into a new one based on a text prompt. You can provide an initial image and have the AI alter its style, contents, or composition.
 - Inpainting and Outpainting: Edit specific, masked parts of an image or extend the canvas beyond the original boundaries to create new, coherent content.
 - Upscaling: Enhance the resolution of a generated image, adding details to increase its size and quality.

Customizable parameters: Adjust various settings to control the generation process, including:
 - Sampling method: The algorithm used to generate the final image.
 - Sampling steps: The number of iterations the model takes; more steps can lead to higher-quality results.
 - CFG scale: Determines how strictly the model should adhere to your prompt.
 - Extensions and custom models: The interface supports a wide variety of community-developed plugins and custom models (checkpoints). These can add new features or allow you to generate images in specific artistic styles.
 - Workflow tools: Features like prompt matrices, batch processing, and the ability to save image generation data (in PNG info) help streamline the creativ


## Setup

The setup is pretty simple if you force xFormers to not be installed because there is a major issue of xFormers on Jetson platforms with getting a CUDa enabled torch.

So the following setup - 
 - runs AUTOMATIC1111 Stable Diffusion WebUI in Docker on your Jetson
 - forces xFormers OFF (and overrides images that would enable it by default)
 - prevents any auto-downloads (like SD-1.5)
 - reuses your ComfyUI models (checkpoints, VAE, LoRA, upscalers, CLIP, embeddings) from the sub-dir 04_setting-up-comfyui-vision
 - binds Gradio on 0.0.0.0 so the UI opens from your LAN

1. Create required directories

       sudo mkdir -p /ssd/stablediffUI/{models,custom_nodes,input,output}
       sudo chown -R "$USER:$USER" /ssd/stablediffUI
   
2. Bring up the docker container

       docker compose -f /ssd/stablediffUI/docker-compose.yml pull
       docker compose -f /ssd/stablediffUI/docker-compose.yml up -d

3. Verifications
    
       # prove the running command has NO --xformers and DOES have our flags
       docker inspect stablediffUI --format '{{.Path}} {{.Args}}'
       
       # show it bound properly & didn’t try to download a model
       docker logs -n 200 stablediffUI | egrep -i 'Running on|local URL|Downloading:|xformers|attention' || true

4. Daily usage

       # start / stop
       docker start stablediffUI
       docker stop  stablediffUI
       
       # show status / logs
       docker ps -a | grep stablediffUI
       docker logs -f stablediffUI
       
       # apply config changes (edit compose, then:)
       docker compose -f /ssd/stablediffUI/docker-compose.yml up -d
       
       # kill immediately (keeps volumes)
       docker kill stablediffUI

Thats it. You can start using it.


