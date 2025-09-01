# OPEN WEB UI

<img width="1916" height="981" alt="image" src="https://github.com/user-attachments/assets/0d4c10da-235c-4296-b151-12c53f2995b6" />


https://github.com/user-attachments/assets/fa38dda3-ef41-4131-b1f1-dc45c73824b3


https://github.com/user-attachments/assets/22d66c32-2319-4e60-a829-0822d5c15432

## Model Selection

All the models that you download will appear in this drop-down list
<img width="1297" height="697" alt="image" src="https://github.com/user-attachments/assets/a2702883-4b72-4ddd-8d43-eff1973fe025" />

In the first prompt with a freshly selected model, it will take a few more seconds because the model needs to load on the RAM. Once loaded, the response time is much faster.

## RAM usage during inferencing

This is for the llama3.2:3b model only
<img width="1821" height="172" alt="image" src="https://github.com/user-attachments/assets/508e381f-f1ba-4bef-9168-9b93e3df3f3a" />

For 8b model the RAM gets completely filled up
<img width="1847" height="160" alt="image" src="https://github.com/user-attachments/assets/4252f1de-ac94-4242-b9e0-e15995309298" />

So, for > 7b models, RAM becomes a bottle neck.
GPU, CPU and SoC temperatures did not cross 65 deg. for me atleast in case of chat based inferencing.

## Setup

My setup of Open WEB UI is based on the usage of the ollama server that is setup based on the information in sub dir : 01_setting-up-ollama-server. Once that is done, the you can proceed with the following -
1. Create a directory in your /ssd

       mkdir -p /ssd/openwebui
   
2. Then run the shell script : setup_openwebui.sh (you would definitely have to do chmod before running

       chmod +x /ssd/openwebui/setup_openwebui.sh
       bash /ssd/openwebui/setup_openwebui.sh
   
## Explanation of the setup

 - The setup assumes you already have ollama setup and its container running
 - It modifies the docker-compose.yml file of the ollama itself. To confirm if the modification has been performed correctly, I have added the final docker-compose.yml file in this sub-dir
 - It expects the models of ollama to be present in /ssd/ollama/models. Thats where open Web UI will also pick the models from. No extra space needed as these models can grow too big in size.
 - Checks Docker/Compose, confirms /ssd is writable, confirms your Ollama container is running.
 - Creates SSD folders:
 -   /ssd/openwebui/data (WebUI DB/uploads/indexes/settings)
 -   /ssd/openwebui/library (read-only doc/image shelf for WebUI)
 - Writes a safe docker-compose.override.yml next to your existing /ssd/ollama/compose/docker-compose.yml (so you don’t modify your working Ollama compose file and avoid the YAML mistakes you hit earlier).
 - Validates compose & pulls the ghcr.io/open-webui/open-webui:main image.
 - Starts the open-webui container and waits for HEALTHY.
 - Smoke-tests HTTP endpoints and container DNS (fixing the “Temporary failure in name resolution” you saw earlier by setting dns:).
 - Pre-pulls a local embeddings model (nomic-embed-text) in Ollama so Open WebUI can use it without any Hugging Face downloads.
 - You’ll see clear [INFO]/[OK]/[WARN] messages for every step; if any error occurs, the script stops and prints the line number.

## Usage

1. You can simply do

       docker start open-webui

and start using it.

2. If you want to stop it,

       docker stop open-webui
and the container will be stopped.

3. If you want to follow the logs :
      
       docker logs -f open-webui

4. Sanity checks 

       curl -s http://127.0.0.1:3000/ | head -n1
       curl -s http://127.0.0.1:11434/api/version && echo

5. You can check whether the container is running or not

       docker ps


