# OLLAMA

## Overview 

Ollama is an open-source software that simplifies running and managing large language models (LLMs) directly on your local computer, rather than relying on cloud-based services. It provides a user-friendly way to download, customize, and interact with a variety of LLMs, such as those from Meta and Mistral, ensuring data privacy by keeping all processing on your own device. Ollama also offers a command-line interface (CLI) and an API for programmatic access, making it a versatile tool for developers, data scientists, and individuals

### Key Features and Benefits:
1. Local Execution:
Runs LLMs locally on your machine (macOS, Linux, Windows), enhancing data privacy and security by keeping data on your device. 
2. Simplified Workflow:
Makes it easy to download, manage, and run various open-source LLMs with a straightforward command-line interface. 
3. Open-Source Focus:
Primarily works with open-source models, fostering transparency and allowing for extensive customization. 
4. Customization:
Users can customize models using a system called Modelfile to adjust parameters like creativity (temperature) or to set specific instructions (system messages). 
5. API Access:
Provides an API to integrate LLM functionality into other applications and services. 
6. Flexibility:
Supports various open-source LLMs, allowing users to experiment with different models for coding assistance, content generation, and other tasks. 

### How it Works:
Ollama acts as a bridge between your computer and an LLM, handling the complex task of deploying and running these models locally. By downloading models through Ollama and using its provided interfaces, you can have a personalized AI environment without needing to manage complex dependencies or send data to external servers. 

## Setup

1. You need to run the pre docker build shell script "check_pre_container_setup.sh" first.
2. docker-compose.yml file can be found inside /compose dir
3. Then you can run "build_container.sh"

### What you will end up with

 - directories : /ssd/ollama/{models,data,logs,bin,compose} (all Ollama data on SSD)
 - A robust ollama wrapper CLI (uses the running container)
 - A quick health check that verifies the API

### What the shell scripts do

 - Validates Docker & Compose — to avoid “command not found” surprises.
 - Refuses to continue if /ssd isn’t a mountpoint — prevents accidentally writing to the microSD.
 - Creates a clean tree on SSD
      /ssd/ollama/models   # models
      /ssd/ollama/data     # ollama internal state
      /ssd/ollama/logs     # API/server logs
      /ssd/ollama/bin      # wrapper CLI
      /ssd/ollama/compose  # docker-compose.yml
- Uses entrypoint: ["bash","-lc","exec ollama serve"] so the server is PID 1 and the container stays alive
- Uses environment OLLAMA_HOST=0.0.0.0:11434 (your image does not support --host)
- Keeps runtime: nvidia for Jetson
- Pulls the image with progress (the tag defaults to r36.4.0 because we previously learned r36.4.4 doesn’t exist).
- Starts and verifies the container is Up (not Exited).
- Polls the HTTP API so we know the service is actually listening.
- Installs a robust wrapper at /ssd/ollama/bin/ollama that:
- Starts the container if it’s not running
- Falls back to sudo docker if the user isn’t in the docker group
- Forwards OLLAMA_HOST if you want to point the CLI at a remote server
- Adds the wrapper to PATH (idempotent).
- This directly addresses all the errors we hit earlier:
- wrong tag → fixed default (r36.4.0)
- YAML broken by quotes/indent → clean YAML, validated with docker compose config
- container exiting → foreground entrypoint with exec ollama serve
- --host unsupported → use OLLAMA_HOST env instead
- wrapper “container not running” → wrapper now auto-starts the stack

If you later want to auto-start ollama on boot you can run this shell script : auto_start_setup.sh

