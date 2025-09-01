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
4. Run it

        chmod +x setup_ollama_on_ssd.sh
        ./setup_ollama_on_ssd.sh

That’s it. You will have a working ollama CLI that uses the container under the hood, all state on /ssd, and a container that stays up.

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
If you don't want to auto-start anymore, run the command
        
        sudo systemctl disable --now ollama-compose

### Usage help

		# Show help (via wrapper)
		ollama --help

		# Pull a small sanity model
		ollama pull llama3.2:3b

		# Run an inference
		ollama run llama3.2:3b "Say hello from Jetson."

		# Check server API directly
		curl -s http://127.0.0.1:11434/api/version && echo

## Detailed explaination of the setup

### How to read the installer

The script is deliberately chatty: it prints status [INFO], [ OK ], [WARN], [ERR ] so you always know where you are. It exits on the first error so you don’t end up in a half-configured state.

At the top of the script you’ll see:

    #!/usr/bin/env bash
    set -Eeuo pipefail

 - #!/usr/bin/env bash says “run this with bash.”
 - set -E keeps error traps across functions/subshells.
 - set -e means “exit immediately if any command fails.”
 - set -u means “error if you reference an undefined variable.”
 - set -o pipefail means “if any command in a pipeline fails, treat the pipeline as failed.”
 - This is basic “safety mode” for shell scripts: it avoids silent errors and makes failures visible right away.
 - Alternatives: we could run without these safeguards and manually check every exit code, but that’s error-prone and easy to miss.

### Configuration block

    SSD_ROOT="/ssd"
    APP_ROOT="$SSD_ROOT/ollama"
    IMAGE_REPO="dustynv/ollama"
    IMAGE_TAG="${IMAGE_TAG:-r36.4.0}"   # override by exporting IMAGE_TAG
    CONTAINER_NAME="ollama"
    HOST_PORT="11434"
    COMPOSE_DIR="$APP_ROOT/compose"
    BIN_DIR="$APP_ROOT/bin"
    MODELS_DIR="$APP_ROOT/models"
    DATA_DIR="$APP_ROOT/data"
    LOGS_DIR="$APP_ROOT/logs"
    WRAPPER="$BIN_DIR/ollama"

 - Where we put things, and which container image we’ll run.
 - IMAGE_TAG defaults to r36.4.0 because we learned together that r36.4.4 doesn’t exist and causes pull failures. You can change it by running export IMAGE_TAG=... before launching the script.
 - HOST_PORT is the TCP port on the Jetson where the Ollama HTTP API will be available.

Alternatives:
We could have used the upstream ollama/ollama image. On Jetson, it’s safer to use Jetson-optimized images (dustynv/ollama) since they’re built with the right userspace for JetPack/L4T and avoid CUDA/TensorRT mismatches.

### Logging helpers and "die fast"
    
    info(){ printf "\033[1;36m[INFO]\033[0m %s\n" "$*"; }
    ok(){   printf "\033[1;32m[ OK ]\033[0m %s\n" "$*"; }
    warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
    err(){  printf "\033[1;31m[ERR ]\033[0m %s\n" "$*"; }
    die(){ err "$1"; exit 1; }
    require_cmd(){ command -v "$1" >/dev/null 2>&1 || die "Missing '$1'. Install it and re-run."; }

 - These give consistent, color-coded messages.
 - require_cmd verifies a command exists before we need it.
 - Clear user feedback. Scripts that silently fail or print raw errors are confusing.
 - Alternatives: we could have relied on raw echos, but explicit levels + colors make it easier to follow.

### Pre-flight checks

    require_cmd docker
    docker version ...
    require_cmd "docker compose"

 - Ensures Docker and the Compose plugin are installed and the Docker daemon is running. If not, we stop early with a clear error.
 - This makes the installer fail fast with instructions.

### SSD mount safety check

    if ! mountpoint -q "$SSD_ROOT"; then
      warn "$SSD_ROOT exists but is not a mountpoint ... Aborting for safety."
      die "Please mount your SSD at $SSD_ROOT ..."
    fi

 - We refuse to continue if /ssd isn’t a real mount.
 - This prevents writing everything to the microSD by accident (which is exactly what you said to avoid).
 - Alternatives: We could create the folder anyway and proceed, but then all data might land on the microSD. Safer to abort and have you mount the SSD properly (via /etc/fstab with the disk’s UUID).

### Create the directory layout on the SSD

    sudo mkdir -p "$MODELS_DIR" "$DATA_DIR" "$LOGS_DIR" "$BIN_DIR" "$COMPOSE_DIR"
    sudo chown -R "$(id -un)":"$(id -gn)" "$APP_ROOT"

 - All Ollama state lives under /ssd/ollama/:
 - models — actual model files
 - data — Ollama’s state/cache (blobs, manifests)
 - logs — API/server logs
 - bin — our wrapper CLI lives here
 - compose — the docker-compose.yml
 - Easy to back up/replicate; guaranteed SSD storage; nothing touches the microSD.
 - Alternatives: Could use default Docker volume locations (/var/lib/docker etc.) but that risks storing on the microSD if Docker’s data-root is not on SSD.
 - We already have Docker’s root at /ssd/docker; even then, bind-mounts like we do make it crystal clear where your important data lives.

### Pull the image with Compose

    docker compose -f "$COMPOSE_FILE" pull

 - Downloads dustynv/ollama:r36.4.0 locally. Size >2 GB is normal for Jetson images with CUDA/TensorRT userspace.
 - Alternatives: docker pull dustynv/ollama:r36.4.0 does the same for a single service, but Compose pull is aligned with your YAML.

### Start the container (detached)

    docker compose -f "$COMPOSE_FILE" up -d

 - “Create if needed and start in the background.”
 - We sleep briefly, then confirm it’s Up (not “Exited”). If it isn’t, we dump the last 120 log lines and abort.
 - If the “Container … Started” but docker ps shows no running containers—because some PID exited. This check catches that.

### Health check: wait for the HTTP API

    until curl -fsS "http://127.0.0.1:${HOST_PORT}/api/version" ...

 - We poll /api/version until it responds. If the API isn’t ready, we wait; if it never comes up, we show logs and exit.
 - It’s the easiest way to know the server is truly listening and usable.
 - Alternatives: we could rely on logs alone, but hitting the API is definitive.

### Install the wrapper CLI (/ssd/ollama/bin/ollama)

    NAME="ollama"
    COMPOSE_DIR="/ssd/ollama/compose"
    # ... helper _docker that falls back to sudo ...
    if ! _docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      ( cd "$COMPOSE_DIR" && _docker compose up -d )
    fi
    # If OLLAMA_HOST is set, pass it into the container env for API mode
    exec _docker exec -e OLLAMA_HOST="$OLLAMA_HOST" -it "$NAME" ollama "$@"

 - This gives you a native-feeling ollama command without installing anything on your OS.
 - It auto-starts the container if it’s not running.
 - It supports both: Direct CLI mode (runs ollama inside the container) & API mode (if you set OLLAMA_HOST=http://JETSON:11434, the CLI inside the container talks to that address—useful for remote servers)
 - It tries docker first and falls back to sudo docker so you don’t get permission errors if your user isn’t in the docker group.
 - It meant to keep everything on SSD and nothing on the microSD/OS (root volume). If we installed a host binary under /usr/local/bin, it would go on the system drive; this wrapper keeps everything in /ssd/ollama/bin.

### Explaintion of the docker-compose file

 - services: is the top-level section—here we only have one service: ollama.
 - image: is which container to run (dustynv/ollama:r36.4.0). We use this tag because:
 - It exists (unlike r36.4.4, which caused a pull error earlier).
 - It’s aligned with L4T/JetPack r36.4 family used by your system.
 - container_name: fixes the name (ollama) so our wrapper can address it.
 - restart: unless-stopped tells Docker to bring it back after reboots/crashes unless you manually docker compose down. (You asked for a version that doesn’t auto-boot previously; this one does auto-restart as a convenience. If you want no auto-restart, set restart: "no".)
 - ports: exposes the container’s port 11434 as host port 11434 (the Ollama HTTP API).
 - environment: configures Ollama:
 - OLLAMA_MODELS=/models tells it where to store models (inside the container).
 - OLLAMA_HOST=0.0.0.0:11434 makes it listen on all interfaces at 11434.
 - OLLAMA_LOGS=/data/logs/ollama.log points its logs to a file under /data/logs (which we bind to /ssd/ollama/logs).
 - volumes: bind mount host directories into the container:
 -    /ssd/ollama/models → /models
 -    /ssd/ollama/data → /root/.ollama
 -    /ssd/ollama/logs → /data/logs
 -    This guarantees everything lands on your SSD.
 -    entrypoint: ["/bin/bash","-lc","exec ollama serve"] is the key to override the entrypoint to launch ollama serve in the foreground.
 -    exec replaces the shell with the server, so the server becomes PID 1. As long as it’s running, the container stays “Up.”
 -    runtime: nvidia grants GPU access on Jetson (uses NVIDIA Container Toolkit).

### Few helpers

1. If you want to change the image later, example

       export IMAGE_TAG=r36.3.0   # or a 22.04/CUDA-qualified tag you’ve checked exists
       ./setup_ollama_on_ssd.sh

2. If you want to cleanly stop/remove everything (without losing any models)

       cd /ssd/ollama/compose
       docker compose down

3. To access ollama from a different device on the same network

       curl http://JETSON_IP:11434/api/version
