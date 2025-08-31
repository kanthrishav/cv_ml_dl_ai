# NVIDIA JETSON ORIN NANO 8GB

<img width="600" height="315" alt="down_scale_00006_" src="https://github.com/user-attachments/assets/4543774c-58ea-4ee7-b3f0-c66c7d7f474a" />

## Overview 

In one sentence - its a tiny monster!
One can either purchase the Orin module and then purchase the carrier board separately. But I would suggest purchasing the complete developer kit. The original name of the dev kit was "Jetson Orin Nano 8GB Developer Kit". After a major SW upgrade they renamed it to "Jetson Orin Nano 8GB Super Developer Kit". The 'super' refers to some SW optimizations performed which increases the performance in multiples.

In the box you will get the Orin module attached to the carrier board and the power adaptor. 

Specifications : 
 - Input Power : 7 W / 15 W / 25 W / 'MAXN POWER'
 - CPU : 6-core ARM Cortex-A78AE
 - GPU : Ampere Arch., 1024 Nvidia CUDA cores, 32 x 3rd Gen Tensor Cores
 - Memory : 8 GB, 128-bit LPDDR5" 68 GB/s (max) (shared RAM between CPU & GPU)
 - Ethernet : 1 x RJ45 Gigabit Ethernet
 - USB connector : 4 x USB 3.2 Type A connector, 1 x USB-C port for only data
 - Camera port : 2 x CSI Camera (MIPI CSI-2)
 - Storage : microSD card slot, NVMe M.2 2242/2280 PCIe Gen3 SSD slot
 - Display :  1 x Dsiplay Port 1.2

Before proceeding towards the setup, I will provide a way to starting with this board activated for highest performance. I went down multiple routes mainly to explore. So, I will say this - for running any kind of LLM/VLM models you will need the following : 
1. Minimum 8 GB RAM variant (4 GB could not handle AI models in a way that was useful to me)
2. Minimum 128 GB microSD card for only OS and to act as your root volume
3. Minimum 1 TB NVMe SSD (for storing a plethora of AI model checkpoints, docker containers, Lora adapters, etc)


Based on my experience, if you miss either of these 3 requirements, you will know very very very soon that you need those now!
 - I have all my containers and all the AI models present in the NVMe SSD.
 - Currently 53 GB of microSD and 440 GB of SSD is already consumed.
 - most 8B models with their docker containers fills up the 90% of RAM  (7.4 GB).
 - coding agent alone fills up the RAM when prompting to the LLM model.
 - So, you get the picture.
 - If you purchase a Gen4 NVMe SSD, the performance will be downgraded to Gen3.
 - The next Jetson board costs 4 times than this one. So, Nvidia really brought down the cost with this one.

Apart from the Dev Kit I used 
- HP 128GB MicroSD Memory Card SDXC mx310 Class 10 UHS-I U1 Card,
- WD_Black Western Digital SN7100 1TB PCIe Gen 4 NVMe SSD M.2 (2280)
- a display monitor,
- a display port to HDMI cable (since my monitor takes HDMI input only),
- keyboard
- & mouse.

Landing page of the product from Nvidia : https://www.nvidia.com/en-in/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
Datasheet from Nvidia : https://nvdam.widen.net/s/zkfqjmtds2/jetson-orin-datasheet-nano-developer-kit-3575392-r2

## Flashing the OS

The OS is called "Jetpack" which is a variant of Ubuntu. Refer to this link to know more about the contents of the Jetpack OS : https://developer.nvidia.com/embedded/jetpack

I followed the information present on the Nvidia's official instructions website and the following installation video shared by Bijan Bowen
1. https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit
2. The above page kind of takes you to multiple pages based on your specific board
3. https://www.youtube.com/watch?v=-PjMC0gyH9s&t=2966s

The above links were enough to handle any flashing and first bootup related issues.

## Post first boot-up

There are few steps I would strongly suggest before beginning any project on the board - 
1. Checking OS and firmware version and confirming with the latest version available for the board
2. For me certain applications like the web browser couldn't launch at all - snapd-AppArmor issue
3. Setting up VNC on Jetson and accessing it via RealVNC (I got into a lot of errors while doing this)
4. Checking health of docker engine, nvidia container packages, jetson containerization packages, CUDA, tensorRT and other requirements for LLM/VLM inference

### 1. Check firmware/bootloader/OS/JetPack versions and compare to latest

This prints what you have and (best-effort) scrapes NVIDIA pages to show the current latest so you can see if you’re up-to-date. Latest JetPack/L4T references: JetPack page and r36.4.4 release notes.

		set -euo pipefail

		echo "=== Host / kernel / Ubuntu ==="
		uname -a
		lsb_release -a || cat /etc/os-release

		echo "=== L4T / JetPack mapping (L4T string) ==="
		# Shows L4T line; e.g. "# R36 (release), REVISION: 4.4, ..."
		head -n1 /etc/nv_tegra_release || true
		dpkg -l | egrep -i '^ii\s+nvidia-l4t|^ii\s+nvidia-jetpack' || echo "JetPack meta package not installed (normal on many images)."

		echo "=== Key NVIDIA stack ==="
		dpkg -l | egrep -i 'cuda-|^ii\s+nvidia-l4t-|cudnn|tensorrt|nvinfer' | awk '{print $1,$2,$3}' || true

		echo "=== UEFI/bootloader (Orin: check UEFI in firmware menu if needed) ==="
		# On Orin devkits, UEFI version is shown in the UEFI menu (ESC at boot). For reference:
		echo "Tip: Press ESC during boot to read UEFI version as per NVIDIA setup guide."

		# --- Compare with latest (best-effort scrape) ---
		echo "=== Latest (online) per NVIDIA (best-effort) ==="
		LATEST_JETPACK=$(curl -fsSL https://developer.nvidia.com/embedded/jetpack | grep -oE 'JetPack [0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)
		echo "Latest on JetPack page: ${LATEST_JETPACK:-unknown}"
		LATEST_L4T=$(curl -fsSL https://docs.nvidia.com/jetson/archives/r36.4.4/ReleaseNotes/Jetson_Linux_Release_Notes_r36.4.4.pdf 2>/dev/null >/dev/null && echo "r36.4.4" || echo "unknown")
		echo "Latest Jetson Linux tested here: ${LATEST_L4T}"

### 2. Check snapd-AppArmor and install flatpak

On some Jetson kernels, AppArmor isn’t enabled ⇒ many snaps won’t run. These commands detect that and either enable what we can (user namespaces), or tell you to prefer Flatpak. (We’ll set up Flatpak in Set 4.)

		set -euo pipefail

		echo "=== Install & enable snapd ==="
		sudo apt update
		sudo apt install -y snapd apparmor apparmor-utils || true
		sudo systemctl enable --now snapd

		echo "=== Check kernel features needed by snap ==="
		if command -v aa-status >/dev/null 2>&1; then
		  aa-status || true
		else
		  echo "AppArmor tools not present."
		fi

		# Enable user namespaces (needed by snap sandboxes and Chromium/Firefox)
		echo 'kernel.unprivileged_userns_clone=1' | sudo tee /etc/sysctl.d/90-userns.conf
		sudo sysctl --system | grep -E 'userns_clone|app' || true

		echo "=== Quick snap sanity ==="
		snap version || true
		snap debug sandbox-features || true

		echo "=== Test-run a trivial snap ==="
		# This will fail cleanly if confinement isn’t supported; that’s OK (we’ll use Flatpak then)
		sudo snap install hello-world || true
		snap run hello-world || true

		echo "Note: if sandbox-features shows AppArmor not enabled, prefer Flatpak (see Set 4)."

If your log says something like these:

 - 'aa-status' printed “apparmor not present.” → the AppArmor LSM isn’t enabled in the kernel.
 - snap debug sandbox-features didn’t show AppArmor enforcement, and snap run hello-world failed with cannot set capabilities: Operation not permitted → classic symptom when snaps try to sandbox without AppArmor support.
 - Enabling kernel.unprivileged_userns_clone=1 helps snaps in general, but doesn’t replace AppArmor.

If you still want to double confirm you can run the following 

  		# Is AppArmor built/loaded?
		grep -i apparmor /proc/cmdline
		cat /sys/module/apparmor/parameters/enabled  # likely: "No such file or directory"
		dmesg -T | grep -i apparmor || echo "No AppArmor messages in dmesg"

If AppArmor is not enabled in the kernel, you can go for flatpak installation (it uninstalls any Chromium you might have installed during first bootup)

		set -euo pipefail

		sudo apt update
		sudo apt install -y flatpak
		sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

		# Remove browser snaps/apt packages so there’s only one copy on the system
		sudo snap remove firefox chromium 2>/dev/null || true
		sudo apt purge -y firefox chromium-browser 2>/dev/null || true
		sudo apt autoremove -y || true

		# Install browsers from Flathub (multi-arch, no AppArmor required)
		sudo flatpak install -y flathub org.mozilla.firefox
		sudo flatpak install -y flathub org.chromium.Chromium

		# Make desktop launchers prefer Flatpak Firefox for xdg-open (optional helper)
		cat <<'SH' | sudo tee /opt/ai/bin/open-url >/dev/null
		#!/bin/sh
		exec flatpak run org.mozilla.firefox "$@"
		SH
		sudo chmod +x /opt/ai/bin/open-url

		echo "Flatpak browsers installed. Use:"
		echo "  flatpak run org.mozilla.firefox"
		echo "  flatpak run org.chromium.Chromium"

### 3. Setting up VNC and accessing Jetson via RealVNC

My aim was to be able to mirror the local monitor's display in RealVNC.

The following set of commands will setup the following : 
 - 	Display manager: LightDM (set as default), claiming VT1 so the monitor always shows GUI.
 - 	Display server : Xorg on :0 (not Wayland)
 - 	Desktop : GNOME shell (Ubuntu session) on Xorg
 - 	VNC server : TigerVNC x0vncserver run a a root systemd service, which
 - 	- wait sofr /tmp/.X11-unix/X0
    - uses LIghtDM's Xauthority (-auth /var/run/lightdm/root/:0 ),
    - mirrors the physical desktop to TCP 5900
    - survives reboots
So, what you will see in RealVNC is a mirror of your actual local GNOME-on-Xorgdesktop - not a separate virtual session and not GNOME's built-in (remote-desktop)

**If you want a different VNC password than jetvnc12, change the VNC_PASS line (VNC only uses the first 8 characters—keep it 6–8 chars)**

		set -euo pipefail
		echo "== 0) Minimal helpers (incl. .deb zstd repack tool) =="
		if [ ! -x /usr/local/bin/repack-deb-zstd ]; then
		  sudo tee /usr/local/bin/repack-deb-zstd >/dev/null <<'EOF'
		#!/usr/bin/env bash
		set -euo pipefail
		for deb in "$@"; do
		  [ -f "$deb" ] || continue
		  tmp="$(mktemp -d)"; pushd "$tmp" >/dev/null
		  ar t "$deb" | grep -Eq 'control\.tar\.(zst|xz)|data\.tar\.(zst|xz)' || { popd >/dev/null; rm -rf "$tmp"; continue; }
		  ar x "$deb" debian-binary || true
		  ar x "$deb" control.tar.zst || true; [ -f control.tar.zst ] && unzstd -f control.tar.zst || true
		  ar x "$deb" control.tar.xz  || true; [ -f control.tar.xz  ] && xz -d -f control.tar.xz  || true
		  ar x "$deb" data.tar.zst    || true; [ -f data.tar.zst    ] && unzstd -f data.tar.zst    || true
		  ar x "$deb" data.tar.xz     || true; [ -f data.tar.xz     ] && xz -d -f data.tar.xz     || true
		  ar rcs "${deb%.deb}_repacked.deb" debian-binary control.tar data.tar
		  mv -f "${deb%.deb}_repacked.deb" "$deb"
		  popd >/dev/null; rm -rf "$tmp"
		done
		EOF
		  sudo chmod +x /usr/local/bin/repack-deb-zstd
		fi
		sudo apt-get update || true
		sudo apt-get install -y zstd xz-utils binutils sed grep coreutils || true
		echo "== 1) Remove earlier headless / conflicting remote bits =="
		sudo systemctl disable --now auto-headless-xorg.service auto-headless-xorg-postboot.timer auto-headless-xorg-postboot.service 2>/dev/null || true
		sudo rm -f /etc/systemd/system/auto-headless-xorg.service /etc/systemd/system/auto-headless-xorg-postboot.{service,timer} 2>/dev/null || true
		sudo rm -f /etc/X11/xorg.conf.d/20-headless.conf /etc/X11/xorg.conf.d/20-headless.conf.template 2>/dev/null || true
		# Turn off GNOME Remote Desktop (we'll use x0vncserver to avoid prompts)
		systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true
		systemctl --user disable gnome-remote-desktop.service 2>/dev/null || true
		gsettings set org.gnome.desktop.remote-desktop.vnc enable false 2>/dev/null || true
		gsettings set org.gnome.desktop.remote-desktop.rdp enable false 2>/dev/null || true
		echo "== 2) Make LightDM + GNOME on Xorg the ONLY display stack =="
		# Make LightDM default; disable GDM to prevent races
		echo "lightdm shared/default-x-display-manager select lightdm" | sudo debconf-set-selections
		sudo dpkg-reconfigure -f noninteractive lightdm || true
		sudo systemctl disable gdm3 2>/dev/null || true
		sudo systemctl enable lightdm
		sudo systemctl set-default graphical.target
		# Claim VT1 so you always land straight in the GUI (not a text TTY)
		sudo mkdir -p /etc/lightdm/lightdm.conf.d
		sudo tee /etc/lightdm/lightdm.conf.d/50-autologin-gnome.conf >/dev/null <<EOF
		[LightDM]
		minimum-vt=1
		[Seat:*]
		autologin-user=$(id -un)
		autologin-user-timeout=0
		user-session=$( for s in ubuntu-xorg ubuntu gnome-xorg gnome; do [ -f "/usr/share/xsessions/\${s}.desktop" ] && echo "\$s" && break; done )
		greeter-session=lightdm-gtk-greeter
		xserver-command=X -core
		EOF
		# Remove any per-user overrides that might force a different DE
		rm -f ~/.xsession ~/.xinitrc ~/.dmrc 2>/dev/null || true
		echo "== 3) Disable splash/plymouth so nothing 'holds' the screen =="
		sudo systemctl disable --now plymouth-start.service plymouth-read-write.service plymouth-quit.service plymouth-quit-wait.service 2>/dev/null || true
		sudo systemctl mask plymouth-start.service plymouth-read-write.service plymouth-quit.service plymouth-quit-wait.service 2>/dev/null || true
		# Jetson uses extlinux—make sure 'quiet splash' are not on the kernel cmdline
		sudo sed -i 's/\<quiet\>//g; s/\<splash\>//g' /boot/extlinux/extlinux.conf || true

		echo "== 4) Ensure TigerVNC is installed (handles dpkg zstd issue automatically) =="
		if ! command -v x0vncserver >/dev/null 2>&1; then
		  if ! sudo apt-get install -y tigervnc-standalone-server tigervnc-common; then
			sudo apt-get -y -d install tigervnc-standalone-server tigervnc-common || true
			sudo /usr/local/bin/repack-deb-zstd /var/cache/apt/archives/tigervnc-*.deb || true
			sudo dpkg -i /var/cache/apt/archives/tigervnc-*.deb || true
			sudo apt-get -f -y install || true
		  fi
		fi
		echo "== 5) Create a root-owned VNC password used by the service =="
		# IMPORTANT: VNC only uses the first 8 chars; keep it 6–8
		VNC_PASS="${VNC_PASS:-jetvnc12}"
		if [ "${#VNC_PASS}" -lt 6 ] || [ "${#VNC_PASS}" -gt 8 ]; then
		  echo "ERROR: VNC_PASS must be 6–8 characters"; exit 1
		fi
		echo -n "$VNC_PASS" | vncpasswd -f | sudo tee /etc/x0vnc.pass >/dev/null
		sudo chmod 600 /etc/x0vnc.pass
		sudo chown root:root /etc/x0vnc.pass

		echo "== 6) System service: mirror the REAL :0 (waits for X socket; uses LightDM auth) =="
		sudo tee /etc/systemd/system/x0vncserver.service >/dev/null <<'EOF'
		[Unit]
		Description=Mirror the active Xorg :0 over VNC (RealVNC compatible)
		After=lightdm.service
		Wants=lightdm.service
		[Service]
		Type=simple
		Environment=DISPLAY=:0
		# Wait deterministically for Xorg :0 socket (no random sleeps)
		ExecStartPre=/bin/bash -lc 'for i in {1..90}; do [ -S /tmp/.X11-unix/X0 ] && exit 0; sleep 1; done; echo "/tmp/.X11-unix/X0 not found"; exit 1'
		# Use LightDM's Xauthority cookie so we can attach to the logged-in :0
		ExecStart=/usr/bin/x0vncserver \
		  -display :0 \
		  -auth /var/run/lightdm/root/:0 \
		  -PasswordFile /etc/x0vnc.pass \
		  -rfbport 5900 \
		  -forever \
		  -Shared \
		  -noxdamage \
		  -AlwaysShared
		Restart=on-failure
		RestartSec=2
		[Install]
		WantedBy=multi-user.target
		EOF
		sudo systemctl daemon-reload
		sudo systemctl enable x0vncserver.service
		echo "== 7) Bring everything up NOW (local GNOME on VT1 + RealVNC mirror) =="
		sudo systemctl restart lightdm
		# Ensure we’re on the GUI VT right away (usually VT1 with the config above)
		sudo chvt 1 || true
		sudo systemctl restart x0vncserver.service

		echo "== 8) Quick verification =="
		systemctl is-active lightdm && echo "lightdm active"
		ps aux | egrep 'Xorg.*:0|gnome-shell|gnome-session' | grep -v grep || true
		ss -tulpn | awk '/:5900/{print}'
		echo "Connect RealVNC Viewer to: $(hostname -I | awk "{print \$1}"):5900  (Encryption: Let server choose)"

If you want to reset the password : 

  		echo -n 'NEWPASS' | vncpasswd -f | sudo tee /etc/x0vnc.pass >/dev/null
		sudo chmod 600 /etc/x0vnc.pass
		sudo systemctl restart x0vncserver

### 4. Checking health of docker engine, nvidia container packages, jetson containerization packages, CUDA, tensorRT and other requirements for LLM/VLM inference

 These set of commands perform the following actions : 
 - Read-only checks for Docker, compose plugin, toolkit packages, default runtime.
 - Host validation: looks for expected Jetson device nodes and CUDA/TensorRT libs (informational).
 - Container validation: pulls the smallest matching public NGC image for your L4T series, runs a one-command probe, and removes the image right after.
 - No config changes, no installs, no edits. If something is missing, the script just prints WARN/FAIL so you know what to fix next.

		set -uo pipefail
		echo "=== 7.0 Select docker command (try without sudo, then with sudo -n) ==="
		DOCKER="docker"
		if ! docker version >/dev/null 2>&1; then
		  if sudo -n docker version >/dev/null 2>&1; then
			DOCKER="sudo docker"
		  fi
		fi
		echo "Using DOCKER cmd: ${DOCKER}"
		echo
		echo "=== 7.1 Docker client/server availability ==="
		${DOCKER} --version || echo "WARN: docker client missing?"
		${DOCKER} version || echo "FAIL: cannot reach docker daemon (check group membership or service)."
		echo
		echo "=== 7.2 Daemon status & socket permissions (informational) ==="
		systemctl is-active --quiet docker && echo "docker.service: active" || echo "docker.service: NOT active"
		[ -S /var/run/docker.sock ] && stat -c 'Socket: %n  Perm: %A  Owner: %U  Group: %G' /var/run/docker.sock || echo "No /var/run/docker.sock"
		id || true
		echo
		echo "=== 7.3 Compose plugin (optional, but useful) ==="
		${DOCKER} compose version 2>/dev/null || echo "INFO: compose plugin not found (optional)."
		echo
		echo "=== 7.4 NVIDIA container toolkit presence (host) ==="
		dpkg -l | egrep 'nvidia-container-toolkit|nvidia-container-runtime' || echo "WARN: toolkit packages not found"
		command -v nvidia-ctk >/dev/null 2>&1 && nvidia-ctk --version || echo "INFO: nvidia-ctk CLI not found"
		echo
		echo "=== 7.5 Docker runtimes and default runtime ==="
		${DOCKER} info --format 'Runtimes: {{.Runtimes}}' 2>/dev/null || true
		${DOCKER} info --format 'Default Runtime: {{.DefaultRuntime}}' 2>/dev/null || true
		echo "daemon.json (first 80 lines if present):"
		sed -n '1,80p' /etc/docker/daemon.json 2>/dev/null || echo "(no /etc/docker/daemon.json)"
		echo
		echo "=== 7.6 Host NVIDIA devices & libraries (outside container) ==="
		# On Jetson there is no nvidia-smi; check device nodes and key libs.
		ls -l /dev/nvhost* /dev/nvmap /dev/nvhost-ctrl-gpu 2>/dev/null | head -n 20 || echo "WARN: expected /dev/nvhost* nodes not listed"
		ldconfig -p | egrep 'libcudart\.so|libcublas\.so|libnvinfer\.so' | head -n 20 || echo "WARN: CUDA/TensorRT libs not found in host ldconfig (may still be present in containers)."
		echo
		echo "=== 7.7 PROBE GPU ACCESS *INSIDE* A CONTAINER (pull & clean up) ==="
		# We prefer nvcr.io/nvidia/l4t-jetpack:<tag> (has CUDA/TensorRT).
		# Your board is r36.x; r36.4.0 exists publicly. If network or tag lookup fails,
		# we fall back to l4t-base:<tag> just to validate device nodes.
		L4T_SERIES=$(head -n1 /etc/nv_tegra_release | tr '[:upper:]' '[:lower:]' | sed -n 's/.*r\([0-9]\+\).*revision: \([0-9]\+\).*/r\1.\2/p')
		: "${L4T_SERIES:=r36.4}"
		# Candidate tags for l4t-jetpack (don’t assume .4 exists on NGC)
		CANDIDATE_TAGS=("${L4T_SERIES}.0" "${L4T_SERIES}.1" "${L4T_SERIES}.2" "${L4T_SERIES}.3" "${L4T_SERIES}" "r36.3.0" "r36.2.0" "r36.1.0")

		TEST_IMAGE=""
		for tag in "${CANDIDATE_TAGS[@]}"; do
		  if ${DOCKER} manifest inspect "nvcr.io/nvidia/l4t-jetpack:${tag}" >/dev/null 2>&1; then
			TEST_IMAGE="nvcr.io/nvidia/l4t-jetpack:${tag}"
			break
		  fi
		done
		if [ -z "${TEST_IMAGE}" ]; then
		  # Fallback: base image (lighter; may not have TensorRT, but devices will be visible)
		  for tag in "${CANDIDATE_TAGS[@]}"; do
			if ${DOCKER} manifest inspect "nvcr.io/nvidia/l4t-base:${tag}" >/dev/null 2>&1; then
			  TEST_IMAGE="nvcr.io/nvidia/l4t-base:${tag}"
			  break
			fi
		  done
		fi
		if [ -z "${TEST_IMAGE}" ]; then
		  echo "FAIL: Could not find a suitable test image on NGC (network / tag issue)."
		  echo "      You can manually check: https://catalog.ngc.nvidia.com/orgs/nvidia/containers"
		else
		  echo "Using test image: ${TEST_IMAGE}"
		  # Pull, run a quick probe, then delete the image
		  if ${DOCKER} pull "${TEST_IMAGE}"; then
			${DOCKER} run --rm --gpus all "${TEST_IMAGE}" /bin/bash -lc \
			  'echo "Devices:"; ls -l /dev/nvhost* /dev/nvmap /dev/nvhost-ctrl-gpu 2>/dev/null | head -n 20; \
			   echo -e "\nCUDA libs:"; ldconfig -p | egrep "libcudart\.so|libcublas\.so" | head -n 10 || true; \
			   echo -e "\nTensorRT libs:"; ldconfig -p | egrep "libnvinfer\.so" | head -n 10 || true'
			# Clean up the image we pulled
			${DOCKER} image rm -f "${TEST_IMAGE}" >/dev/null 2>&1 || true
			echo "Cleaned test image: ${TEST_IMAGE}"
		  else
			echo "FAIL: docker pull ${TEST_IMAGE} failed (network/auth?). Skipping container probe."
		  fi
		fi
		echo
		echo "=== 7.8 SUMMARY (what to look for) ==="
		echo "- PASS if: docker.client+server shown, daemon active, Default Runtime: nvidia"
		echo "- PASS if: host shows /dev/nvhost* nodes"
		echo "- PASS if: container probe lists /dev/nvhost* and at least libcudart.so (TensorRT libs shown if using l4t-jetpack)"
		echo "- If any FAIL/WARN above, paste that section and we’ll zero in without changing your system."

## Beyond setting up

I have already tried out and using the following AI inference servers, UIs and setups
1. ollama server with the following models
 - dolphincoder:7b-starcoder2-q4_K_M     4.6 GB
 - codeqwen:7b                           4.2 GB
 - deepseek-coder:6.7b-instruct-q4_K_M   4.1 GB
 - deepseek-coder:6.7b-instruct          3.8 GB
 - llama3.1:8b                           4.9 GB
 - qwen2.5-coder:7b                      4.7 GB
 - llava:7b                              4.7 GB
 - moondream:latest                      1.7 GB
 - granite3.2:8b                         4.9 GB
 - llama3.1:8b-instruct-q4_K_M           4.9 GB
 - llama3.2:3b                           2.0 GB

You can see the storage space these smaller models take.

2. Open webUI and using the ollama server in the backend to get a chatgpt type chat interface
3. Stable Diffusion to get a UI for generating images
4. ComfyUI for creating larger vision workflows
5. Building a generic coding agent that uses ollama server in the backend
