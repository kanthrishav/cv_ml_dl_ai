# NVIDIA JETSON ORIN NANO 8GB

![jetson-orin-nano-super-developer-kit-og](https://github.com/user-attachments/assets/88b7fafb-ce85-4a10-887c-b9c9595899c3)

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

### Check firmware/bootloader/OS/JetPack versions and compare to latest

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

### Check snapd-AppArmor and install flatpak

### Check firmware/bootloader/OS/JetPack versions and compare to latest

### Setting up VNC and accessing Jetson via RealVNC

###  Checking health of docker engine, nvidia container packages, jetson containerization packages, CUDA, tensorRT and other requirements for LLM/VLM inference


