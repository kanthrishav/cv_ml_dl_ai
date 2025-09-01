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

To setup the SSD this link from Nvidia Jetson AI labs would suffice : https://www.jetson-ai-lab.com/tips_ssd-docker.html

### 1. Check firmware/bootloader/OS/JetPack versions and compare to latest

This prints what you have and (best-effort) scrapes NVIDIA pages to show the current latest so you can see if you’re up-to-date. Latest JetPack/L4T references: JetPack page and r36.4.4 release notes.

Run the shell script : check_firmware_os_bootloader.sh

### 2. Check snapd-AppArmor and install flatpak

On some Jetson kernels, AppArmor isn’t enabled ⇒ many snaps won’t run. These commands detect that and either enable what we can (user namespaces), or tell you to prefer Flatpak. (We’ll set up Flatpak in Set 4.)

Run the shell script : check_snapd_apparmor.sh

If your log says something like these:

 - 'aa-status' printed “apparmor not present.” → the AppArmor LSM isn’t enabled in the kernel.
 - snap debug sandbox-features didn’t show AppArmor enforcement, and snap run hello-world failed with cannot set capabilities: Operation not permitted → classic symptom when snaps try to sandbox without AppArmor support.
 - Enabling kernel.unprivileged_userns_clone=1 helps snaps in general, but doesn’t replace AppArmor.

If you still want to double confirm you can run the following 

  		# Is AppArmor built/loaded?
		grep -i apparmor /proc/cmdline
		cat /sys/module/apparmor/parameters/enabled  # likely: "No such file or directory"
		dmesg -T | grep -i apparmor || echo "No AppArmor messages in dmesg"

If AppArmor is not enabled in the kernel, you can go for flatpak based installation (it uninstalls any Chromium you might have installed during first bootup)

Run the shell script : install_flatpak_firefox_chromium.sh

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

Run the shell script : setup_vnc.sh

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

Run the shell script : check_docker_jetsonCont_tensorRT.sh

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
