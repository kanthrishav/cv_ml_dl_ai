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
 - Storage : microSD card slot, NVMe M.2 2242/2280 SSD slot
 - Display :  1 x Dsiplay Port 1.2

Before proceeding towards the setup, I will provide a way to starting with this board activated for highest performance. I went down multiple routes mainly to explore. So, I will say this - for running any kind of LLM/VLM models you will need the following : 
1. Minimum 8 GB RAM variant (4 GB could not handle AI models in a way that was useful to me)
2. Minimum 128 GB microSD card for only OS and to act as your root volume
3. Minimum 1 TB NVMe SSD (for storing a plethora of AI model checkpoints, docker containers, Lora adapters, etc)

Based on my experience, if you miss either of these 3 requirements, you will know very very very soon that you need those now!

Apart from these I used a display monitor, a display port to HDMI cable (since my monitor takes HDMI input only), keyboard and mouse.



## Flashing the OS

