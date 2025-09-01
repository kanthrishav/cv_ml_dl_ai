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
