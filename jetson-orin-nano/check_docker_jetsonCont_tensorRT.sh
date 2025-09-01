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
