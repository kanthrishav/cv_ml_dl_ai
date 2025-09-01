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
