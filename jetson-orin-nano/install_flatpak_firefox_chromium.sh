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
