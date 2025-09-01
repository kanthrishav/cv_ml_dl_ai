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
