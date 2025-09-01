sudo tee /etc/systemd/system/ollama-compose.service >/dev/null <<'UNIT'
[Unit]
Description=Ollama Compose Stack (SSD)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/ssd/ollama/compose
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now ollama-compose
systemctl status ollama-compose --no-pager -l
