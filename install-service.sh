#!/bin/bash
# Install Qwen3-TTS Server as systemd service
# Usage: sudo ./install-service.sh

set -e

USER="$(logname 2>/dev/null || echo $SUDO_USER)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="/etc/systemd/system/qwen3-tts.service"

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo $0"
    exit 1
fi

echo "Installing Qwen3-TTS Server service..."
echo "  User: $USER"
echo "  Path: $SCRIPT_DIR"
echo

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Qwen3-TTS Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/tts_server.py
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

echo "Service file created: $SERVICE_FILE"

systemctl daemon-reload
systemctl enable qwen3-tts
systemctl start qwen3-tts

echo
echo "✅ Qwen3-TTS Server installed and started!"
echo
echo "Commands:"
echo "  sudo systemctl status qwen3-tts"
echo "  sudo systemctl restart qwen3-tts"
echo "  sudo journalctl -u qwen3-tts -f"
echo
echo "API: http://localhost:5050"
