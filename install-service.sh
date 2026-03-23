#!/bin/bash
# Install Qwen3-TTS Server as systemd service
# Usage:
#   sudo ./install-service.sh                              # faster-large (default)
#   sudo ./install-service.sh --legacy                     # legacy fallback engine
#   sudo ./install-service.sh --profile small             # faster-small
#   sudo ./install-service.sh --service-name qwen3-tts-small --profile small --port 5053

set -euo pipefail

USER_NAME="$(logname 2>/dev/null || echo ${SUDO_USER:-$USER})"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SERVICE_NAME="qwen3-tts"
SERVER="tts_server_faster.py"
PROFILE="large"
PORT=""
MAX_SEQ_LEN=""
MAX_NEW_TOKENS=""
MIN_MEM_GB=""
WARMUP_MODE=""
WARMUP_MAX_NEW_TOKENS=""
STARTUP_HEADROOM_GB=""
STARTUP_SOFT_GAP_GB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --legacy)
            SERVER="tts_server.py"
            PROFILE="fallback"
            shift
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max-seq-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --min-mem-gb)
            MIN_MEM_GB="$2"
            shift 2
            ;;
        --warmup-mode)
            WARMUP_MODE="$2"
            shift 2
            ;;
        --warmup-max-new-tokens)
            WARMUP_MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --startup-headroom-gb)
            STARTUP_HEADROOM_GB="$2"
            shift 2
            ;;
        --startup-soft-gap-gb)
            STARTUP_SOFT_GAP_GB="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: sudo ./install-service.sh [options]

Options:
  --legacy                        Install legacy fallback engine
  --profile <name>                Faster profile (default: large) or legacy profile fallback
  --service-name <name>           systemd unit name (default: qwen3-tts)
  --port <n>                      HTTP port override (default from server, usually 5050)
  --max-seq-len <n>               Override faster static cache length
  --max-new-tokens <n>            Override generation cap
  --min-mem-gb <gb>               Override minimum MemAvailable preflight threshold
  --warmup-mode <none|minimal>    Control startup warmup / graph capture
  --warmup-max-new-tokens <n>     Cap warmup generation length
  --startup-headroom-gb <gb>      Extra MemAvailable required before startup warmup path
  --startup-soft-gap-gb <gb>      Soft startup gap before hard failure (faster only)
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo $0"
    exit 1
fi

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "Installing Qwen3-TTS Server service..."
echo "  User:    $USER_NAME"
echo "  Path:    $SCRIPT_DIR"
echo "  Server:  $SERVER"
echo "  Profile: $PROFILE"
echo "  Service: $SERVICE_NAME"
if [ -n "$PORT" ]; then
    echo "  Port:    $PORT"
fi
echo

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Qwen3-TTS Server ($SERVICE_NAME)
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/$SERVER
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=QWEN3_TTS_PROFILE=$PROFILE
EOF

if [ -n "$PORT" ]; then
    echo "Environment=PORT=$PORT" >> "$SERVICE_FILE"
fi
if [ -n "$MAX_SEQ_LEN" ]; then
    echo "Environment=QWEN3_TTS_MAX_SEQ_LEN=$MAX_SEQ_LEN" >> "$SERVICE_FILE"
fi
if [ -n "$MAX_NEW_TOKENS" ]; then
    echo "Environment=QWEN3_TTS_MAX_NEW_TOKENS=$MAX_NEW_TOKENS" >> "$SERVICE_FILE"
fi
if [ -n "$MIN_MEM_GB" ]; then
    echo "Environment=QWEN3_TTS_MIN_MEM_GB=$MIN_MEM_GB" >> "$SERVICE_FILE"
fi
if [ -n "$WARMUP_MODE" ]; then
    echo "Environment=QWEN3_TTS_WARMUP_MODE=$WARMUP_MODE" >> "$SERVICE_FILE"
fi
if [ -n "$WARMUP_MAX_NEW_TOKENS" ]; then
    echo "Environment=QWEN3_TTS_WARMUP_MAX_NEW_TOKENS=$WARMUP_MAX_NEW_TOKENS" >> "$SERVICE_FILE"
fi
if [ -n "$STARTUP_HEADROOM_GB" ]; then
    echo "Environment=QWEN3_TTS_STARTUP_HEADROOM_GB=$STARTUP_HEADROOM_GB" >> "$SERVICE_FILE"
fi
if [ -n "$STARTUP_SOFT_GAP_GB" ]; then
    echo "Environment=QWEN3_TTS_STARTUP_SOFT_GAP_GB=$STARTUP_SOFT_GAP_GB" >> "$SERVICE_FILE"
fi

cat >> "$SERVICE_FILE" <<'EOF'

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo
echo "✅ Qwen3-TTS Server installed and started!"
echo "   Engine:  $SERVER"
echo "   Profile: $PROFILE"
echo "   Service: $SERVICE_NAME"
echo
echo "Commands:"
echo "  sudo systemctl status $SERVICE_NAME"
echo "  sudo systemctl restart $SERVICE_NAME"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo
if [ -n "$PORT" ]; then
    echo "API: http://localhost:$PORT"
else
    echo "API: http://localhost:5050"
fi
