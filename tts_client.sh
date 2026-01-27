#!/bin/bash
# Qwen3-TTS Client - talks to the local TTS server

set -e

show_help() {
    cat << EOF
Usage: $(basename "$0") "text" [output.wav] [speaker] [language]

Generate speech from text using local Qwen3-TTS server.

Arguments:
  text        Text to synthesize (required)
  output      Output file path (default: output/tts_output.wav)
  speaker     Voice to use (default: serena)
  language    Language (default: german)

Speakers:
  serena (default) - female, clear, natural
  sohee            - female, good alternative
  vivian, ryan, aiden, eric, dylan

Examples:
  $(basename "$0") "Hallo Welt"
  $(basename "$0") "Hello" out.wav serena english

Server: http://localhost:5050
EOF
}

[[ "$1" == "-h" || "$1" == "--help" ]] && { show_help; exit 0; }

TEXT="${1:-}"
OUTPUT="${2:-output/tts_output.wav}"
SPEAKER="${3:-serena}"
LANGUAGE="${4:-german}"
SERVER="http://localhost:5050"

[ -z "$TEXT" ] && { show_help; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"

curl -s -X POST "$SERVER/tts" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$TEXT\", \"speaker\": \"$SPEAKER\", \"language\": \"$LANGUAGE\"}" \
    -o "$OUTPUT"

if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$OUTPUT" 2>/dev/null || echo "?")
    SIZE=$(wc -c < "$OUTPUT" | tr -d ' ')
    echo "Output: $OUTPUT ($SIZE bytes, ${DURATION}s)"
else
    echo "Error: No output" >&2
    exit 1
fi
