#!/bin/bash
# Qwen3-TTS Client - talks to the local TTS server
# Usage: ./tts_client.sh "Text to speak" [output.wav] [speaker] [language]

TEXT="${1:-Hallo}"
OUTPUT="${2:-output/tts_output.wav}"
SPEAKER="${3:-serena}"
LANGUAGE="${4:-german}"
SERVER="http://localhost:5050"

echo "Text: $TEXT"
echo "Speaker: $SPEAKER"
echo "Language: $LANGUAGE"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

time curl -s -X POST "$SERVER/tts" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$TEXT\", \"speaker\": \"$SPEAKER\", \"language\": \"$LANGUAGE\"}" \
    -o "$OUTPUT"

if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$OUTPUT" 2>/dev/null || echo "?")
    SIZE=$(ls -lh "$OUTPUT" | awk "{print \$5}")
    echo "Output: $OUTPUT ($SIZE, ${DURATION}s)"
else
    echo "Error: No output or server not running"
    rm -f "$OUTPUT"
    exit 1
fi
