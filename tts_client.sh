#!/bin/bash
# Qwen3-TTS Client - spricht mit dem lokalen TTS Server
# Usage: ./tts_client.sh "Text zum Sprechen" [output.wav] [speaker]

TEXT="${1:-Hallo}"
OUTPUT="${2:-output/tts_output.wav}"
SPEAKER="${3:-serena}"
SERVER="http://localhost:5050"

echo "Text: $TEXT"
echo "Sprecher: $SPEAKER"

time curl -s -X POST "$SERVER/tts" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$TEXT\", \"speaker\": \"$SPEAKER\", \"language\": \"german\"}" \
    -o "$OUTPUT"

if [ -f "$OUTPUT" ]; then
    DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$OUTPUT" 2>/dev/null || echo "?")
    SIZE=$(ls -lh "$OUTPUT" | awk "{print \$5}")
    echo "Output: $OUTPUT ($SIZE, ${DURATION}s)"
else
    echo "Fehler: Keine Ausgabe"
fi
