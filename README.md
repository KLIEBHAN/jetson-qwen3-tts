# Qwen3-TTS Server for Jetson

HTTP server for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) on NVIDIA Jetson Orin Nano.  
Uses [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) with CUDA Graph Capture for **3.5x faster inference**.

## Performance

| Metric | Value |
|--------|-------|
| GPU Memory | ~2.3 GB |
| RTF (Real-Time Factor) | **~1.7x** |
| Engine | faster-qwen3-tts + CUDA Graphs |
| Attention | SDPA |
| Dtype | bfloat16 |

*RTF 1.7x means 10 seconds of audio takes ~17 seconds to generate.*

## Quick Start

```bash
# Install dependencies (PyTorch for Jetson must be installed separately)
pip install -r requirements.txt

# Install as systemd service
sudo ./install-service.sh

# Or run directly
python3 tts_server_faster.py
```

## API

### POST /tts
Generate speech from text.

```bash
curl -X POST http://localhost:5050/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo Welt", "speaker": "sohee", "language": "german"}' \
  -o output.wav
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `speaker` | string | `"sohee"` | Voice (see below) |
| `language` | string | `"german"` | Language |
| `max_new_tokens` | int | `4096` | Max audio tokens |

**Response headers:** `X-Audio-Duration`, `X-Processing-Time`, `X-RTF`

### GET /speakers
List available voices.

### GET /health
Health check + GPU memory usage.

### GET /info
Full server configuration.

## Speakers

`aiden` · `dylan` · `eric` · `ono_anna` · `ryan` · `serena` · `sohee` · `uncle_fu` · `vivian`

## Languages

Chinese · English · German · French · Japanese · Korean · Russian · Portuguese · Spanish · Italian

## Architecture

```
tts_server_faster.py          ← HTTP API (Flask, port 5050)
  └── faster-qwen3-tts        ← CUDA Graph inference engine
       └── qwen-tts 0.1.1     ← Model loading & tokenization
            └── Qwen3-TTS-12Hz-0.6B-CustomVoice (HuggingFace)
```

## Service Management

```bash
sudo systemctl start qwen3-tts
sudo systemctl stop qwen3-tts
sudo systemctl restart qwen3-tts
sudo journalctl -u qwen3-tts -f
```

**Note:** On Jetson 8GB, stop Whisper/Ollama before starting TTS to free VRAM.

## Files

| File | Purpose |
|------|---------|
| `tts_server_faster.py` | HTTP server (faster engine, **recommended**) |
| `tts_server.py` | HTTP server (standard qwen-tts, fallback) |
| `install-service.sh` | Install as systemd service |
| `JETSON_NOTES.md` | Detailed optimization notes & benchmarks |

## Jetson-Specific Notes

See [JETSON_NOTES.md](JETSON_NOTES.md) for:
- Full benchmark results
- VRAM optimization details
- Troubleshooting guide
- Comparison: standard vs faster engine

## License

Apache-2.0
