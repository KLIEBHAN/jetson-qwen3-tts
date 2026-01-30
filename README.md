# Qwen3-TTS Server

HTTP server for Qwen3-TTS running on Jetson Orin Nano.

## Features

- **Flash Attention 2** enabled for optimized inference
- Model stays loaded in GPU memory (~2 GB)
- Multiple speaker voices supported
- Simple REST API

## Performance

| Metric | Value |
|--------|-------|
| GPU Memory | ~2 GB |
| RTF (Real-Time Factor) | ~6.2x |
| Attention | flash_attention_2 |

*RTF 6.2x means 1 second of audio takes ~6.2 seconds to generate.*

## API Endpoints

### POST /tts
Generate speech from text.

```bash
curl -X POST http://localhost:5050/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo Welt", "speaker": "sohee", "language": "german"}' \
  -o output.wav
```

### GET /speakers
List available speakers.

### GET /health
Health check (returns GPU memory usage).

### GET /info
Server configuration info.

## Available Speakers

- aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian

## Service Management

```bash
sudo systemctl start qwen3-tts
sudo systemctl stop qwen3-tts
sudo systemctl restart qwen3-tts
sudo journalctl -u qwen3-tts -f
```

## Changelog

### 2026-01-30
- Added Flash Attention 2 support (`attn_implementation="flash_attention_2"`)
- Added `/info` endpoint for configuration details
- Added attention implementation logging on startup
- ~5% performance improvement with Flash Attention
