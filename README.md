# Jetson Qwen3-TTS

GPU-accelerated Text-to-Speech server for NVIDIA Jetson devices using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice).

Tested on **Jetson Orin Nano 8GB** with JetPack 6.x.

## Features

- 🚀 HTTP API for easy integration
- 🎭 Multiple voice speakers (serena, vivian, ryan, sohee, ...)
- 🔄 Model stays loaded in GPU memory
- 🐧 Systemd service for auto-start

## Requirements

- NVIDIA Jetson (Orin Nano, AGX Orin, etc.)
- JetPack 6.x with CUDA 12+
- Python 3.10+
- ~2 GB GPU memory

### PyTorch for Jetson

⚠️ **Standard PyTorch pip packages do NOT work on Jetson!**

You need a Jetson-specific build. Options:

1. **NVIDIA Pre-built Wheels** (recommended):
   ```bash
   # Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/
   # For JetPack 6.x / L4T R36.x:
   pip install torch-2.3.0+nv24.05-cp310-linux_aarch64.whl
   ```

2. **Build from Source** (if pre-built unavailable):
   ```bash
   # See: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
   # Build with: USE_CUDA=1 USE_CUDNN=1 python setup.py install
   ```

This project was tested with PyTorch 2.7.1a0 (built from source, CUDA 12.6).

## Installation

```bash
# Clone
git clone https://github.com/KLIEBHAN/jetson-qwen3-tts.git
cd jetson-qwen3-tts

# Virtual environment (recommended)
python3 -m venv venv --system-site-packages  # inherit system PyTorch
source venv/bin/activate

# Dependencies (PyTorch must be installed separately, see above)
pip install soundfile flask qwen-tts

# Create output directory
mkdir -p output

# Test (standalone, without server)
python test_tts.py "Hallo Welt"
```

## Usage

### Start Server

```bash
python tts_server.py
# Server runs on http://0.0.0.0:5050
```

### API Endpoints

#### POST /tts - Generate Speech

```bash
curl -X POST http://localhost:5050/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo Welt", "speaker": "serena", "language": "german"}' \
  -o output.wav
```

| Parameter | Type   | Default | Description              |
|-----------|--------|---------|--------------------------|
| text      | string | "Hallo." | Text to synthesize       |
| speaker   | string | serena  | Voice (see /speakers)    |
| language  | string | german  | Language                 |

#### GET /speakers - List Voices

```bash
curl http://localhost:5050/speakers
# {"speakers": ["serena", "vivian", "ryan", "sohee", ...]}
```

#### GET /health - Server Status

```bash
curl http://localhost:5050/health
# {"status": "ok", "gpu_memory_gb": 2.1}
```

### Client Script

```bash
./tts_client.sh "Text to speak" output.wav serena german
```

## Systemd Service (Auto-Start)

```bash
# Copy and edit service file
sudo cp qwen3-tts.service /etc/systemd/system/
sudo nano /etc/systemd/system/qwen3-tts.service  # adjust paths and user!

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable qwen3-tts
sudo systemctl start qwen3-tts

# Check status
sudo systemctl status qwen3-tts
sudo journalctl -u qwen3-tts -f
```

## Performance

| Metric                  | Jetson Orin Nano 8GB        |
|-------------------------|----------------------------|
| GPU Memory              | ~2 GB                      |
| RTF (Real-Time Factor)  | ~8x slower than realtime   |
| Startup                 | ~20s (including warmup)    |

**Note:** RTF ~8x means 10s of audio takes ~80s to generate. This is a hardware limitation of the Orin Nano.

## Speakers

Best results for German: **serena** (clear, natural) or **sohee** (good alternative)

All available: serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan

## Known Issues

- **Must use bfloat16**: float16 causes CUDA errors on Jetson
- **First request slow**: JIT compilation on first inference
- **Long texts**: May take several minutes for long passages
- **Flask dev server**: Not production-ready; consider gunicorn for production

## License

MIT
