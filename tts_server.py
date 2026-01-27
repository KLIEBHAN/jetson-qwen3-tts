#!/usr/bin/env python3
"""Qwen3-TTS HTTP Server - Model bleibt im Speicher"""

import torch
import soundfile as sf
import io
import time
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, send_file, jsonify
from qwen_tts import Qwen3TTSModel

app = Flask(__name__)
model = None

def load_model():
    global model
    print("Lade Qwen3-TTS Model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    print(f"Model geladen! GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    # Warmup
    model.generate_custom_voice(text=".", speaker="serena", language="german")
    print("Warmup done!")

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "Hallo.")
    speaker = data.get("speaker", "serena")
    language = data.get("language", "german")
    
    start = time.time()
    wavs, sr = model.generate_custom_voice(
        text=text, speaker=speaker, language=language
    )
    elapsed = time.time() - start
    
    # Audio als WAV zurückgeben
    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    buf.seek(0)
    
    duration = len(wavs[0]) / sr
    print(f"TTS: {duration:.1f}s Audio in {elapsed:.1f}s (RTF={elapsed/duration:.2f})")
    
    return send_file(buf, mimetype="audio/wav")

@app.route("/speakers", methods=["GET"])
def speakers():
    spks = list(model.model.get_supported_speakers())
    return jsonify({"speakers": spks})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gpu_memory_gb": torch.cuda.memory_allocated()/1024**3})

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5050)
