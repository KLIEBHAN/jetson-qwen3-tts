#!/usr/bin/env python3
"""Qwen3-TTS HTTP Server - keeps model loaded in GPU memory."""

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

VALID_SPEAKERS = None  # populated on load

def load_model():
    global model, VALID_SPEAKERS
    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    VALID_SPEAKERS = set(model.model.get_supported_speakers())
    print(f"Model loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Available speakers: {', '.join(sorted(VALID_SPEAKERS))}")
    # Warmup
    model.generate_custom_voice(text=".", speaker="serena", language="german")
    print("Warmup done!")

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json or {}
    text = data.get("text", "Hallo.")
    speaker = data.get("speaker", "serena")
    language = data.get("language", "german")
    
    # Validate speaker
    if speaker not in VALID_SPEAKERS:
        return jsonify({"error": f"Invalid speaker '{speaker}'", "valid": sorted(VALID_SPEAKERS)}), 400
    
    # Validate text
    if not text or not text.strip():
        return jsonify({"error": "Empty text"}), 400
    
    try:
        start = time.time()
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=speaker, language=language
        )
        elapsed = time.time() - start
        
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)
        
        duration = len(wavs[0]) / sr
        print(f"TTS: {duration:.1f}s audio in {elapsed:.1f}s (RTF={elapsed/duration:.2f})")
        
        return send_file(buf, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/speakers", methods=["GET"])
def speakers():
    return jsonify({"speakers": sorted(VALID_SPEAKERS)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gpu_memory_gb": round(torch.cuda.memory_allocated()/1024**3, 2)})

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5050)
