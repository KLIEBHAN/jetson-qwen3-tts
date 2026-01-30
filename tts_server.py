#!/usr/bin/env python3
"""
Qwen3-TTS HTTP Server
Keeps model loaded in GPU memory for fast inference.
Supports Flash Attention 2 for optimized performance.
"""

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
VALID_SPEAKERS = None
ATTN_IMPLEMENTATION = None


def load_model():
    """Load Qwen3-TTS model with Flash Attention 2."""
    global model, VALID_SPEAKERS, ATTN_IMPLEMENTATION
    
    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    
    VALID_SPEAKERS = set(model.model.get_supported_speakers())
    
    # Log configuration
    if hasattr(model.model, "config"):
        ATTN_IMPLEMENTATION = getattr(model.model.config, "_attn_implementation", "unknown")
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded! GPU: {gpu_mem:.2f} GB")
    print(f"Attention: {ATTN_IMPLEMENTATION}")
    print(f"Speakers: {', '.join(sorted(VALID_SPEAKERS))}")
    
    # Warmup (first inference is slow due to CUDA compilation)
    print("Warming up...")
    model.generate_custom_voice(text=".", speaker="serena", language="german")
    print("Ready!")


@app.route("/tts", methods=["POST"])
def tts():
    """Generate speech from text."""
    data = request.json or {}
    text = data.get("text", "Hallo.")
    speaker = data.get("speaker", "serena")
    language = data.get("language", "german")
    
    if speaker not in VALID_SPEAKERS:
        return jsonify({
            "error": f"Invalid speaker: {speaker}",
            "valid_speakers": sorted(VALID_SPEAKERS)
        }), 400
    
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
        rtf = elapsed / duration
        print(f"TTS: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})")
        
        return send_file(buf, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speakers", methods=["GET"])
def speakers():
    """List available speakers."""
    return jsonify({"speakers": sorted(VALID_SPEAKERS)})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)
    })


@app.route("/info", methods=["GET"])
def info():
    """Return server configuration info."""
    return jsonify({
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "attention": ATTN_IMPLEMENTATION,
        "dtype": "bfloat16",
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "speakers": sorted(VALID_SPEAKERS)
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5050)
