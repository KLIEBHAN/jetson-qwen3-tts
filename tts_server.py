#!/usr/bin/env python3
"""
Qwen3-TTS HTTP Server — Standard/Legacy Edition
Fallback server using plain qwen-tts (no CUDA graphs).
Use tts_server_faster.py for 3.5x better performance.

Optimized for Jetson Orin Nano (8GB shared memory).
Uses SDPA attention (flash_attention_2 has kernel issues on Jetson sm_87).
Uses non_streaming_mode=False to reduce peak VRAM for long texts.
"""

import os
# Reduce CUDA memory fragmentation on shared-memory devices (Jetson)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import soundfile as sf
import io
import time
import gc
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, send_file, jsonify
from qwen_tts import Qwen3TTSModel

app = Flask(__name__)
model = None
VALID_SPEAKERS = None
ATTN_IMPLEMENTATION = None

# Generation defaults optimized for Jetson 8GB
MAX_NEW_TOKENS = 4096  # default 2048 limits audio length
NON_STREAMING_MODE = False  # simulates streaming text input → lower peak VRAM


def vram_cleanup():
    """Free GPU memory after inference."""
    gc.collect()
    torch.cuda.empty_cache()


def load_model():
    """Load Qwen3-TTS model with SDPA attention."""
    global model, VALID_SPEAKERS, ATTN_IMPLEMENTATION
    
    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    
    VALID_SPEAKERS = set(model.model.get_supported_speakers())
    
    # Log configuration
    if hasattr(model.model, "config"):
        ATTN_IMPLEMENTATION = getattr(model.model.config, "_attn_implementation", "unknown")
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded! GPU: {gpu_mem:.2f} GB")
    print(f"Attention: {ATTN_IMPLEMENTATION}")
    print(f"Speakers: {', '.join(sorted(VALID_SPEAKERS))}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"non_streaming_mode: {NON_STREAMING_MODE}")
    
    # Warmup (first inference is slow due to CUDA compilation)
    print("Warming up...")
    model.generate_custom_voice(text=".", speaker="serena", language="german")
    vram_cleanup()
    print("Ready!")


@app.route("/tts", methods=["POST"])
def tts():
    """Generate speech from text."""
    data = request.json or {}
    text = data.get("text", "Hallo.")
    speaker = data.get("speaker", "serena")
    language = data.get("language", "german")
    max_tokens = data.get("max_new_tokens", MAX_NEW_TOKENS)
    
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
            text=text,
            speaker=speaker,
            language=language,
            non_streaming_mode=NON_STREAMING_MODE,
            max_new_tokens=max_tokens,
        )
        elapsed = time.time() - start
        
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)
        
        duration = len(wavs[0]) / sr
        rtf = elapsed / duration
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"TTS: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f}) | text={len(text)} chars | VRAM={gpu_mem:.2f}GB")
        
        vram_cleanup()
        
        response = send_file(buf, mimetype="audio/wav")
        response.headers["X-Audio-Duration"] = f"{duration:.1f}"
        response.headers["X-Processing-Time"] = f"{elapsed:.1f}"
        response.headers["X-RTF"] = f"{rtf:.2f}"
        return response
        
    except Exception as e:
        vram_cleanup()
        print(f"TTS ERROR: {e} | text={len(text)} chars")
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
        "version": "0.1.1",
        "attention": ATTN_IMPLEMENTATION,
        "dtype": "bfloat16",
        "max_new_tokens": MAX_NEW_TOKENS,
        "non_streaming_mode": NON_STREAMING_MODE,
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "speakers": sorted(VALID_SPEAKERS)
    })


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port)
