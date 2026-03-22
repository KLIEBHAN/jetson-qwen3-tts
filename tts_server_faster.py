#!/usr/bin/env python3
"""
Qwen3-TTS HTTP Server — Faster Edition
Uses faster-qwen3-tts with CUDA Graph Capture for ~3.5x speedup.
Optimized for Jetson Orin Nano (8GB shared memory).

Requires: Whisper/Ollama to be stopped before start (shared VRAM).
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import soundfile as sf
import io
import time
import gc
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, send_file, jsonify


def meminfo():
    info = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0])  # kB
    except Exception:
        pass
    return info


def system_memory_snapshot():
    info = meminfo()
    return {
        "mem_total_gb": round(info.get("MemTotal", 0) / 1024 / 1024, 2),
        "mem_available_gb": round(info.get("MemAvailable", 0) / 1024 / 1024, 2),
        "mem_free_gb": round(info.get("MemFree", 0) / 1024 / 1024, 2),
        "swap_free_gb": round(info.get("SwapFree", 0) / 1024 / 1024, 2),
    }

app = Flask(__name__)
model = None
VALID_SPEAKERS = None
VERSION = "faster-0.2.4"

# Generation defaults
MAX_NEW_TOKENS = 4096


def vram_cleanup():
    """Free GPU memory after inference."""
    gc.collect()
    torch.cuda.empty_cache()


def load_model():
    """Load Qwen3-TTS with CUDA graph acceleration."""
    global model, VALID_SPEAKERS

    print("Loading FasterQwen3TTS model...")
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    VALID_SPEAKERS = set(model.model.model.get_supported_speakers())

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    mem = system_memory_snapshot()
    print(f"Model loaded! GPU: {gpu_mem:.2f} GB | MemAvailable: {mem['mem_available_gb']:.2f} GB")
    print(f"Speakers: {', '.join(sorted(VALID_SPEAKERS))}")

    # Warmup triggers CUDA graph capture
    print("Warming up (CUDA graph capture)...")
    model.generate_custom_voice(text="Test.", speaker="sohee", language="german")
    vram_cleanup()

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Ready! VRAM after warmup: {gpu_mem:.2f} GB")


@app.route("/tts", methods=["POST"])
def tts():
    """Generate speech from text."""
    data = request.json or {}
    text = data.get("text", "Hallo.")
    speaker = data.get("speaker", "sohee")
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
            text=text,
            speaker=speaker,
            language=language,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        elapsed = time.time() - start

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)

        duration = len(wavs[0]) / sr
        rtf = elapsed / duration
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        mem = system_memory_snapshot()
        print(
            f"TTS: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f}) | "
            f"text={len(text)} chars | VRAM={gpu_mem:.2f}GB reserved={gpu_reserved:.2f}GB | "
            f"MemAvailable={mem['mem_available_gb']:.2f}GB"
        )

        vram_cleanup()

        response = send_file(buf, mimetype="audio/wav")
        response.headers["X-Audio-Duration"] = f"{duration:.1f}"
        response.headers["X-Processing-Time"] = f"{elapsed:.1f}"
        response.headers["X-RTF"] = f"{rtf:.2f}"
        response.headers["X-Torch-Allocated-GB"] = f"{gpu_mem:.2f}"
        response.headers["X-Torch-Reserved-GB"] = f"{gpu_reserved:.2f}"
        response.headers["X-MemAvailable-GB"] = f"{mem['mem_available_gb']:.2f}"
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
    mem = system_memory_snapshot()
    return jsonify({
        "status": "ok",
        "engine": "faster-qwen3-tts",
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        **mem,
    })


@app.route("/info", methods=["GET"])
def info():
    """Return server configuration info."""
    mem = system_memory_snapshot()
    return jsonify({
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "engine": "faster-qwen3-tts",
        "version": VERSION,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "cuda_graphs": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        **mem,
        "speakers": sorted(VALID_SPEAKERS)
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5050)
