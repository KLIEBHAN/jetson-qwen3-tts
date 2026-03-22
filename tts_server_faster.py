#!/usr/bin/env python3
"""
Qwen3-TTS HTTP Server — Faster Edition

Langtext-first server for Jetson:
- faster-large profile is the default and primary path
- env-based profile parameters instead of duplicate files
- legacy engine remains the safety net outside this process
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import io
import time
import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file

from tts_config import MODEL_NAME, get_faster_profile


PROFILE = get_faster_profile()
VERSION = "faster-0.3.0"
APP_ENGINE = f"faster-qwen3-tts:{PROFILE.name}"

app = Flask(__name__)
model = None
VALID_SPEAKERS = None
STARTUP_ERROR = None
STARTED_AT = time.time()


def meminfo():
    info = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0])
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


def runtime_snapshot():
    mem = system_memory_snapshot()
    return {
        **mem,
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "profile": PROFILE.name,
        "engine": APP_ENGINE,
        "startup_error": STARTUP_ERROR,
        "uptime_s": round(time.time() - STARTED_AT, 1),
    }


def vram_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def preflight_can_run():
    mem = system_memory_snapshot()
    return mem["mem_available_gb"] >= PROFILE.min_mem_available_gb, mem



def load_model():
    global model, VALID_SPEAKERS, STARTUP_ERROR
    print(f"Loading FasterQwen3TTS model with profile={PROFILE.name}...")
    print(f"Profile config: max_seq_len={PROFILE.max_seq_len}, max_new_tokens={PROFILE.max_new_tokens}, min_mem_available_gb={PROFILE.min_mem_available_gb}")

    ok, mem = preflight_can_run()
    if not ok:
        raise RuntimeError(
            f"Insufficient MemAvailable for faster profile '{PROFILE.name}': {mem['mem_available_gb']:.2f} GB < {PROFILE.min_mem_available_gb:.2f} GB"
        )

    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_seq_len=PROFILE.max_seq_len,
    )
    VALID_SPEAKERS = set(model.model.model.get_supported_speakers())

    after_load = runtime_snapshot()
    print(
        f"Model loaded! GPU={after_load['gpu_memory_gb']:.2f} GB reserved={after_load['gpu_reserved_gb']:.2f} GB | MemAvailable={after_load['mem_available_gb']:.2f} GB"
    )
    print(f"Speakers: {', '.join(sorted(VALID_SPEAKERS))}")

    print("Warming up (captures CUDA graph on first inference)...")
    model.generate_custom_voice(
        text=PROFILE.warmup_text,
        speaker=PROFILE.warmup_speaker,
        language=PROFILE.warmup_language,
        max_new_tokens=min(PROFILE.max_new_tokens, 512),
    )
    vram_cleanup()
    ready = runtime_snapshot()
    print(
        f"Ready! GPU={ready['gpu_memory_gb']:.2f} GB reserved={ready['gpu_reserved_gb']:.2f} GB | MemAvailable={ready['mem_available_gb']:.2f} GB"
    )
    STARTUP_ERROR = None


@app.route("/tts", methods=["POST"])
def tts():
    data = request.json or {}
    text = data.get("text", "Hallo.")
    speaker = data.get("speaker", PROFILE.warmup_speaker)
    language = data.get("language", PROFILE.warmup_language)
    max_tokens = int(data.get("max_new_tokens", PROFILE.max_new_tokens))

    if VALID_SPEAKERS is None or model is None:
        return jsonify({"error": "Model not loaded", "startup_error": STARTUP_ERROR, **runtime_snapshot()}), 503
    if speaker not in VALID_SPEAKERS:
        return jsonify({"error": f"Invalid speaker: {speaker}", "valid_speakers": sorted(VALID_SPEAKERS)}), 400
    if not text or not text.strip():
        return jsonify({"error": "Empty text"}), 400

    ok, mem = preflight_can_run()
    if not ok:
        return jsonify({
            "error": "Insufficient MemAvailable for faster inference",
            "hint": "Route this request to the legacy fallback and retry faster later.",
            "required_mem_available_gb": PROFILE.min_mem_available_gb,
            **runtime_snapshot(),
        }), 503

    try:
        start = time.time()
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            max_new_tokens=max_tokens,
        )
        elapsed = time.time() - start

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)

        duration = len(wavs[0]) / sr
        rtf = elapsed / duration if duration else 0.0
        snapshot = runtime_snapshot()
        print(
            f"TTS: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f}) | text={len(text)} chars | GPU={snapshot['gpu_memory_gb']:.2f}GB reserved={snapshot['gpu_reserved_gb']:.2f}GB | MemAvailable={snapshot['mem_available_gb']:.2f}GB"
        )

        vram_cleanup()
        response = send_file(buf, mimetype="audio/wav")
        response.headers["X-Audio-Duration"] = f"{duration:.1f}"
        response.headers["X-Processing-Time"] = f"{elapsed:.1f}"
        response.headers["X-RTF"] = f"{rtf:.2f}"
        response.headers["X-Torch-Allocated-GB"] = f"{snapshot['gpu_memory_gb']:.2f}"
        response.headers["X-Torch-Reserved-GB"] = f"{snapshot['gpu_reserved_gb']:.2f}"
        response.headers["X-MemAvailable-GB"] = f"{snapshot['mem_available_gb']:.2f}"
        response.headers["X-Profile"] = PROFILE.name
        return response
    except Exception as e:
        vram_cleanup()
        print(f"TTS ERROR: {e} | text={len(text)} chars")
        return jsonify({"error": str(e), **runtime_snapshot()}), 500


@app.route("/speakers", methods=["GET"])
def speakers():
    return jsonify({"speakers": sorted(VALID_SPEAKERS or [])})


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model is not None and STARTUP_ERROR is None else "degraded"
    return jsonify({"status": status, **runtime_snapshot()})


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model": MODEL_NAME,
        "engine": APP_ENGINE,
        "version": VERSION,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "cuda_graphs": True,
        "profile": PROFILE.to_dict(),
        **runtime_snapshot(),
        "speakers": sorted(VALID_SPEAKERS or []),
    })


if __name__ == "__main__":
    try:
        load_model()
    except Exception as exc:
        STARTUP_ERROR = str(exc)
        print(f"STARTUP ERROR: {STARTUP_ERROR}")
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port)
