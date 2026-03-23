#!/usr/bin/env python3
"""
Qwen3-TTS HTTP Server — Legacy Edition

Robust fallback server for Jetson:
- no CUDA graphs
- lower startup complexity
- slower, but safer when faster-large cannot start or run
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import io
import threading
import time
import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file
from qwen_tts import Qwen3TTSModel

from tts_config import MODEL_NAME, get_legacy_profile

app = Flask(__name__)
model = None
VALID_SPEAKERS = None
ATTN_IMPLEMENTATION = None
STARTUP_ERROR = None
STARTED_AT = time.time()
PROFILE = get_legacy_profile()
VERSION = "legacy-0.3.0"
APP_ENGINE = f"qwen-tts-legacy:{PROFILE.name}"
WARMUP_STATE = "pending"
LAST_WARMUP_AT = None


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
        "warmup_state": WARMUP_STATE,
        "last_warmup_at": LAST_WARMUP_AT,
        "last_request_at": LAST_REQUEST_AT,
        "last_success_at": LAST_SUCCESS_AT,
        "last_error": LAST_ERROR,
        "busy": INFER_LOCK.locked(),
    }


def vram_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def maybe_warmup(reason: str):
    global WARMUP_STATE, LAST_WARMUP_AT

    if model is None:
        return False

    mode = (PROFILE.warmup_mode or "minimal").lower()
    if mode == "none":
        WARMUP_STATE = "disabled"
        return False
    if WARMUP_STATE == "complete":
        return False

    print(
        f"Warmup ({reason}) with mode={mode}, max_new_tokens={PROFILE.warmup_max_new_tokens}..."
    )
    WARMUP_STATE = "running"
    start = time.time()
    try:
        model.generate_custom_voice(
            text=PROFILE.warmup_text,
            speaker=PROFILE.warmup_speaker,
            language=PROFILE.warmup_language,
            max_new_tokens=min(PROFILE.max_new_tokens, PROFILE.warmup_max_new_tokens),
        )
        elapsed = time.time() - start
        vram_cleanup()
        WARMUP_STATE = "complete"
        LAST_WARMUP_AT = round(time.time() - STARTED_AT, 1)
        ready = runtime_snapshot()
        print(
            f"Warmup done in {elapsed:.1f}s | GPU={ready['gpu_memory_gb']:.2f} GB reserved={ready['gpu_reserved_gb']:.2f} GB | MemAvailable={ready['mem_available_gb']:.2f} GB"
        )
        return True
    except Exception as exc:
        WARMUP_STATE = "failed"
        print(f"Warmup failed ({reason}): {exc}")
        vram_cleanup()
        return False



def load_model():
    global model, VALID_SPEAKERS, ATTN_IMPLEMENTATION, STARTUP_ERROR
    print(f"Loading Qwen3-TTS legacy model with profile={PROFILE.name}...")
    print(
        f"Profile config: max_new_tokens={PROFILE.max_new_tokens}, non_streaming_mode={PROFILE.non_streaming_mode}, "
        f"min_mem_available_gb={PROFILE.min_mem_available_gb}, warmup_mode={PROFILE.warmup_mode}, "
        f"warmup_max_new_tokens={PROFILE.warmup_max_new_tokens}"
    )

    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    VALID_SPEAKERS = set(model.model.get_supported_speakers())
    if hasattr(model.model, "config"):
        ATTN_IMPLEMENTATION = getattr(model.model.config, "_attn_implementation", "unknown")

    loaded = runtime_snapshot()
    print(
        f"Model loaded! GPU={loaded['gpu_memory_gb']:.2f} GB reserved={loaded['gpu_reserved_gb']:.2f} GB | MemAvailable={loaded['mem_available_gb']:.2f} GB"
    )
    print(f"Speakers: {', '.join(sorted(VALID_SPEAKERS))}")
    maybe_warmup("startup")
    ready = runtime_snapshot()
    print(
        f"Ready! GPU={ready['gpu_memory_gb']:.2f} GB reserved={ready['gpu_reserved_gb']:.2f} GB | MemAvailable={ready['mem_available_gb']:.2f} GB | warmup={ready['warmup_state']}"
    )
    STARTUP_ERROR = None


@app.route("/tts", methods=["POST"])
def tts():
    global LAST_REQUEST_AT, LAST_SUCCESS_AT, LAST_ERROR
    data = request.json or {}
    text = data.get("text", "Hallo.")
    speaker = data.get("speaker", PROFILE.warmup_speaker)
    language = data.get("language", PROFILE.warmup_language)
    requested_max_tokens = int(data.get("max_new_tokens", PROFILE.max_new_tokens))
    max_tokens = min(requested_max_tokens, PROFILE.max_new_tokens)

    if VALID_SPEAKERS is None or model is None:
        return jsonify({"error": "Model not loaded", "startup_error": STARTUP_ERROR, **runtime_snapshot()}), 503
    if speaker not in VALID_SPEAKERS:
        return jsonify({"error": f"Invalid speaker: {speaker}", "valid_speakers": sorted(VALID_SPEAKERS)}), 400
    if not text or not text.strip():
        return jsonify({"error": "Empty text"}), 400
    if INFER_LOCK.locked():
        return jsonify({"error": "TTS busy", "hint": "Retry shortly or route via queue.", **runtime_snapshot()}), 429

    with INFER_LOCK:
        LAST_REQUEST_AT = round(time.time() - STARTED_AT, 1)
        maybe_warmup("first_request")
        try:
            start = time.time()
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                non_streaming_mode=PROFILE.non_streaming_mode,
                max_new_tokens=max_tokens,
            )
            elapsed = time.time() - start

            buf = io.BytesIO()
            sf.write(buf, wavs[0], sr, format="WAV")
            buf.seek(0)

            duration = len(wavs[0]) / sr
            rtf = elapsed / duration if duration else 0.0
            LAST_SUCCESS_AT = round(time.time() - STARTED_AT, 1)
            LAST_ERROR = None
            snapshot = runtime_snapshot()
            print(
                f"TTS: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f}) | text={len(text)} chars | max_new_tokens={max_tokens} | GPU={snapshot['gpu_memory_gb']:.2f}GB reserved={snapshot['gpu_reserved_gb']:.2f}GB | MemAvailable={snapshot['mem_available_gb']:.2f}GB"
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
            response.headers["X-Warmup-State"] = snapshot["warmup_state"]
            return response
        except Exception as e:
            LAST_ERROR = str(e)
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
    snapshot = runtime_snapshot()
    return jsonify({
        **snapshot,
        "model": MODEL_NAME,
        "version": VERSION,
        "attention": ATTN_IMPLEMENTATION,
        "dtype": "bfloat16",
        "profile": PROFILE.to_dict(),
        "non_streaming_mode": PROFILE.non_streaming_mode,
        "warmup_state": WARMUP_STATE,
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
