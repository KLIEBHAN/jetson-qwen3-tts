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
import threading
import time
import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file

from tts_config import MODEL_NAME, get_faster_profile, get_required_mem_available_gb


PROFILE = get_faster_profile()
VERSION = "faster-0.4.0"
APP_ENGINE = f"faster-qwen3-tts:{PROFILE.name}"

app = Flask(__name__)
model = None
VALID_SPEAKERS = None
STARTUP_ERROR = None
STARTED_AT = time.time()
WARMUP_STATE = "pending"
LAST_WARMUP_AT = None
LAST_REQUEST_AT = None
LAST_SUCCESS_AT = None
LAST_ERROR = None
STARTUP_RELAXED = False
INFER_LOCK = threading.Lock()


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
        "startup_relaxed": STARTUP_RELAXED,
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



def preflight_can_run(text_chars: int | None = None, extra_headroom_gb: float = 0.0):
    mem = system_memory_snapshot()
    required = get_required_mem_available_gb(PROFILE, text_chars) + extra_headroom_gb
    required = round(required, 2)
    return mem["mem_available_gb"] >= required, mem, required



def warmup_required_mem_gb() -> float:
    base = get_required_mem_available_gb(PROFILE, text_chars=len(PROFILE.warmup_text))
    routing = getattr(PROFILE, "routing", None)
    if routing is not None:
        base = max(base, float(routing.medium_mem_gb))
    return round(base + PROFILE.startup_mem_headroom_gb, 2)



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

    mem = system_memory_snapshot()
    required = warmup_required_mem_gb()
    ok = mem["mem_available_gb"] >= required
    if not ok:
        WARMUP_STATE = "deferred"
        print(
            f"Warmup deferred ({reason}): MemAvailable={mem['mem_available_gb']:.2f} GB < {required:.2f} GB"
        )
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



def is_ready_state() -> bool:
    if model is None or STARTUP_ERROR is not None:
        return False
    mode = (PROFILE.warmup_mode or "minimal").lower()
    if mode == "none":
        return LAST_SUCCESS_AT is not None
    return WARMUP_STATE == "complete" or LAST_SUCCESS_AT is not None



def readiness_reason() -> str:
    if STARTUP_ERROR is not None:
        return "startup-error"
    if model is None:
        return "model-not-loaded"
    mode = (PROFILE.warmup_mode or "minimal").lower()
    if mode == "none" and LAST_SUCCESS_AT is None:
        return "awaiting-first-success"
    if WARMUP_STATE == "complete" or LAST_SUCCESS_AT is not None:
        return "ready"
    return f"warmup-{WARMUP_STATE}"



def ensure_ready(reason: str) -> bool:
    if is_ready_state():
        return True
    mode = (PROFILE.warmup_mode or "minimal").lower()
    if mode == "none" or model is None or STARTUP_ERROR is not None:
        return is_ready_state()
    if INFER_LOCK.locked():
        return False
    with INFER_LOCK:
        if is_ready_state():
            return True
        maybe_warmup(reason)
    return is_ready_state()



def load_model():
    global model, VALID_SPEAKERS, STARTUP_ERROR, STARTUP_RELAXED, WARMUP_STATE
    print(f"Loading FasterQwen3TTS model with profile={PROFILE.name}...")
    print(
        "Profile config: "
        f"max_seq_len={PROFILE.max_seq_len}, "
        f"max_new_tokens={PROFILE.max_new_tokens}, "
        f"min_mem_available_gb={PROFILE.min_mem_available_gb}, "
        f"warmup_mode={PROFILE.warmup_mode}, "
        f"warmup_max_new_tokens={PROFILE.warmup_max_new_tokens}, "
        f"startup_mem_headroom_gb={PROFILE.startup_mem_headroom_gb}, "
        f"startup_soft_gap_gb={PROFILE.startup_soft_gap_gb}"
    )

    ok, mem, required = preflight_can_run(extra_headroom_gb=PROFILE.startup_mem_headroom_gb)
    if not ok:
        deficit = round(required - mem['mem_available_gb'], 2)
        if deficit <= PROFILE.startup_soft_gap_gb:
            STARTUP_RELAXED = True
            WARMUP_STATE = "deferred"
            print(
                f"Relaxed startup enabled: MemAvailable={mem['mem_available_gb']:.2f} GB, "
                f"required={required:.2f} GB, deficit={deficit:.2f} GB <= soft_gap={PROFILE.startup_soft_gap_gb:.2f} GB"
            )
        else:
            raise RuntimeError(
                f"Insufficient MemAvailable for faster profile '{PROFILE.name}': {mem['mem_available_gb']:.2f} GB < {required:.2f} GB"
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

    if not STARTUP_RELAXED:
        maybe_warmup("startup")
    else:
        print("Startup warmup skipped because relaxed startup was used")
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
    text_chars = len(text or "")

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
        ok, mem, required = preflight_can_run(text_chars=text_chars)
        if not ok:
            LAST_ERROR = f"preflight:{mem['mem_available_gb']:.2f}<{required:.2f}"
            return jsonify({
                "error": "Insufficient MemAvailable for faster inference",
                "hint": "Route this request to the legacy fallback and retry faster later.",
                "required_mem_available_gb": required,
                "text_chars": text_chars,
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
            LAST_SUCCESS_AT = round(time.time() - STARTED_AT, 1)
            LAST_ERROR = None
            snapshot = runtime_snapshot()
            print(
                f"TTS: {duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f}) | text={text_chars} chars | max_new_tokens={max_tokens} | GPU={snapshot['gpu_memory_gb']:.2f}GB reserved={snapshot['gpu_reserved_gb']:.2f}GB | MemAvailable={snapshot['mem_available_gb']:.2f}GB"
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
            print(f"TTS ERROR: {e} | text={text_chars} chars")
            return jsonify({"error": str(e), **runtime_snapshot()}), 500


@app.route("/speakers", methods=["GET"])
def speakers():
    return jsonify({"speakers": sorted(VALID_SPEAKERS or [])})


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model is not None and STARTUP_ERROR is None else "degraded"
    return jsonify({
        "status": status,
        "ready": is_ready_state(),
        "ready_reason": readiness_reason(),
        **runtime_snapshot(),
    })


@app.route("/ready", methods=["GET"])
def ready():
    ready_now = ensure_ready("ready_probe")
    status = 200 if ready_now else 503
    return jsonify({
        "ready": ready_now,
        "ready_reason": readiness_reason(),
        **runtime_snapshot(),
    }), status


@app.route("/info", methods=["GET"])
def info():
    snapshot = runtime_snapshot()
    return jsonify({
        **snapshot,
        "model": MODEL_NAME,
        "engine": APP_ENGINE,
        "version": VERSION,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "cuda_graphs": True,
        "profile": PROFILE.to_dict(),
        "warmup_state": WARMUP_STATE,
        "ready": is_ready_state(),
        "ready_reason": readiness_reason(),
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
