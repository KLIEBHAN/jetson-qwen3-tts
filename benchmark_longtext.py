#!/usr/bin/env python3
"""Run long-text benchmarks against the local TTS HTTP API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = os.environ.get("TTS_BENCH_SERVER", "http://127.0.0.1:5050")
SPEAKER = os.environ.get("TTS_BENCH_SPEAKER", "sohee")
LANGUAGE = os.environ.get("TTS_BENCH_LANGUAGE", "german")
TIMEOUT = int(os.environ.get("TTS_BENCH_TIMEOUT", "2400"))
OUTFILE = Path(os.environ.get("TTS_BENCH_OUT", "benchmark_results_longtext.json"))

BASE_TEXT = (
    "Dies ist ein realistischer deutscher Langtext für einen Jetson-TTS-Benchmark. "
    "Er enthält vollständige Sätze, Nebensätze, Zahlen wie 2026 und unterschiedliche Satzlängen. "
    "Ziel ist nicht maximale Token-Dichte, sondern ein praxisnaher Fließtext, wie er in Telegram- oder Vorlese-Szenarien vorkommt. "
    "Die Engine soll den gesamten Text in einem Stück verarbeiten, ohne Chunking als Standardweg. "
)
TARGETS = [1000, 1500, 2000, 2500, 3000]


def build_text(target_chars: int) -> str:
    text = []
    while len("".join(text)) < target_chars:
        text.append(BASE_TEXT)
    joined = "".join(text)
    return joined[:target_chars]



def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))



def post_tts(text: str) -> tuple[dict, bytes, float]:
    body = json.dumps({"text": text, "speaker": SPEAKER, "language": LANGUAGE}).encode("utf-8")
    req = urllib.request.Request(
        SERVER + "/tts",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = resp.read()
        elapsed = time.time() - start
        headers = dict(resp.headers.items())
    return headers, payload, elapsed



def main() -> int:
    results = []
    for chars in TARGETS:
        text = build_text(chars)
        health_before = fetch_json(SERVER + "/health")
        row = {
            "chars": len(text),
            "engine": health_before.get("engine"),
            "profile": health_before.get("profile"),
            "mem_available_before_gb": health_before.get("mem_available_gb"),
            "gpu_reserved_before_gb": health_before.get("gpu_reserved_gb"),
            "status": "unknown",
        }
        try:
            headers, payload, wall = post_tts(text)
            health_after = fetch_json(SERVER + "/health")
            row.update(
                {
                    "status": "ok",
                    "wav_bytes": len(payload),
                    "audio_duration_s": float(headers.get("X-Audio-Duration", 0.0)),
                    "processing_time_s": float(headers.get("X-Processing-Time", wall)),
                    "wall_time_s": round(wall, 2),
                    "rtf": float(headers.get("X-RTF", 0.0)),
                    "mem_available_after_gb": health_after.get("mem_available_gb"),
                    "gpu_reserved_after_gb": health_after.get("gpu_reserved_gb"),
                }
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            row.update({"status": f"http_{exc.code}", "error": body})
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)})
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))
    OUTFILE.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {OUTFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
