#!/usr/bin/env python3
"""Shared configuration helpers for Jetson Qwen3-TTS servers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_PORT = 5050
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


@dataclass(frozen=True)
class ServerProfile:
    name: str
    engine: str
    description: str
    max_new_tokens: int
    max_seq_len: int | None = None
    non_streaming_mode: bool | None = None
    warmup_text: str = "Test."
    warmup_speaker: str = "sohee"
    warmup_language: str = "german"
    min_mem_available_gb: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


FASTER_PROFILES: Dict[str, ServerProfile] = {
    "large": ServerProfile(
        name="large",
        engine="faster-qwen3-tts",
        description="Langtext-first Profil für Jetson. Primärer Pfad für ~1k-3k Zeichen.",
        max_new_tokens=4096,
        max_seq_len=4096,
        min_mem_available_gb=3.0,
    ),
    "small": ServerProfile(
        name="small",
        engine="faster-qwen3-tts",
        description="Speichersparendes Faster-Profil für Debug/Notfälle, nicht Standard.",
        max_new_tokens=2048,
        max_seq_len=2048,
        min_mem_available_gb=2.0,
    ),
}

LEGACY_PROFILES: Dict[str, ServerProfile] = {
    "fallback": ServerProfile(
        name="fallback",
        engine="qwen-tts-legacy",
        description="Robustes Fallback-Profil ohne CUDA Graphs.",
        max_new_tokens=4096,
        non_streaming_mode=False,
        min_mem_available_gb=1.0,
    )
}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)



def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)



def get_faster_profile() -> ServerProfile:
    profile_name = os.environ.get("QWEN3_TTS_PROFILE", "large").strip().lower() or "large"
    profile = FASTER_PROFILES.get(profile_name)
    if profile is None:
        valid = ", ".join(sorted(FASTER_PROFILES))
        raise ValueError(f"Unknown faster profile '{profile_name}'. Valid: {valid}")
    return ServerProfile(
        **{
            **profile.to_dict(),
            "max_new_tokens": env_int("QWEN3_TTS_MAX_NEW_TOKENS", profile.max_new_tokens),
            "max_seq_len": env_int("QWEN3_TTS_MAX_SEQ_LEN", profile.max_seq_len or 2048),
            "min_mem_available_gb": env_float(
                "QWEN3_TTS_MIN_MEM_GB", profile.min_mem_available_gb
            ),
        }
    )



def get_legacy_profile() -> ServerProfile:
    profile = LEGACY_PROFILES["fallback"]
    return ServerProfile(
        **{
            **profile.to_dict(),
            "max_new_tokens": env_int("QWEN3_TTS_MAX_NEW_TOKENS", profile.max_new_tokens),
            "non_streaming_mode": os.environ.get("QWEN3_TTS_NON_STREAMING_MODE", str(profile.non_streaming_mode)).lower() == "true",
            "min_mem_available_gb": env_float(
                "QWEN3_TTS_MIN_MEM_GB", profile.min_mem_available_gb
            ),
        }
    )
