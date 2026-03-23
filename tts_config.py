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
class RoutingThresholds:
    short_chars: int = 400
    medium_chars: int = 1200
    long_chars: int = 2200
    short_mem_gb: float = 1.6
    medium_mem_gb: float = 2.2
    long_mem_gb: float = 2.8
    xlong_mem_gb: float = 3.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


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
    warmup_mode: str = "minimal"
    warmup_max_new_tokens: int = 256
    startup_mem_headroom_gb: float = 0.5
    startup_soft_gap_gb: float = 0.0
    routing: RoutingThresholds | None = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        if self.routing is not None:
            data["routing"] = self.routing.to_dict()
        return data


FASTER_PROFILES: Dict[str, ServerProfile] = {
    "large": ServerProfile(
        name="large",
        engine="faster-qwen3-tts",
        description="Langtext-first Profil für Jetson. Primärer Pfad für ~1k-3k Zeichen.",
        max_new_tokens=4096,
        max_seq_len=3584,
        min_mem_available_gb=3.0,
        warmup_mode="minimal",
        warmup_max_new_tokens=128,
        startup_mem_headroom_gb=0.4,
        startup_soft_gap_gb=0.2,
        routing=RoutingThresholds(
            short_chars=400,
            medium_chars=1200,
            long_chars=2200,
            short_mem_gb=2.6,
            medium_mem_gb=2.8,
            long_mem_gb=3.0,
            xlong_mem_gb=3.0,
        ),
    ),
    "small": ServerProfile(
        name="small",
        engine="faster-qwen3-tts",
        description="Konservativer Faster-Primärpfad für Jetson-Bootstabilität.",
        max_new_tokens=1536,
        max_seq_len=1536,
        min_mem_available_gb=1.8,
        warmup_mode="none",
        warmup_max_new_tokens=64,
        startup_mem_headroom_gb=0.0,
        startup_soft_gap_gb=0.3,
        routing=RoutingThresholds(
            short_chars=350,
            medium_chars=800,
            long_chars=1400,
            short_mem_gb=1.0,
            medium_mem_gb=1.3,
            long_mem_gb=1.8,
            xlong_mem_gb=2.0,
        ),
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
        warmup_mode="minimal",
        warmup_max_new_tokens=64,
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



def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip()



def build_routing_thresholds(defaults: RoutingThresholds | None) -> RoutingThresholds | None:
    if defaults is None:
        return None
    return RoutingThresholds(
        short_chars=env_int("QWEN3_TTS_ROUTE_SHORT_CHARS", defaults.short_chars),
        medium_chars=env_int("QWEN3_TTS_ROUTE_MEDIUM_CHARS", defaults.medium_chars),
        long_chars=env_int("QWEN3_TTS_ROUTE_LONG_CHARS", defaults.long_chars),
        short_mem_gb=env_float("QWEN3_TTS_ROUTE_SHORT_MEM_GB", defaults.short_mem_gb),
        medium_mem_gb=env_float("QWEN3_TTS_ROUTE_MEDIUM_MEM_GB", defaults.medium_mem_gb),
        long_mem_gb=env_float("QWEN3_TTS_ROUTE_LONG_MEM_GB", defaults.long_mem_gb),
        xlong_mem_gb=env_float("QWEN3_TTS_ROUTE_XLONG_MEM_GB", defaults.xlong_mem_gb),
    )



def get_required_mem_available_gb(profile: ServerProfile, text_chars: int | None = None) -> float:
    if text_chars is None or profile.routing is None:
        return profile.min_mem_available_gb

    routing = profile.routing
    if text_chars <= routing.short_chars:
        required = routing.short_mem_gb
    elif text_chars <= routing.medium_chars:
        required = routing.medium_mem_gb
    elif text_chars <= routing.long_chars:
        required = routing.long_mem_gb
    else:
        required = routing.xlong_mem_gb

    return min(profile.min_mem_available_gb, required)



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
            "warmup_mode": env_str("QWEN3_TTS_WARMUP_MODE", profile.warmup_mode).lower(),
            "warmup_max_new_tokens": env_int(
                "QWEN3_TTS_WARMUP_MAX_NEW_TOKENS", profile.warmup_max_new_tokens
            ),
            "startup_mem_headroom_gb": env_float(
                "QWEN3_TTS_STARTUP_HEADROOM_GB", profile.startup_mem_headroom_gb
            ),
            "startup_soft_gap_gb": env_float(
                "QWEN3_TTS_STARTUP_SOFT_GAP_GB", profile.startup_soft_gap_gb
            ),
            "routing": build_routing_thresholds(profile.routing),
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
            "warmup_mode": env_str("QWEN3_TTS_WARMUP_MODE", profile.warmup_mode).lower(),
            "warmup_max_new_tokens": env_int(
                "QWEN3_TTS_WARMUP_MAX_NEW_TOKENS", profile.warmup_max_new_tokens
            ),
        }
    )
