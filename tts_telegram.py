#!/usr/bin/env python3
"""Small Python orchestrator for Jetson Qwen3-TTS -> Telegram voice."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

DEFAULT_RESTORE_WAIT_SECONDS = int(os.environ.get("TTS_RESTORE_WAIT_SECONDS", "180"))
DEFAULT_RESTORE_POLL_SECONDS = int(os.environ.get("TTS_RESTORE_POLL_SECONDS", "10"))
DEFAULT_RESTORE_STABLE_SAMPLES = int(os.environ.get("TTS_RESTORE_STABLE_SAMPLES", "2"))
DEFAULT_RESTORE_HYSTERESIS_GB = float(os.environ.get("TTS_RESTORE_HYSTERESIS_GB", "0.2"))

DEFAULT_SERVER = os.environ.get("TTS_SERVER", "http://127.0.0.1:5050").rstrip("/")
DEFAULT_MAX_TIME = int(os.environ.get("TTS_MAX_TIME", "1800"))
DEFAULT_LEGACY_PORT = int(os.environ.get("TTS_LEGACY_PORT", "5052"))
DEFAULT_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
ROOT = Path(__file__).resolve().parent
LEGACY_SERVER_SCRIPT = ROOT / "tts_server.py"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"


class OrchestrationError(RuntimeError):
    pass


@dataclass
class ServiceSnapshot:
    base_url: str
    health: dict[str, Any]
    info: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.health.get("status") or "")

    @property
    def startup_error(self) -> str:
        return str(self.health.get("startup_error") or "")

    @property
    def engine(self) -> str:
        return str(self.info.get("engine") or self.health.get("engine") or "")

    @property
    def profile(self) -> Any:
        return self.info.get("profile") or self.health.get("profile") or {}

    @property
    def mem_available_gb(self) -> float:
        try:
            return float(self.health.get("mem_available_gb") or 0.0)
        except Exception:
            return 0.0


def log(message: str) -> None:
    print(f"[tts-telegram] {message}", file=sys.stderr, flush=True)


def read_bot_token(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("BOT_TOKEN", "").strip()
    if env:
        return env
    if OPENCLAW_CONFIG.exists():
        try:
            data = json.loads(OPENCLAW_CONFIG.read_text(encoding="utf-8"))
            token = (((data or {}).get("plugins") or {}).get("telegram") or {}).get("botToken")
            if token:
                return str(token)
            token = data.get("botToken")
            if token:
                return str(token)
        except Exception:
            pass
    raise OrchestrationError("BOT_TOKEN not set and not found in ~/.openclaw/openclaw.json")


def fetch_json(url: str, timeout: int = 10) -> dict[str, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def post_json_bytes(url: str, payload: dict[str, Any], timeout: int) -> tuple[bytes, dict[str, str], int]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            headers = {k: v for k, v in resp.headers.items()}
            return resp.read(), headers, resp.status
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise OrchestrationError(f"HTTP {exc.code} from {url}: {payload}") from exc
    except error.URLError as exc:
        raise OrchestrationError(f"Request to {url} failed: {exc}") from exc


def snapshot_service(base_url: str) -> ServiceSnapshot:
    return ServiceSnapshot(base_url=base_url, health=fetch_json(base_url + "/health"), info=fetch_json(base_url + "/info"))


def host_mem_available_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        return 0.0
    return 0.0


def profile_required_mem(profile: Any, text_chars: int, override_gb: float | None) -> float:
    if override_gb is not None:
        return override_gb
    if not isinstance(profile, dict):
        return 3.0
    routing = profile.get("routing") or {}
    base = float(profile.get("min_mem_available_gb", 3.0) or 3.0)
    if not isinstance(routing, dict) or not routing:
        return base
    short_chars = int(routing.get("short_chars", 400))
    medium_chars = int(routing.get("medium_chars", 1200))
    long_chars = int(routing.get("long_chars", 2200))
    if text_chars <= short_chars:
        required = float(routing.get("short_mem_gb", base))
    elif text_chars <= medium_chars:
        required = float(routing.get("medium_mem_gb", base))
    elif text_chars <= long_chars:
        required = float(routing.get("long_mem_gb", base))
    else:
        required = float(routing.get("xlong_mem_gb", base))
    return min(base, required)


class LegacyFallback:
    def __init__(self, port: int, tmpdir: Path) -> None:
        self.port = port
        self.tmpdir = tmpdir
        self.process: subprocess.Popen[str] | None = None
        self.log_path = tmpdir / f"tts_legacy_fallback_{os.getpid()}.log"
        self.url = f"http://127.0.0.1:{port}"

    def _port_pid(self) -> int | None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) != 0:
                return None
        result = subprocess.run(
            ["ss", "-ltnpH", f"sport = :{self.port}"],
            check=False,
            capture_output=True,
            text=True,
        )
        for token in result.stdout.split():
            if "pid=" in token:
                fragment = token.split("pid=", 1)[1]
                digits = "".join(ch for ch in fragment if ch.isdigit())
                if digits:
                    return int(digits)
        return None

    def ensure_port_free(self) -> None:
        pid = self._port_pid()
        if not pid:
            return
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8", errors="ignore").replace("\x00", " ")
        except Exception:
            cmdline = ""
        if "tts_server.py" not in cmdline:
            raise OrchestrationError(f"Legacy fallback port {self.port} already in use by unexpected process: {cmdline or f'pid={pid}'}")
        log(f"Cleaning up stale legacy fallback on port {self.port} (pid={pid})")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return
            time.sleep(1.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return

    def start(self, wait_seconds: int = 360) -> None:
        if self.process and self.process.poll() is None:
            return
        self.ensure_port_free()
        log(f"Starting temporary legacy fallback on port {self.port}")
        subprocess.run(["sudo", "systemctl", "stop", "qwen3-tts"], check=False)
        subprocess.run(["sync"], check=False)
        subprocess.run(["sudo", "tee", "/proc/sys/vm/drop_caches"], input="3\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        with self.log_path.open("w", encoding="utf-8") as log_file:
            self.process = subprocess.Popen(
                ["/usr/bin/python3", str(LEGACY_SERVER_SCRIPT)],
                cwd=str(ROOT),
                env={**os.environ, "QWEN3_TTS_PROFILE": "fallback", "PORT": str(self.port)},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        self.wait_ready(wait_seconds)

    def wait_ready(self, wait_seconds: int) -> None:
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="ignore")[-2000:] if self.log_path.exists() else ""
                raise OrchestrationError(f"Legacy fallback exited early with rc={self.process.returncode}:\n{tail}")
            snap = snapshot_service(self.url)
            if snap.status == "ok":
                log("Legacy fallback ready")
                return
            time.sleep(5)
        tail = self.log_path.read_text(encoding="utf-8", errors="ignore")[-2000:] if self.log_path.exists() else ""
        raise OrchestrationError(f"Legacy fallback not ready after {wait_seconds}s\n{tail}")

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.text_chars = len(args.text)
        self.bot_token = read_bot_token(args.bot_token)
        self.primary_url = args.server.rstrip("/")
        self.legacy = LegacyFallback(args.legacy_port, Path(args.tmpdir))
        self.restore_primary = False
        self.primary_profile: dict[str, Any] = {}
        self.primary_engine = ""
        self.tmpdir_obj: tempfile.TemporaryDirectory[str] | None = None
        self.temp_dir_path: Path | None = None
        self.wav_path: Path | None = None
        self.ogg_path: Path | None = None

    def _temp_paths(self) -> tuple[Path, Path]:
        if self.args.keep:
            self.temp_dir_path = Path(tempfile.mkdtemp(prefix="tts-telegram-", dir=self.args.tmpdir))
        else:
            self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="tts-telegram-", dir=self.args.tmpdir)
            self.temp_dir_path = Path(self.tmpdir_obj.name)
        base = self.temp_dir_path
        return base / "output.wav", base / "output.ogg"

    def restart_primary(self, aggressive: bool = False) -> None:
        if aggressive:
            log("Stopping Whisper/Ollama to free shared RAM")
            subprocess.run(["sudo", "systemctl", "stop", "whisper-server", "ollama"], check=False)
            subprocess.run(["sync"], check=False)
            subprocess.run(["sudo", "tee", "/proc/sys/vm/drop_caches"], input="3\n", text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        log("Restarting qwen3-tts")
        subprocess.run(["sudo", "systemctl", "restart", "qwen3-tts"], check=False)

    def wait_primary(self, wait_seconds: int = 360) -> ServiceSnapshot:
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            snap = snapshot_service(self.primary_url)
            if snap.status == "ok":
                return snap
            time.sleep(5)
        raise OrchestrationError(f"Primary TTS not ready after {wait_seconds}s")

    def choose_route(self) -> str:
        snap = snapshot_service(self.primary_url)
        self.primary_profile = snap.profile if isinstance(snap.profile, dict) else {}
        self.primary_engine = snap.engine
        startup_error = snap.startup_error.lower()
        if snap.status != "ok":
            current_mem = host_mem_available_gb()
            if "insufficient memavailable" in startup_error:
                log("Primary faster service reports clear memory shortage -> route directly to legacy")
                if not self.args.dry_run:
                    self.legacy.start()
                    self.restore_primary = True
                return self.legacy.url
            if current_mem < self.restore_target_mem_gb():
                log(
                    "Primary faster service unavailable and host RAM still below restore target "
                    f"({current_mem:.2f}GB < {self.restore_target_mem_gb():.2f}GB) -> route directly to legacy"
                )
                if not self.args.dry_run:
                    self.legacy.start()
                    self.restore_primary = True
                return self.legacy.url
            log("Primary service unhealthy -> restart once and retry")
            if self.args.dry_run:
                return self.primary_url
            self.restart_primary(aggressive=False)
            snap = self.wait_primary()

        log(
            "Primary service: "
            f"engine={snap.engine or 'unknown'}, mem_available={snap.mem_available_gb:.2f}GB, text_chars={self.text_chars}"
        )
        if snap.engine.startswith("faster-qwen3-tts:"):
            required = profile_required_mem(snap.profile, self.text_chars, self.args.longtext_mem_gb)
            if snap.mem_available_gb < required:
                log(f"MemAvailable {snap.mem_available_gb:.2f}GB < required {required:.2f}GB -> route to legacy fallback")
                if not self.args.dry_run:
                    self.legacy.start()
                    self.restore_primary = True
                return self.legacy.url
        return self.primary_url

    def synthesize(self, server_url: str) -> None:
        payload = {
            "text": self.args.text,
            "speaker": self.args.speaker,
            "language": self.args.language,
        }
        self.wav_path, self.ogg_path = self._temp_paths()
        log(f"Generating TTS via {server_url}")
        try:
            wav_bytes, headers, _ = post_json_bytes(server_url + "/tts", payload, timeout=self.args.max_time)
        except OrchestrationError:
            if server_url == self.primary_url:
                snap = snapshot_service(self.primary_url)
                if snap.engine.startswith("faster-qwen3-tts:"):
                    log("Primary faster request failed -> trying legacy fallback once")
                    self.legacy.start()
                    self.restore_primary = True
                    wav_bytes, headers, _ = post_json_bytes(self.legacy.url + "/tts", payload, timeout=self.args.max_time)
                else:
                    raise
            else:
                raise
        self.wav_path.write_bytes(wav_bytes)
        log(f"WAV generated: {self.wav_path} ({self.wav_path.stat().st_size} bytes, duration={headers.get('X-Audio-Duration', '?')}s)")

    def convert_to_ogg(self) -> None:
        assert self.wav_path and self.ogg_path
        log("Converting WAV -> OGG/Opus")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(self.wav_path), "-c:a", "libopus", "-b:a", "64k", str(self.ogg_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        log(f"OGG generated: {self.ogg_path} ({self.ogg_path.stat().st_size} bytes)")

    def send_voice(self) -> int:
        assert self.ogg_path
        log(f"Sending voice to Telegram (chat_id={self.args.chat_id})")
        form = [
            ("chat_id", str(self.args.chat_id)),
        ]
        if self.args.reply_to:
            form.append(("reply_to_message_id", str(self.args.reply_to)))
        if self.args.caption:
            form.append(("caption", self.args.caption))
        boundary = f"----OpenClawBoundary{int(time.time() * 1000)}"
        body = bytearray()
        for key, value in form:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="voice"; filename="voice.ogg"\r\n')
        body.extend(b"Content-Type: audio/ogg\r\n\r\n")
        body.extend(self.ogg_path.read_bytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        req = request.Request(
            f"https://api.telegram.org/bot{self.bot_token}/sendVoice",
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("ok"):
            raise OrchestrationError(f"Telegram send failed: {payload.get('description', 'unknown error')}")
        result = payload["result"]
        return int(result["message_id"])

    def restore_target_mem_gb(self) -> float:
        profile = self.primary_profile if isinstance(self.primary_profile, dict) else {}
        base = float(profile.get("min_mem_available_gb", 3.0) or 3.0)
        headroom = float(profile.get("startup_mem_headroom_gb", 0.6) or 0.6)
        return base + headroom + float(self.args.restore_hysteresis_gb)

    def wait_for_restore_window(self) -> bool:
        target = self.restore_target_mem_gb()
        deadline = time.time() + self.args.restore_wait_seconds
        stable = 0
        while time.time() < deadline:
            mem = host_mem_available_gb()
            if mem >= target:
                stable += 1
                log(
                    f"Restore precheck sample {stable}/{self.args.restore_stable_samples}: "
                    f"MemAvailable={mem:.2f}GB >= target {target:.2f}GB"
                )
                if stable >= self.args.restore_stable_samples:
                    return True
            else:
                if stable:
                    log(
                        f"Restore precheck reset: MemAvailable={mem:.2f}GB < target {target:.2f}GB"
                    )
                stable = 0
            time.sleep(self.args.restore_poll_seconds)
        mem = host_mem_available_gb()
        log(
            f"Skip faster restore for now: MemAvailable={mem:.2f}GB < target {target:.2f}GB "
            f"after waiting {self.args.restore_wait_seconds}s"
        )
        return False

    def restore(self) -> None:
        self.legacy.stop()
        if self.restore_primary:
            if not self.wait_for_restore_window():
                return
            log("Restoring faster primary service")
            subprocess.run(["sudo", "systemctl", "start", "qwen3-tts"], check=False)
            try:
                self.wait_primary(wait_seconds=self.args.restore_wait_seconds)
            except Exception as exc:
                log(f"Primary restore stayed degraded: {exc}")

    def cleanup(self) -> None:
        if self.args.keep:
            if self.temp_dir_path:
                log(f"Keeping temp dir: {self.temp_dir_path}")
            self.tmpdir_obj = None
            return
        if self.tmpdir_obj:
            self.tmpdir_obj.cleanup()
            self.tmpdir_obj = None

    def run(self) -> int:
        route = self.choose_route()
        if self.args.dry_run:
            log(f"Dry-run: selected route {route}")
            return 0
        try:
            self.synthesize(route)
            self.convert_to_ogg()
            message_id = self.send_voice()
            print(message_id)
            return 0
        finally:
            self.restore()
            self.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS -> OGG -> Telegram voice orchestrator for Jetson")
    parser.add_argument("text")
    parser.add_argument("chat_id")
    parser.add_argument("--reply-to")
    parser.add_argument("--caption")
    parser.add_argument("--speaker", default="sohee")
    parser.add_argument("--language", default="german")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument("--legacy-port", type=int, default=DEFAULT_LEGACY_PORT)
    parser.add_argument("--tmpdir", default=str(DEFAULT_TMPDIR))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve routing and planned actions, but do not synthesize or send")
    parser.add_argument("--bot-token")
    parser.add_argument("--longtext-mem-gb", type=float, default=(float(os.environ["TTS_LONGTEXT_MEM_GB"]) if os.environ.get("TTS_LONGTEXT_MEM_GB") else None))
    parser.add_argument("--restore-wait-seconds", type=int, default=DEFAULT_RESTORE_WAIT_SECONDS)
    parser.add_argument("--restore-poll-seconds", type=int, default=DEFAULT_RESTORE_POLL_SECONDS)
    parser.add_argument("--restore-stable-samples", type=int, default=DEFAULT_RESTORE_STABLE_SAMPLES)
    parser.add_argument("--restore-hysteresis-gb", type=float, default=DEFAULT_RESTORE_HYSTERESIS_GB)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    orch = Orchestrator(args)
    try:
        return orch.run()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed: {exc}")
    except OrchestrationError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
