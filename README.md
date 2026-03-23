# Qwen3-TTS Server for Jetson

Langtext-orientierter HTTP-Server für [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) auf NVIDIA Jetson Orin Nano.

## Zielbild

- **Stabiler Produktionspfad vor Maximalleistung**: auf diesem Jetson wird der Primärdienst aktuell bewusst konservativ betrieben.
- **Kein Chunking als Standard**: Texte werden am Stück verarbeitet.
- **Legacy nur als Sicherheitsnetz**: wenn faster wegen shared RAM / Startfehlern nicht tragfähig ist.
- **Jetson-spezifisch**: Preflight auf `MemAvailable`, nicht nur auf PyTorch-VRAM.
- **Bots/Automationen nutzen den Orchestrator**: nicht den nackten `/tts`-Endpoint.

## Architektur

```text
Bots / Wrapper / lokale Clients
          |
          v
  orchestrierte Entry-Points
    - ~/workspace/scripts/tts-telegram.sh
    - ~/workspace/scripts/tts-ursula.sh
          |
          v
      qwen3-tts (Port 5050)
          |
          +--> tts_server_faster.py   [faster]
          |      - textlängenabhängiger MemAvailable-Preflight
          |      - konservativer Start / Warmup
          |      - Produktionsmodus aktuell meist: profile=small
          |
          `--> tts_server.py          [legacy fallback]
                 - langsamer, aber robuster
```

Zusätzlich nutzt `~/workspace/scripts/tts-telegram.sh` den Python-Orchestrator `tts_telegram.py` für Routing, Fallback, OGG und Telegram-Upload.

## Profile

Die Profile werden zentral in `tts_config.py` definiert.

| Profil | Zweck | `max_seq_len` | `max_new_tokens` | `min_mem_available_gb` |
|---|---|---:|---:|---:|
| `large` | Langtext-orientiert, aber auf diesem Host oft zu speicherhungrig | 3584 | 4096 | 3.0 |
| `small` | konservativer Faster-Primärpfad für Jetson-Bootstabilität | 1536 | 1536 | 1.8 |
| `fallback` | Legacy-Sicherheitsnetz | – | 4096 | – |

## Installation

```bash
pip install -r requirements.txt
sudo ./install-service.sh
```

### Varianten

```bash
# Standarddienst
sudo ./install-service.sh

# Konservativer Faster-Betrieb
sudo ./install-service.sh --profile small

# Legacy explizit installieren
sudo ./install-service.sh --legacy
```

## API

### `POST /tts`

```bash
curl -X POST http://localhost:5050/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hallo Welt","speaker":"sohee","language":"german"}' \
  -o output.wav
```

**Wichtig:** Für Bots/Automationen ist das **nicht** der offizielle Produktionspfad. Dafür die Wrapper unten verwenden.

### `GET /health`

Liefert Runtime-Werte wie:
- `engine`, `profile`
- `mem_available_gb`, `gpu_memory_gb`, `gpu_reserved_gb`
- `startup_error`, `warmup_state`
- `busy`, `last_request_at`, `last_success_at`, `last_error`
- `ready` (bool), `ready_reason` — echte Readiness-Semantik

### `GET /ready`

Readiness-Probe. Liefert `ready: true` erst nach erster erfolgreicher Inferenz.
HTTP 200 = ready, HTTP 503 = not ready.

### `GET /info`

Liefert zusätzlich das aktive Profil, Routing-Schwellen und `ready`/`ready_reason`.

## Produktionspfade

### Telegram Voice

```bash
~/workspace/scripts/tts-telegram.sh "Text hier" <chat_id> [--reply-to <msg_id>] [--caption "..."]
```

- nutzt `tts_telegram.py`
- pausiert standardmäßig speicherstarke Nebenservices (vor allem Whisper) vor dem Run
- nutzt bei Bedarf Fallback
- stellt pausierte Dienste danach wieder her

### WAV-only lokal

```bash
~/workspace/scripts/tts-ursula.sh "Text" /tmp/output.wav [speaker] [language]
```

- pausiert standardmäßig Whisper vor dem Run
- wartet auf TTS-Health (`/health` → `status=ok`)
- startet `qwen3-tts` nur bei Bedarf neu
- ist bewusst leichter als der Telegram-Orchestrator, folgt aber denselben Betriebsprinzipien

### Readiness

- `/health` → `status=ok` = Modell geladen (Liveness)
- `/ready` → `ready=true` = erste Inferenz erfolgreich (Readiness)
- `~/workspace/scripts/tts-smoke.sh` = manueller Request-Test

## Praxis auf Jetson

Wichtige Erkenntnisse:
- Health allein reicht nicht; erste echte Inferenz kann trotzdem scheitern.
- Direkte `/tts`-Calls können unter RAM-Druck weiter 500/503 liefern.
- Der orchestrierte Pfad hat sich praktisch bewährt: kurzer Test erfolgreich, längerer Test erfolgreich mit Fallback.

## Betriebsregeln

- Für Bots und Automationen **nicht direkt** `POST /tts` auf `:5050` nutzen.
- Offizieller Produktionspfad:
  - Telegram Voice → `~/workspace/scripts/tts-telegram.sh`
  - WAV-only lokal → `~/workspace/scripts/tts-ursula.sh`
  - Remote von clawdbot → SSH-Wrapper auf diese Ursula-Skripte
- `ollama` ist aktuell bewusst deaktiviert, solange TTS-Stabilität Priorität hat.
- Der separate Zusatzservice `qwen3-tts-small` wurde wieder entfernt, um das Betriebsmodell zu vereinfachen.

## Wichtige Dateien

| Datei | Zweck |
|---|---|
| `tts_config.py` | Zentrale Profil-, Warmup- und Routing-Konfiguration |
| `tts_server_faster.py` | Faster-Server |
| `tts_server.py` | Legacy-Fallback-Server |
| `tts_telegram.py` | Python-Orchestrierung für Routing, Fallback, OGG und Telegram-Upload |
| `tests/test_tts_telegram.py` | Gezielte Tests für Routing- und Cleanup-Logik |
| `install-service.sh` | Systemd-Installation |
| `JETSON_NOTES.md` | Detaillierte Jetson-Analyse |

## Lizenz

Apache-2.0
