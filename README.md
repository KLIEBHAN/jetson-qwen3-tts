# Qwen3-TTS Server for Jetson

Langtext-first HTTP-Server für [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) auf NVIDIA Jetson Orin Nano.

## Zielbild

- **Lange Texte zuerst**: Der Standardpfad ist `faster-large`
- **Kein Chunking als Standard**: Texte werden am Stück verarbeitet
- **Legacy nur als Sicherheitsnetz**: wenn `faster-large` wegen shared RAM / Startfehlern nicht tragfähig ist
- **Jetson-spezifisch**: Preflight auf `MemAvailable`, nicht nur auf PyTorch-VRAM
- **Weniger Startstress**: Warmup ist jetzt kontrollierbar und kann bei knappem RAM sauber verschoben werden

## Architektur

```text
Telegram / lokale Clients
        |
        v
  qwen3-tts service (Port 5050)
        |
        +--> tts_server_faster.py   [default: profile=large]
        |      - faster-qwen3-tts
        |      - CUDA Graphs (kontrolliertes Warmup)
        |      - statischer Cache via max_seq_len
        |      - textlängenabhängiger MemAvailable-Preflight
        |      - langtext-first
        |
        `--> tts_server.py          [fallback]
               - qwen-tts legacy
               - ohne CUDA Graphs
               - langsamer, aber robuster
```

Zusätzlich nutzt `~/workspace/scripts/tts-telegram.sh` jetzt einen kleinen Python-Orchestrator (`tts_telegram.py`) für sauberes Routing:
1. primär `faster-large`
2. bei zu wenig `MemAvailable` oder klarem Startup-Memory-Fehler → temporärer `legacy`-Fallback
3. danach Wiederherstellung von `faster-large`
4. keine unnötigen Restart-Loops bei bereits erkennbarem Speichermangel
5. WAV→OGG und Telegram-Upload laufen ebenfalls in Python mit klaren Fehlerpfaden

## Profile statt Datei-Duplikate

Die Profile werden zentral in `tts_config.py` definiert.

### Faster-Profile

| Profil | Zweck | `max_seq_len` | `max_new_tokens` | `min_mem_available_gb` | Warmup |
|---|---|---:|---:|---:|---|
| `large` | **Standard für lange Texte** | 3584 | 4096 | 3.0 | `minimal` |
| `small` | Debug / Notfall, nicht Standard | 2048 | 2048 | 2.0 | `minimal` |

### Legacy-Profil

| Profil | Zweck | `max_new_tokens` | `non_streaming_mode` | Warmup |
|---|---|---:|---|---|
| `fallback` | Sicherheitsnetz ohne CUDA Graphs | 4096 | `False` | `minimal` |

## Warum `faster-large`?

`faster-qwen3-tts` nutzt CUDA Graph Capture und einen statischen Talker-Cache.
Für lange Texte ist das auf Jetson der sinnvollste Primärpfad, weil die Legacy-Engine bei langen Sequenzen massiv langsamer wird.

Wichtig ist dabei `max_seq_len`:
- zu klein → lange Requests riskieren harte Grenzen / ineffiziente Nutzung
- zu groß → mehr statischer Cache, höherer Druck auf shared RAM

Der neue Default ist **3584 statt 4096**:
- immer noch klar langtext-orientiert
- etwas weniger statischer Cache / Startdruck
- auf dem 8-GB-Jetson der pragmatischere Trade-off

## Warmup / Startverhalten

`tts_server_faster.py` hat jetzt ein kontrolliertes Warmup-Modell:
- `QWEN3_TTS_WARMUP_MODE=minimal` (Default): kleiner Warmup statt aggressivem Start-Capture
- `QWEN3_TTS_WARMUP_MODE=none`: kein Startup-Warmup; Capture bleibt bis zum ersten echten Request aus
- `QWEN3_TTS_WARMUP_MAX_NEW_TOKENS`: begrenzt die Warmup-Last zusätzlich
- `QWEN3_TTS_STARTUP_HEADROOM_GB`: zusätzlicher Sicherheitsabstand vor Warmup / Modellstart

Wenn der Host zu knapp ist, wird Warmup sauber **deferred** statt den Start aggressiv weiter zu belasten.

## Textlängenabhängiges Memory-Routing

Im faster-Server wird der Request-Preflight nicht mehr nur mit einer starren Schwelle bewertet.

Default für `large`:

| Textlänge | erforderliches `MemAvailable` |
|---|---:|
| `<= 400` Zeichen | 2.6 GB |
| `<= 1200` Zeichen | 2.8 GB |
| `<= 2200` Zeichen | 3.0 GB |
| `> 2200` Zeichen | 3.0 GB |

Das verhindert zwei Extreme:
- **zu aggressiv** für kurze/mittlere Texte
- **zu optimistisch** für echte Langtexte

## Installation

```bash
pip install -r requirements.txt
sudo ./install-service.sh
```

### Varianten

```bash
# Standard: faster-large
sudo ./install-service.sh

# Optional: anderes faster-Profil
sudo ./install-service.sh --profile small

# Legacy explizit installieren
sudo ./install-service.sh --legacy

# Langtext-sparsam, aber ohne Chunking als Standard
sudo ./install-service.sh --max-seq-len 3584 --max-new-tokens 4096 --min-mem-gb 3.0 \
  --warmup-mode minimal --warmup-max-new-tokens 192 --startup-headroom-gb 0.6
```

## Direkt starten

```bash
# faster-large (Standard)
QWEN3_TTS_PROFILE=large python3 tts_server_faster.py

# faster-small (nur Debug)
QWEN3_TTS_PROFILE=small python3 tts_server_faster.py

# legacy fallback
QWEN3_TTS_PROFILE=fallback python3 tts_server.py
```

## Wichtige Umgebungsvariablen

| Variable | Zweck |
|---|---|
| `QWEN3_TTS_PROFILE` | Profilname (`large`, `small`, `fallback`) |
| `QWEN3_TTS_MAX_SEQ_LEN` | Override für statischen Faster-Cache |
| `QWEN3_TTS_MAX_NEW_TOKENS` | Override für Generierungsgrenze |
| `QWEN3_TTS_MIN_MEM_GB` | Preflight-Schwelle für `MemAvailable` |
| `QWEN3_TTS_WARMUP_MODE` | Warmup-Steuerung (`minimal`, `none`) |
| `QWEN3_TTS_WARMUP_MAX_NEW_TOKENS` | Obergrenze für Warmup-Generierung |
| `QWEN3_TTS_STARTUP_HEADROOM_GB` | Zusätzlicher Sicherheitsabstand für Start/Warmup |
| `QWEN3_TTS_ROUTE_SHORT_MEM_GB` ... `QWEN3_TTS_ROUTE_XLONG_MEM_GB` | Feintuning der textabhängigen Routing-Schwellen |
| `PYTORCH_CUDA_ALLOC_CONF` | Standard: `expandable_segments:True` |

## API

### `POST /tts`

```bash
curl -X POST http://localhost:5050/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hallo Welt","speaker":"sohee","language":"german"}' \
  -o output.wav
```

Response-Header:
- `X-Audio-Duration`
- `X-Processing-Time`
- `X-RTF`
- `X-Torch-Allocated-GB`
- `X-Torch-Reserved-GB`
- `X-MemAvailable-GB`
- `X-Profile`
- `X-Warmup-State`

### `GET /health`

Liefert jetzt Jetson-relevante Runtime-Werte:
- `engine`
- `profile`
- `mem_available_gb`
- `gpu_memory_gb`
- `gpu_reserved_gb`
- `startup_error`
- `warmup_state`
- `uptime_s`

### `GET /info`

Liefert zusätzlich:
- vollständiges Profil inkl. Routing-Schwellen
- Warmup-Status
- aktive Server-Parameter

## Benchmarks / Realität auf Jetson

Es gibt zwei Wahrheiten gleichzeitig:

1. **Auf einem sauberen Jetson ist `faster-large` der richtige Pfad für lange Texte.**
2. **Auf einem bereits belegten Jetson ist `MemAvailable` die harte Grenze.**

### Bereits dokumentierte Vorwerte aus sauberen Läufen

| Engine | Zeichen | Audio | Rechenzeit | RTF |
|---|---:|---:|---:|---:|
| faster | 493 | 33s | 57s | 1.72 |
| faster | 911 | 69s | 118s | 1.71 |
| legacy | 1251 | 82s | 493s | 6.0 |
| legacy | 1485 | 112s | 697s | 6.2 |

### Neue Validierung dieses Schritts

Auf dem Hostzustand dieser Session war `MemAvailable` teils nur noch **~0.7–1.0 GB**.
Das ist absichtlich ein harter Test für Robustheit, nicht für Durchsatz.

Messbare Ergebnisse:
- neuer `large`-Default startet bei Speichermangel sauber mit **klarer `startup_error`-Meldung** statt blindem Warmup-Druck
- `/health` und `/info` zeigen jetzt zusätzlich `warmup_state`
- textabhängige Schwellen ergeben für `large`:
  - 100 Zeichen → 1.6 GB
  - 1000 Zeichen → 2.2 GB
  - 1500/2200 Zeichen → 2.8 GB
  - 3000 Zeichen → 3.0 GB
- bei extrem niedrigem RAM blieb das System in **degraded**, aber kontrolliert ansprechbar
- der Wrapper kann bei explizitem Speichermangel jetzt direkt auf Legacy gehen, statt erst Restart-Schleifen zu drehen

## Telegram-Pfad

Wrapper: `~/workspace/scripts/tts-telegram.sh`

Der Workspace-Wrapper ist jetzt absichtlich dünn und ruft das Repo-CLI `tts_telegram.py` auf.
Die eigentliche Orchestrierung liegt damit zentral, testbar und ohne fragile Bash-JSON-/PID-Logik im Repository.

Neues Routing:
- prüft `/health` + `/info`
- berücksichtigt Textlänge gegen die vom Server gelieferten Routing-Schwellen
- nutzt bei healthy primary adaptiv `large -> small -> legacy`
- behandelt unhealthy/degraded primary konservativ und geht direkt auf `legacy`, statt blind `small` zu erzwingen
- vermeidet Restart-Stürme, wenn `startup_error` bereits auf `Insufficient MemAvailable` zeigt
- kann optional vor Langtext RAM freimachen (Whisper/Ollama stoppen, Cache-Drop als Notfallmaßnahme)
- nutzt Legacy nur temporär und stellt danach `faster-large` nur dann wieder her, wenn `MemAvailable` für den Startup-Pfad wieder stabil tragfähig ist
- führt WAV→OGG und Telegram `sendVoice` im selben Python-Prozess aus

## Wichtige Dateien

| Datei | Zweck |
|---|---|
| `tts_config.py` | Zentrale Profil-, Warmup- und Routing-Konfiguration |
| `tts_server_faster.py` | Langtext-first Faster-Server |
| `tts_server.py` | Legacy-Fallback-Server |
| `tts_telegram.py` | Kleine Python-Orchestrierung für Routing, Fallback, OGG und Telegram-Upload |
| `tests/test_tts_telegram.py` | Gezielte Tests für Routing- und Cleanup-Logik |
| `benchmark_longtext.py` | Langtext-Benchmarking |
| `install-service.sh` | Systemd-Installation mit Profilen + Warmup-Optionen |
| `JETSON_NOTES.md` | Detaillierte Jetson-Analyse |

## Lizenz

Apache-2.0
