# Qwen3-TTS Server for Jetson

Langtext-first HTTP-Server für [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) auf NVIDIA Jetson Orin Nano.

## Zielbild

- **Lange Texte zuerst**: Der Standardpfad ist `faster-large`
- **Kein Chunking als Standard**: Texte werden am Stück verarbeitet
- **Legacy nur als Sicherheitsnetz**: wenn `faster-large` wegen shared RAM / Startfehlern nicht tragfähig ist
- **Jetson-spezifisch**: Preflight auf `MemAvailable`, nicht nur auf PyTorch-VRAM

## Architektur

```text
Telegram / lokale Clients
        |
        v
  qwen3-tts service (Port 5050)
        |
        +--> tts_server_faster.py   [default: profile=large]
        |      - faster-qwen3-tts
        |      - CUDA Graphs
        |      - static cache via max_seq_len
        |      - langtext-first
        |
        `--> tts_server.py          [fallback]
               - qwen-tts legacy
               - ohne CUDA Graphs
               - langsamer, aber robuster
```

Zusätzlich nutzt `~/workspace/scripts/tts-telegram.sh` jetzt sauberes Routing:
1. primär `faster-large`
2. bei zu wenig `MemAvailable` oder Fehlern → temporärer `legacy`-Fallback
3. danach Wiederherstellung von `faster-large`

## Profile statt Datei-Duplikate

Die Profile werden zentral in `tts_config.py` definiert.

### Faster-Profile

| Profil | Zweck | `max_seq_len` | `max_new_tokens` | `min_mem_available_gb` |
|---|---|---:|---:|---:|
| `large` | **Standard für lange Texte** | 4096 | 4096 | 3.0 |
| `small` | Debug / Notfall, nicht Standard | 2048 | 2048 | 2.0 |

### Legacy-Profil

| Profil | Zweck | `max_new_tokens` | `non_streaming_mode` |
|---|---|---:|---|
| `fallback` | Sicherheitsnetz ohne CUDA Graphs | 4096 | `False` |

## Warum `faster-large`?

`faster-qwen3-tts` nutzt CUDA Graph Capture und einen statischen Talker-Cache.
Für lange Texte ist das auf Jetson der einzig sinnvolle Primärpfad, weil die Legacy-Engine bei langen Sequenzen massiv langsamer wird.

Wichtig ist dabei `max_seq_len`:
- zu klein → lange Requests riskieren harte Grenzen / ineffiziente Nutzung
- zu groß → mehr statischer Cache, höherer Druck auf shared RAM

Für den Jetson-Use-Case ist `4096` der saubere Langtext-Default:
- genug Puffer für echte Langtexte
- noch klar als Profil parametrierbar
- ohne zusätzliche Server-Dateien

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

# Overrides
sudo ./install-service.sh --max-seq-len 3584 --max-new-tokens 4096 --min-mem-gb 2.5
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

### `GET /health`

Liefert jetzt Jetson-relevante Runtime-Werte:
- `engine`
- `profile`
- `mem_available_gb`
- `gpu_memory_gb`
- `gpu_reserved_gb`
- `startup_error`
- `uptime_s`

### `GET /info`

Liefert zusätzlich das vollständige Profil und die aktiven Server-Parameter.

## Benchmarks / Realität auf Jetson

Es gibt zwei Wahrheiten gleichzeitig:

1. **Auf einem sauberen Jetson ist `faster-large` der richtige Pfad für lange Texte.**
2. **Auf einem bereits belegten Jetson ist `MemAvailable` die harte Grenze.**

Neu gemessene Langtext-Preflight-Läufe mit `faster-large` auf diesem Jetson:

| Zeichen | Ergebnis | MemAvailable vorher | Bemerkung |
|---:|---|---:|---|
| 1000 | 503 | 1.35 GB | sauber abgefangen, kein harter OOM |
| 1500 | 503 | 1.35 GB | sauber abgefangen |
| 2000 | 503 | 1.35 GB | sauber abgefangen |
| 2500 | 503 | 1.35 GB | sauber abgefangen |
| 3000 | 503 | 1.35 GB | sauber abgefangen |

Das ist **kein Qualitätsproblem der Engine**, sondern eine ehrliche Jetson-Grenze unter Speicherdruck.
Die Architektur reagiert jetzt korrekt: lieber sauber auf Legacy routen als OOM / Hänger.

Zusätzlicher Versuch mit abgesenkter Schwelle (`QWEN3_TTS_MIN_MEM_GB=1.0`) zeigte auf demselben Host bereits bei ~1000 Zeichen wieder `NvMap`-/Allocator-Fehler → bestätigt, dass der Default von 3.0 GB konservativ, aber sinnvoll ist.

## Benchmark-Script

```bash
python3 benchmark_longtext.py
```

Output: `benchmark_results_longtext.json`

Erfasst pro Lauf:
- Zeichen
- Erfolg / Fehler
- Audio-Dauer
- Rechenzeit
- RTF
- `MemAvailable` vorher / nachher
- `gpu_reserved_gb` vorher / nachher

## Telegram-Pfad

Wrapper: `~/workspace/scripts/tts-telegram.sh`

Neues Routing:
- prüft `/health` + `/info`
- bevorzugt `faster-large`
- routet bei niedrigem `MemAvailable` direkt auf temporären Legacy-Fallback
- stellt danach `faster-large` wieder her

## Wichtige Dateien

| Datei | Zweck |
|---|---|
| `tts_config.py` | Zentrale Profil- und Env-Konfiguration |
| `tts_server_faster.py` | Langtext-first Faster-Server |
| `tts_server.py` | Legacy-Fallback-Server |
| `benchmark_longtext.py` | Langtext-Benchmarking |
| `install-service.sh` | Systemd-Installation mit Profilen |
| `JETSON_NOTES.md` | Detaillierte Jetson-Analyse |

## Lizenz

Apache-2.0
