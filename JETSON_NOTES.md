# Qwen3-TTS auf Jetson Orin Nano — Langtext-first Notizen

*Stand: 2026-03-22*

## Kurzfazit

Für den Jetson-Orin-Nano-Use-Case ist die saubere Architektur jetzt:
- **`faster-large` als Standard** für lange Texte
- **Legacy als Fallback** bei Speicherdruck oder Startfehlern
- **Preflight auf `MemAvailable`** statt blindem Vertrauen auf `torch.cuda.memory_allocated()`

## Warum diese Architektur?

Der Jetson hat **8 GB shared RAM**. CPU und GPU teilen sich denselben Speicherpool.
Darum reichen klassische CUDA-Metriken allein nicht aus.

### Das eigentliche Problem

Die früheren OOM-/Startprobleme kamen nicht nur vom Modell selbst, sondern von der Kombination aus:
1. shared RAM auf Jetson
2. statischem Cache / CUDA-Graph-Pools der faster Engine
3. zusätzlichem Warmup / Request-Overhead
4. konkurrierenden Diensten wie Whisper

**Wichtig:** Ein System kann bei `torch.cuda.memory_allocated() ~2.5 GB` trotzdem faktisch zu knapp sein, weil `MemAvailable` schon kollabiert.

## Profil-Design

### `faster-large` (neu: Primärpfad)

Ziel: realistischer Langtextbetrieb ohne Chunking als Default.

| Parameter | Wert | Begründung |
|---|---:|---|
| `max_seq_len` | 4096 | sinnvoller statischer Cache für lange Sequenzen |
| `max_new_tokens` | 4096 | verhindert frühes Abschneiden längerer Ausgaben |
| `min_mem_available_gb` | 3.0 | konservativer, aber stabiler Jetson-Preflight |
| Attention | SDPA | Flash Attention 2 ist auf Jetson weiter problematisch |
| Dtype | bfloat16 | guter Trade-off für Jetson |

### Warum nicht noch größer?

Größerer `max_seq_len` bedeutet mehr statischen Cache.
Das erhöht die Robustheit gegen lange Requests **nur dann**, wenn der Host genug freien shared RAM hat.
Auf diesem Jetson kippt das Preis-Leistungs-Verhältnis schnell.

Darum ist `4096` aktuell ein sauberer Langtext-Default:
- klar langtext-orientiert
- nicht künstlich klein
- aber noch vertretbar im shared-RAM-Modell

### `small` nur optional

Es gibt optional ein `small`-Profil (`2048 / 2048`) für Debug und Notfälle.
Es wird bewusst **nicht** als Hauptpfad dokumentiert.

### `legacy:fallback`

Die Legacy-Engine bleibt erhalten für:
- Startfehler der faster Engine
- knappen Speicher
- temporäre Notfälle im Telegram-Pfad

Sie ist langsamer, aber strukturell simpler.

## Server-Refactor

### Neu eingeführt: `tts_config.py`

Statt mehrere fast identische Serverdateien zu pflegen, sind die Parameter jetzt zentralisiert:
- Faster-Profile
- Legacy-Profil
- Env-Overrides

Damit lassen sich Profil-Anpassungen sauber über Umgebungsvariablen fahren:

```bash
QWEN3_TTS_PROFILE=large
QWEN3_TTS_MAX_SEQ_LEN=4096
QWEN3_TTS_MAX_NEW_TOKENS=4096
QWEN3_TTS_MIN_MEM_GB=3.0
```

### `tts_server_faster.py`

Jetzt klar als **Langtext-first Server** strukturiert:
- lädt Profil `large` standardmäßig
- prüft Preflight beim Start
- prüft Preflight auch vor jedem Request
- gibt bei Speichermangel **503 + Routing-Hinweis** zurück statt hart zu crashen
- liefert in `/health` und `/info` Jetson-relevante Werte

### `tts_server.py`

Jetzt klar als **Fallback-Engine** dokumentiert:
- gleiche Observability-Felder (`engine`, `profile`, `startup_error`, Memory-Snapshot)
- weiterhin ohne CUDA Graphs
- nur Sicherheitsnetz, nicht Hauptpfad

## Routing / Preflight im Telegram-Wrapper

`~/workspace/scripts/tts-telegram.sh` wurde auf sauberes Routing umgebaut.

### Neues Verhalten

1. Status von `/health` und `/info` lesen
2. wenn `faster-large` gesund **und** genug `MemAvailable` vorhanden → direkt dort generieren
3. wenn `MemAvailable` zu niedrig oder faster Request fehlschlägt → temporären Legacy-Server starten
4. WAV über Legacy erzeugen
5. anschließend `faster-large` wiederherstellen

### Warum das besser ist

Vorher war der Fallback eher ein Notanker.
Jetzt ist er ein definierter Teil der Architektur:
- vorhersehbar
- reversibel
- ohne dauerhafte Umschaltung des Systems auf Legacy

## Benchmarks

### 1) Dokumentierte Vorwerte aus früheren sauberen Läufen

#### faster-qwen3-tts

| Zeichen | Audio | Rechenzeit | RTF |
|---:|---:|---:|---:|
| 493 | 33s | 57s | 1.72 |
| 911 | 69s | 118s | 1.71 |

Das bestätigt den grundsätzlichen Vorteil von `faster` für lange Texte.

#### Legacy-Vergleichswerte

| Zeichen | Audio | Rechenzeit | RTF |
|---:|---:|---:|---:|
| 1251 | 82s | 493s | 6.0 |
| 1485 | 112s | 697s | 6.2 |
| 1691 | 135s | 810s | 6.0 |
| 2090 | 140s | 841s | 6.0 |

### 2) Neue Langtext-Benchmarkserie (dieser Umbau)

Datei: `benchmark_results_longtext.json`

#### Messung A — Standard-Preflight (`faster-large`, `min_mem_available_gb=3.0`)

Alle Zielgrößen wurden auf dem laufenden Jetson **sauber abgefangen**:

| Zeichen | Erfolg | Audio | Rechenzeit | RTF | MemAvailable | Ergebnis |
|---:|---|---:|---:|---:|---:|---|
| 1000 | nein | – | – | – | 1.35 GB | HTTP 503 Preflight |
| 1500 | nein | – | – | – | 1.35 GB | HTTP 503 Preflight |
| 2000 | nein | – | – | – | 1.35 GB | HTTP 503 Preflight |
| 2500 | nein | – | – | – | 1.35 GB | HTTP 503 Preflight |
| 3000 | nein | – | – | – | 1.35 GB | HTTP 503 Preflight |

**Interpretation:** Die neue Architektur verhält sich korrekt. Kein harter OOM, kein Hänger, sondern definierte Rückgabe an das Routing.

#### Messung B — experimentell abgesenkte Schwelle (`QWEN3_TTS_MIN_MEM_GB=1.0`)

Ziel war zu prüfen, ob man mit aggressiverem Routing noch echte Langtextläufe erzwingen kann.

Ergebnis:
- Server startete zwar noch
- bereits beim ersten Langtextversuch traten `NvMapMemAllocInternalTagged ... error 12` auf
- der Request endete mit HTTP 500

**Interpretation:** Unter ~1.3-1.4 GB `MemAvailable` ist `faster-large` auf diesem Host nicht mehr verlässlich langtexttauglich.

## Technische Grenze

Die Grenze liegt aktuell nicht primär bei der Textlänge allein, sondern bei der Kombination aus:
- `max_seq_len=4096`
- CUDA Graph / static cache
- aktueller Shared-RAM-Lage des Hosts
- parallel laufenden Diensten

### Ehrliche Aussage zur 3000-Zeichen-Frage

**3000 Zeichen sind architektonisch das Ziel, aber nicht unter beliebigem Hostzustand garantiert.**

Auf einem sauberen Jetson mit freigeräumtem Speicher ist das Profil dafür ausgelegt.
Auf dem konkret gemessenen Hostzustand dieser Session war bereits der Einstieg in die Langtextgenerierung speicherseitig nicht mehr sauber tragfähig.

Deshalb ist die Grenze derzeit praktisch:
- **langtexttauglich auf freigeräumtem Host:** ja
- **robust bei Speicherdruck:** nur mit sauberem Legacy-Fallback
- **3000 Zeichen unter Last:** aktuell nicht ehrlich garantierbar

## Install / Service

`install-service.sh` unterstützt jetzt Profile und Overrides:

```bash
sudo ./install-service.sh
sudo ./install-service.sh --profile small
sudo ./install-service.sh --legacy
sudo ./install-service.sh --max-seq-len 3584 --max-new-tokens 4096 --min-mem-gb 2.5
```

## Empfehlung für den Betrieb

### Für echte Langtexte

1. `faster-large` als Standardservice laufen lassen
2. Whisper / andere GPU-lastige Dienste vor Langtextjobs möglichst stoppen
3. `MemAvailable` beobachten
4. bei Preflight-503 kontrolliert auf Legacy routen

### Nicht tun

- Chunking als Standardlösung einbauen
- mehrere fast identische Faster-Serverdateien pflegen
- `gpu_memory_gb` als alleinige Wahrheit behandeln

## Dateien

- `tts_config.py` — Profil-Definitionen
- `tts_server_faster.py` — Langtext-first Faster-Server
- `tts_server.py` — Legacy-Fallback
- `benchmark_longtext.py` — Benchmark-Helfer
- `benchmark_results_longtext.json` — letzte Messserie

---

*Gepflegt von Ursula 🦎 auf dem Jetson Orin Nano*
