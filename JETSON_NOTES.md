# Qwen3-TTS auf Jetson Orin Nano — Langtext-first Notizen

*Stand: 2026-03-22*

## Kurzfazit

Für den Jetson-Orin-Nano-Use-Case ist die saubere Architektur jetzt:
- **`faster-small` als konservativer Dauer-Primärdienst auf Ursula**
- **`faster-large` als Zielbild / Langtextpfad**, wenn genug `MemAvailable` vorhanden ist
- **Legacy als Fallback** bei Speicherdruck oder Startfehlern
- **Preflight auf `MemAvailable`** statt blindem Vertrauen auf `torch.cuda.memory_allocated()`
- **kontrolliertes Warmup** statt unnötig aggressivem Startverhalten
- **textlängenabhängiges Routing** statt rein starrer 3-GB-Logik für jeden Request

## Warum diese Architektur?

Der Jetson hat **8 GB shared RAM**. CPU und GPU teilen sich denselben Speicherpool.
Darum reichen klassische CUDA-Metriken allein nicht aus.

### Das eigentliche Problem

Die OOM-/Startprobleme kommen nicht nur vom Modell selbst, sondern von der Kombination aus:
1. shared RAM auf Jetson
2. statischem Cache / CUDA-Graph-Pools der faster Engine
3. Warmup / erster Inferenzpfad
4. konkurrierenden Diensten wie Whisper oder Ollama

**Wichtig:** Ein System kann bei `torch.cuda.memory_allocated() ~2.5 GB` trotzdem faktisch zu knapp sein, weil `MemAvailable` schon kollabiert.

## Profil-Design

### `faster-large` (Zielbild / Langtextpfad)

Ziel: realistischer Langtextbetrieb ohne Chunking, wenn der Hostzustand es zulässt.

| Parameter | Wert | Begründung |
|---|---:|---|
| `max_seq_len` | 3584 | etwas weniger statischer Cache als 4096, aber weiter langtexttauglich |
| `max_new_tokens` | 4096 | verhindert frühes Abschneiden längerer Ausgaben |
| `min_mem_available_gb` | 3.0 | konservativer Langtext-Deckel |
| `warmup_mode` | `minimal` | kleineres Startup-Warmup |
| `warmup_max_new_tokens` | 192 | begrenzt Warmup-/Graph-Capture-Last |
| `startup_mem_headroom_gb` | 0.6 | verhindert Start auf der letzten Reserve |
| Attention | SDPA | Flash Attention 2 ist auf Jetson weiter problematisch |
| Dtype | bfloat16 | guter Trade-off für Jetson |

### Warum 3584 statt 4096?

`4096` war als Langtext-Default brauchbar, bindet aber mehr statischen Cache.
Für den konkreten 8-GB-Jetson ist `3584` der pragmatischere Default:
- immer noch langtext-first
- etwas weniger Start-/Speicherdruck
- ohne auf Chunking als Standard auszuweichen

### Textlängenabhängige Routing-Schwellen

Der faster-Server bewertet Requests jetzt differenzierter.

Default für `large`:

| Textlänge | benötigtes `MemAvailable` |
|---|---:|
| `<= 400` Zeichen | 1.6 GB |
| `<= 1200` Zeichen | 2.2 GB |
| `<= 2200` Zeichen | 2.8 GB |
| `> 2200` Zeichen | 3.0 GB |

Das ist absichtlich simpel:
- kurze Texte werden nicht unnötig streng geblockt
- echte Langtexte bleiben konservativ geschützt

### `small` nur optional

Es gibt weiter ein `small`-Profil (`2048 / 2048`) für Debug und Notfälle.
Es wird bewusst **nicht** als Hauptpfad dokumentiert.

### `legacy:fallback`

Die Legacy-Engine bleibt erhalten für:
- Startfehler der faster Engine
- knappen Speicher
- temporäre Notfälle im Telegram-Pfad

Sie ist langsamer, aber strukturell simpler.

## Server-Änderungen

### `tts_config.py`

Zentralisiert jetzt zusätzlich:
- Warmup-Steuerung
- Startup-Headroom
- textlängenabhängige Routing-Schwellen

Wichtige Env-Overrides:

```bash
QWEN3_TTS_PROFILE=large
QWEN3_TTS_MAX_SEQ_LEN=3584
QWEN3_TTS_MAX_NEW_TOKENS=4096
QWEN3_TTS_MIN_MEM_GB=3.0
QWEN3_TTS_WARMUP_MODE=minimal
QWEN3_TTS_WARMUP_MAX_NEW_TOKENS=192
QWEN3_TTS_STARTUP_HEADROOM_GB=0.6
```

### `tts_server_faster.py`

Jetzt klar als **Langtext-first Server mit kontrolliertem Warmup** strukturiert:
- prüft Preflight beim Start mit zusätzlichem Headroom
- kann Warmup bei niedrigem RAM **deferred** markieren statt aggressiv auszuführen
- versucht Warmup bei Bedarf später nochmals beim ersten Request
- prüft Requests textlängenabhängig
- liefert in `/health` und `/info` zusätzlich `warmup_state`
- gibt bei Speichermangel weiterhin **503 + Routing-Hinweis** zurück statt hart zu crashen

### `tts_server.py`

Auch die Legacy-Engine hat jetzt konsistente Warmup-/Health-Felder:
- `warmup_state`
- `last_warmup_at`
- begrenztes Warmup statt ungebremster Startlast

## Routing / Recovery im Telegram-Wrapper

`~/workspace/scripts/tts-telegram.sh` ist jetzt nur noch ein dünner Einstiegspunkt.
Die eigentliche Telegram-/TTS-Orchestrierung sitzt in `~/projects/qwen3-tts/tts_telegram.py`.

### Neues Verhalten

1. `/health` und `/info` lesen
2. textabhängige RAM-Schwelle aus dem Serverprofil ableiten
3. wenn faster gesund **und** genug RAM für genau diesen Text vorhanden → direkt dort generieren
4. wenn `startup_error` bereits klar auf Speichermangel zeigt → **kein unnötiger Restart**, direkt auf temporären Legacy-Fallback
5. vor Legacy-Start einen evtl. hängengebliebenen alten Fallback-Prozess auf dem Fallback-Port gezielt aufräumen
6. nach dem Fallback `faster-large` nur dann wiederherstellen, wenn `MemAvailable` inkl. Startup-Headroom mehrfach stabil erreicht wird
7. wenn das Zeitfenster dafür nicht reicht, den Restore ehrlich auslassen statt einen degradierten Sofort-Neustart zu erzwingen
8. WAV→OGG und Telegram `sendVoice` direkt im Python-Orchestrator ausführen

### Warum Python statt Bash?

Die alte Bash-Variante war funktional, aber strukturell fragil an genau den Stellen, die hier kritisch sind:
- JSON-Felder aus `/health` und `/info`
- Port-/PID-Erkennung für temporäre Fallback-Prozesse
- mehrstufige Fehlerpfade mit Retry/Fallback/Restore
- sauberes Aufräumen temporärer Dateien
- Telegram-Multipart-Upload plus TTS-/ffmpeg-Fehlerbehandlung

Mit dem kleinen Python-CLI sind diese Punkte jetzt an einem Ort gebündelt, testbar und deutlich weniger whitespace-/pipe-abhängig.

Zusätzlich gilt jetzt explizit:
- ein allgemeiner Faster-Neustart stoppt **nicht mehr pauschal Whisper/Ollama**
- das aggressive Freiräumen bleibt auf echte Memory-Notfälle begrenzt
- die Wiederherstellung von `faster-large` ist jetzt absichtlich konservativer: der Orchestrator wartet erst auf ein kleines stabiles Fenster oberhalb von `min_mem_available_gb + startup_mem_headroom_gb` (plus kleiner Hysterese), statt sofort einen degradierten Neustart zu provozieren

### Warum das besser ist

Vorher konnte der Wrapper bei knapper Maschine unnötig nervös werden.
Jetzt ist der Fallback klarer definiert:
- vorhersehbar
- reversibel
- weniger Restart-Stress
- näher an realem Alltagsbetrieb statt nur Happy-Path-Benchmarking

## Validierung dieses Schritts

### 1) Syntax / Integrität

Erfolgreich geprüft:
- `python3 -m py_compile tts_config.py tts_server_faster.py tts_server.py benchmark_longtext.py`
- `bash -n ~/workspace/scripts/tts-telegram.sh`

### 2) Threshold-Validierung

Direkt aus der neuen Serverlogik:

| Zeichen | erforderliches `MemAvailable` |
|---:|---:|
| 100 | 1.6 GB |
| 500 | 2.2 GB |
| 1000 | 2.2 GB |
| 1500 | 2.8 GB |
| 2200 | 2.8 GB |
| 3000 | 3.0 GB |

### 3) Startverhalten unter realem Speicherdruck

Auf dem Hostzustand dieser Session lag `MemAvailable` nur noch bei ungefähr **0.7–1.0 GB**.

Ergebnisse:
- `large` auf Port 5060 startet kontrolliert in **degraded** mit klarer Meldung:
  - `startup_error="Insufficient MemAvailable ... 0.82 GB < 3.60 GB"`
- `small` auf Port 5061 verhält sich analog:
  - `startup_error="Insufficient MemAvailable ... 0.70 GB < 2.40 GB"`
- damit wird sichtbar: **der neue Startup-Headroom greift**
- es gibt keinen zusätzlichen aggressiven Warmup-Versuch mehr, wenn schon vorab klar ist, dass die Maschine zu knapp ist

### 4) Harte Grenze bleibt sichtbar

Bei extrem knapper Maschine kann selbst `legacy` beim Laden noch an CUDA-/NvMap-OOM scheitern.
Das ist keine Routing-Schwäche, sondern die physische Grenze des Hosts.

## Bereits dokumentierte Leistungswerte aus sauberen Läufen

### faster-qwen3-tts

| Zeichen | Audio | Rechenzeit | RTF |
|---:|---:|---:|---:|
| 493 | 33s | 57s | 1.72 |
| 911 | 69s | 118s | 1.71 |

### Legacy-Vergleichswerte

| Zeichen | Audio | Rechenzeit | RTF |
|---:|---:|---:|---:|
| 1251 | 82s | 493s | 6.0 |
| 1485 | 112s | 697s | 6.2 |
| 1691 | 135s | 810s | 6.0 |
| 2090 | 140s | 841s | 6.0 |

## Technische Grenze

Die Grenze liegt aktuell nicht primär bei der Textlänge allein, sondern bei der Kombination aus:
- `max_seq_len=3584`
- CUDA Graph / static cache
- Shared-RAM-Lage des Hosts
- parallel laufenden Diensten

### Ehrliche Aussage zur 3000-Zeichen-Frage

**3000 Zeichen bleiben das Ziel, aber nicht unter beliebigem Hostzustand garantiert.**

Auf einem sauberen Jetson mit freigeräumtem Speicher ist das Profil dafür ausgelegt.
Auf dem konkret gemessenen Hostzustand dieser Session war bereits das Laden der Engine wegen extrem niedrigem `MemAvailable` sauber zu Recht blockiert.

Deshalb ist die Grenze derzeit praktisch:
- **langtexttauglich auf freigeräumtem Host:** ja
- **robust bei Speicherdruck:** besser als vorher, weil Start/Warmup/Fallback ehrlicher reagieren
- **3000 Zeichen unter starker Last:** aktuell nicht ehrlich garantierbar

## Install / Service

`install-service.sh` unterstützt jetzt zusätzlich Warmup-Optionen:

```bash
sudo ./install-service.sh
sudo ./install-service.sh --profile small
sudo ./install-service.sh --legacy
sudo ./install-service.sh \
  --max-seq-len 3584 \
  --max-new-tokens 4096 \
  --min-mem-gb 3.0 \
  --warmup-mode minimal \
  --warmup-max-new-tokens 192 \
  --startup-headroom-gb 0.6
```

## Empfehlung für den Betrieb

### Für den aktuellen Produktionsbetrieb auf Ursula

1. `faster-small` als Standardservice laufen lassen
2. `faster-large` nicht als Dauerdefault erzwingen, sondern gezielt nur bei ausreichend freiem RAM nutzen
3. Whisper / Ollama vor großen Langtextjobs möglichst stoppen
4. `MemAvailable` beobachten
5. bei Preflight-503 oder klarem Startup-Memory-Fehler kontrolliert auf small oder Legacy routen

### Nicht tun

- Chunking als Standardlösung einbauen
- mehrere fast identische Faster-Serverdateien pflegen
- `gpu_memory_gb` als alleinige Wahrheit behandeln
- Warmup künstlich hochdrehen, wenn der Host schon auf Reserve läuft

## Dateien

- `tts_config.py` — Profil-, Warmup- und Routing-Definitionen
- `tts_server_faster.py` — Langtext-first Faster-Server
- `tts_server.py` — Legacy-Fallback
- `benchmark_longtext.py` — Benchmark-Helfer
- `benchmark_results_longtext.json` — letzte Messserie

---

*Gepflegt von Ursula 🦎 auf dem Jetson Orin Nano*
