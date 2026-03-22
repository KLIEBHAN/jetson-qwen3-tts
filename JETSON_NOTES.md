# Qwen3-TTS auf Jetson Orin Nano — Performance & Optimierungen

*Stand: 2026-03-22*

---

## Konfiguration

| Parameter | Wert |
|-----------|------|
| **Hardware** | Jetson Orin Nano 8GB (ARM64, shared GPU/CPU RAM) |
| **Modell** | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice |
| **qwen-tts** | 0.1.1 (PyPI) |
| **Server** | `tts_server.py` auf Port 5050 (Flask) |
| **Systemd-Service** | `qwen3-tts.service` |
| **Attention** | SDPA (flash_attention_2 hat Kernel-Issues auf sm_87) |
| **Dtype** | bfloat16 |
| **CUDA Alloc** | `expandable_segments:True` (gegen Fragmentierung) |
| **Streaming** | `non_streaming_mode=False` (reduziert Peak-VRAM) |
| **max_new_tokens** | 4096 (default war 2048) |
| **VRAM (Modell)** | ~2.0 GB |
| **Sample Rate** | 24kHz WAV Output |
| **Stimmen** | aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian |

---

## Testergebnisse

Alle Tests ohne Text-Splitting, kompletter Text in einem Request.

| # | Zeichen | Audio (s) | Rechenzeit (s) | RTF | Speaker | Status |
|---|---------|-----------|----------------|-----|---------|--------|
| 1 | 493 | 8 | ~54 | ~6.8 | sohee | ✅ |
| 2 | ~800 | 8 | ~55 | ~6.9 | serena | ✅ |
| 3 | ~800 | 10 | ~68 | ~6.8 | ryan (EN) | ✅ |
| 4 | 1251 | 82 | 493 | 6.0 | sohee | ✅ |
| 5 | 1363 | 86 | 515 | 6.0 | sohee | ✅ |
| 6 | 1485 | 112 | 697 | 6.2 | sohee | ✅ |
| 7 | 1691 | 135 | 810 | 6.0 | sohee | ✅ |
| 8 | 2090 | 140 | 841 | 6.0 | serena | ✅ |

*RTF = Real-Time Factor (Rechenzeit / Audio-Dauer). ~6x bedeutet 10s Audio brauchen ~60s Rechenzeit.*

### Hochrechnung

| Zeichen | ~Audio | ~Rechenzeit |
|---------|--------|-------------|
| 500 | 8–10s | ~55s |
| 1000 | ~60s | ~6 Min |
| 1500 | ~110s | ~11 Min |
| 2000 | ~140s | ~14 Min |
| 2500+ | ? | >15 Min |

---

## Optimierungen (was hat geholfen)

### 1. SDPA statt Flash Attention 2
`flash_attention_2` hat mit qwen-tts ≥0.1.0 auf Jetson (CUDA sm_87) einen Kernel-Fehler:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
→ **Fix:** `attn_implementation="sdpa"` — PyTorch-native Alternative, gleiche Qualität.

### 2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Reduziert CUDA Memory Fragmentierung. Ohne dieses Flag kam bei ~1000+ Zeichen OOM:
```
CUDA out of memory. Tried to allocate 106.00 MiB. GPU 0 has a total capacity of 7.44 GiB
```
→ **Fix:** `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` vor torch-Import.

### 3. non_streaming_mode=False
Laut Qwen-Doku "simuliert Streaming-Text-Input" — verarbeitet intern sequenzieller.
Reduziert Peak-VRAM bei langen Texten deutlich. War der entscheidende Fix gegen OOM.

### 4. max_new_tokens=4096 (statt default 2048)
Erlaubt längere Audio-Ausgabe ohne Abschneiden.

### 5. gc.collect() + torch.cuda.empty_cache()
Nach jeder Generierung UND im Error-Handler. Kritisch bei 8GB shared memory.
Auch nach dem Warmup bei Server-Start.

### 6. VRAM-Cleanup nach Warmup
Der erste Warmup-Inference hinterlässt Speicher-Artefakte. Cleanup danach gibt ~200MB frei.

---

## Bekannte Grenzen

- **8GB shared RAM** — GPU und CPU teilen sich den Speicher
- **RTF ~6x** — 10s Audio brauchen ~60s Rechenzeit
- **Kein echtes Streaming** — Server gibt erst nach kompletter Generierung zurück
- **Memory-Leak** — Nach Tagen Dauerbetrieb friert der Prozess ein (Health OK, /tts antwortet nicht)
  → `tts-telegram.sh` hat Auto-Detection + Restart eingebaut
- **Maximale Textlänge** — 2000+ Zeichen funktioniert, aber dauert >14 Min
- **Kein paralleler Betrieb** — Whisper/Ollama gleichzeitig erhöht OOM-Risiko

---

## Tipps für Betrieb

### Service-Management
```bash
sudo systemctl status qwen3-tts
sudo systemctl restart qwen3-tts   # Braucht ~30s zum Laden
curl -s http://localhost:5050/health
curl -s http://localhost:5050/info
```

### Für Telegram Voice
Immer `tts-telegram.sh` verwenden — macht Health-Check, Quick-Test, Auto-Restart, WAV→OGG, Versand:
```bash
~/workspace/scripts/tts-telegram.sh "Text" <chat_id> [--reply-to <msg_id>]
```

### exec-Timeouts
- **Kurze Texte (<500 Zeichen):** timeout=120, yieldMs=60000
- **Mittlere Texte (500–1500):** timeout=900, yieldMs=600000
- **Lange Texte (1500+):** timeout=1800, yieldMs=900000

### Monitoring
```bash
tegrastats                           # GPU/RAM/Temp
sudo journalctl -u qwen3-tts -f     # Live-Logs
```

---

## faster-qwen3-tts Engine (empfohlen)

**3.5x schneller** durch CUDA Graph Capture — `pip install faster-qwen3-tts`

### Wie es funktioniert
Statt bei jedem Token einen separaten CUDA-Kernel-Launch zu machen, captured `faster-qwen3-tts` 
den Berechnungsgraph einmalig beim Warmup. Folgeaufrufe laufen dann über den statischen Graph → 
minimal Overhead, konstanter RTF unabhängig von der Textlänge.

### Performance-Vergleich

| Engine | Zeichen | Audio | Rechenzeit | RTF |
|--------|---------|-------|-----------|-----|
| qwen-tts (alt) | 493 | 32s | 197s | ~6.0 |
| **faster-qwen3-tts** | 493 | 32s | 57s | **1.72** |
| qwen-tts (alt) | 1079 | 61s | 493s | ~6.0 |
| **faster-qwen3-tts** | 911 | 69s | 118s | **1.71** |

### Einschränkung
Braucht mehr freien VRAM beim Laden (~5GB frei nötig). Whisper/Ollama müssen gestoppt sein.
Server: `tts_server_faster.py` (Standard-Service nutzt das automatisch).

---

## Changelog

### 2026-03-22
- **qwen-tts 0.0.5 → 0.1.1** — Neue Stimmen (ono_anna, uncle_fu), API-Verbesserungen
- Flash Attention 2 → SDPA (Jetson-Kompatibilität)
- `expandable_segments:True` gegen CUDA-Fragmentierung
- `non_streaming_mode=False` gegen OOM bei langen Texten
- `max_new_tokens` 2048 → 4096
- VRAM-Cleanup nach Warmup und im Error-Handler
- **faster-qwen3-tts 0.2.4** — CUDA Graph Capture, **3.5x Speedup** (RTF 6.0 → 1.72)
- `tts_server_faster.py` als neuer Standard-Server

---

*Gepflegt von Ursula 🦎 auf dem Jetson Orin Nano*
