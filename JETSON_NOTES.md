# Qwen3-TTS auf Jetson Orin Nano — Performance & Optimierungen

*Stand: 2026-03-22*

---

## Konfiguration (aktuell)

| Parameter | Wert |
|-----------|------|
| **Hardware** | Jetson Orin Nano 8GB (ARM64, shared GPU/CPU RAM) |
| **Modell** | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice |
| **Engine** | faster-qwen3-tts 0.2.4 (CUDA Graph Capture) |
| **qwen-tts** | 0.1.1 |
| **Server** | `tts_server_faster.py` auf Port 5050 (Flask) |
| **Attention** | SDPA |
| **Dtype** | bfloat16 |
| **CUDA Alloc** | `expandable_segments:True` |
| **max_new_tokens** | 4096 |
| **VRAM (Modell)** | ~2.3 GB |
| **Stimmen** | aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian |

---

## Testergebnisse

### faster-qwen3-tts (aktuelle Engine)

| Zeichen | Audio | Rechenzeit | RTF | Speaker |
|---------|-------|-----------|-----|---------|
| 493 | 33s | 57s | **1.72** | sohee |
| 911 | 69s | 118s | **1.71** | sohee |

### qwen-tts Standard (Legacy-Engine, zum Vergleich)

| Zeichen | Audio | Rechenzeit | RTF | Speaker |
|---------|-------|-----------|-----|---------|
| 493 | 8s | 54s | 6.8 | sohee |
| 1251 | 82s | 493s | 6.0 | sohee |
| 1363 | 86s | 515s | 6.0 | sohee |
| 1485 | 112s | 697s | 6.2 | sohee |
| 1691 | 135s | 810s | 6.0 | sohee |
| 2090 | 140s | 841s | 6.0 | serena |

### Hochrechnung (faster Engine)

| Zeichen | ~Audio | ~Rechenzeit |
|---------|--------|-------------|
| 500 | ~30s | ~55s |
| 1000 | ~60s | ~2 Min |
| 1500 | ~100s | ~3 Min |
| 2000 | ~140s | ~4 Min |

---

## Warum faster-qwen3-tts?

Die Standard `qwen-tts` Library macht bei jedem Token einen separaten CUDA-Kernel-Launch. 
Bei langen Texten werden das tausende Launches → der Overhead dominiert (RTF ~6x).

`faster-qwen3-tts` captured den Berechnungsgraph einmalig beim Warmup als CUDA Graph. 
Alle folgenden Requests laufen über den statischen Graph → minimaler Overhead, **konstanter RTF ~1.7x**.

### Trade-off

| | Standard (`tts_server.py`) | Faster (`tts_server_faster.py`) |
|---|---|---|
| RTF | ~6.0 | **~1.7** |
| VRAM beim Start | ~3 GB frei nötig | ~5 GB frei nötig |
| Parallelbetrieb | Mit Whisper möglich | Whisper/Ollama müssen gestoppt sein |
| Stabilität | Bewährt | Neu, CUDA Graphs können auf Jetson Eigenheiten haben |

---

## Jetson-spezifische Optimierungen

### 1. SDPA statt Flash Attention 2
`flash_attention_2` hat mit qwen-tts ≥0.1.0 auf Jetson (CUDA sm_87) einen Kernel-Fehler:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
→ `attn_implementation="sdpa"` — PyTorch-native Alternative, gleiche Qualität.

### 2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Reduziert CUDA Memory Fragmentierung auf 8GB shared memory.

### 3. gc.collect() + torch.cuda.empty_cache()
Nach jeder Generierung und im Error-Handler. Gibt VRAM zurück an den Pool.

### 4. max_new_tokens=4096
Default war 2048 — zu wenig für längere Texte (Audio wird abgeschnitten).

---

## Bekannte Grenzen

- **8GB shared RAM** — GPU und CPU teilen sich den Speicher
- **Kein echtes Streaming** — Server gibt erst nach kompletter Generierung zurück
- **Memory-Leak** — Nach Tagen Dauerbetrieb kann der Prozess einfrieren
  → `tts-telegram.sh` hat Auto-Detection + Restart eingebaut
- **Faster Engine braucht exklusiven GPU-Zugriff** — Whisper/Ollama vorher stoppen

---

## Betrieb

### Service-Management
```bash
sudo systemctl status qwen3-tts
sudo systemctl restart qwen3-tts   # ~70s (Modell + CUDA Graph Warmup)
curl -s http://localhost:5050/health
curl -s http://localhost:5050/info
```

### Vor dem Start: VRAM freigeben
```bash
sudo systemctl stop whisper-server ollama
echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo systemctl start qwen3-tts
```

### Telegram Voice
```bash
~/workspace/scripts/tts-telegram.sh "Text" <chat_id> [--reply-to <msg_id>]
```

### Monitoring
```bash
tegrastats                           # GPU/RAM/Temp
sudo journalctl -u qwen3-tts -f     # Live-Logs
```

### Fallback auf Legacy-Engine
```bash
sudo ./install-service.sh --legacy   # Nutzt tts_server.py
```

---

## Changelog

### 2026-03-22
- **qwen-tts 0.0.5 → 0.1.1** — Neue Stimmen (ono_anna, uncle_fu)
- Flash Attention 2 → SDPA (Jetson sm_87 Kompatibilität)
- `expandable_segments:True` gegen CUDA-Fragmentierung
- `max_new_tokens` 2048 → 4096
- VRAM-Cleanup nach Warmup und im Error-Handler
- **faster-qwen3-tts 0.2.4** — CUDA Graph Capture, **3.5x Speedup** (RTF 6.0 → 1.72)
- `tts_server_faster.py` als neuer Standard-Server
- Repo aufgeräumt: README neu, stale Files entfernt

---

*Gepflegt von Ursula 🦎 auf dem Jetson Orin Nano*
