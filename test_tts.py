#!/usr/bin/env python3
"""Qwen3-TTS Test Script für Jetson Orin Nano"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import time
import sys

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hallo, ich bin Ursula."
    speaker = sys.argv[2] if len(sys.argv) > 2 else "serena"
    
    print(f"Lade Model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print(f"Text: {text}")
    print(f"Sprecher: {speaker}")
    
    start = time.time()
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language="german",
    )
    elapsed = time.time() - start
    
    output = "output/test.wav"
    sf.write(output, wavs[0], sr)
    duration = len(wavs[0]) / sr
    rtf = elapsed / duration
    
    print(f"Fertig: {elapsed:.1f}s Generation, {duration:.1f}s Audio")
    print(f"RTF: {rtf:.2f} (< 1 = Echtzeit)")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()
