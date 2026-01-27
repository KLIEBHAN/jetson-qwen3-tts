#!/usr/bin/env python3
"""Standalone Qwen3-TTS test script for Jetson."""

import os
import sys
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hallo, ich bin ein Jetson."
    speaker = sys.argv[2] if len(sys.argv) > 2 else "serena"
    language = sys.argv[3] if len(sys.argv) > 3 else "german"
    
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print(f"Text: {text}")
    print(f"Speaker: {speaker}")
    print(f"Language: {language}")
    
    start = time.time()
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
    )
    elapsed = time.time() - start
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    output = "output/test.wav"
    sf.write(output, wavs[0], sr)
    duration = len(wavs[0]) / sr
    rtf = elapsed / duration
    
    print(f"Done: {elapsed:.1f}s generation, {duration:.1f}s audio")
    print(f"RTF: {rtf:.2f} (< 1 = realtime)")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()
