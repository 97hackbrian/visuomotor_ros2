#!/usr/bin/env python3
"""Genera los pitidos de inicio (uno, 880 Hz) y de fin (dos, 660 Hz) de los ensayos. Uso: make_beeps.py <carpeta>"""
import math, struct, sys, wave
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
def tone(path, freq, dur, gap=0.0, n=1, rate=44100, vol=0.5):
    frames = []
    for k in range(n):
        for i in range(int(rate * dur)):
            env = min(1.0, i / 400, (rate * dur - i) / 400)
            frames.append(int(32767 * vol * env * math.sin(2 * math.pi * freq * i / rate)))
        if k < n - 1:
            frames += [0] * int(rate * gap)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % len(frames), *frames))
tone(out / "inicio.wav", 880, 0.25)
tone(out / "fin.wav", 660, 0.18, gap=0.12, n=2)
