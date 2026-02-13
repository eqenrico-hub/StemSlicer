#!/usr/bin/env python3
"""Generate short test stems with clear regions and gaps for demo video."""

import os
import math
import wave
import struct

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "stems")
SR = 44100
BPM = 120
BEAT_MS = int(60000 / BPM)  # 500ms
BAR_MS = BEAT_MS * 4  # 2000ms
TOTAL_BARS = 8
TOTAL_MS = BAR_MS * TOTAL_BARS  # 16000ms


def make_samples(duration_ms):
    return [0] * int(SR * duration_ms / 1000)


def add_tone(samples, start_ms, duration_ms, freq, volume=0.5, sr=SR):
    """Add a sine tone to the sample buffer at the given position."""
    start_idx = int(sr * start_ms / 1000)
    n = int(sr * duration_ms / 1000)
    fade_n = min(int(sr * 0.005), n // 2)
    for i in range(n):
        idx = start_idx + i
        if idx >= len(samples):
            break
        t = i / sr
        val = volume * math.sin(2 * math.pi * freq * t)
        # Fade in/out
        if i < fade_n:
            val *= i / fade_n
        elif i > n - fade_n:
            val *= (n - i) / fade_n
        samples[idx] = max(-1.0, min(1.0, samples[idx] + val))


def write_wav(filepath, float_samples, sr=SR):
    int_samples = [int(s * 32767) for s in float_samples]
    int_samples = [max(-32768, min(32767, s)) for s in int_samples]
    with wave.open(filepath, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(struct.pack(f'{len(int_samples)}h', *int_samples))


def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Kick: bars 1-2 on every beat, bar 3-4 silent, bars 5-6 on every beat, 7-8 silent
    kick = make_samples(TOTAL_MS)
    for bar in [0, 1, 4, 5]:
        for beat in range(4):
            t = bar * BAR_MS + beat * BEAT_MS
            add_tone(kick, t, 80, 55, volume=0.8)
    write_wav(os.path.join(OUTPUT_DIR, "Kick.wav"), kick)

    # Snare: beats 2 and 4, bars 1-3 and 5-7
    snare = make_samples(TOTAL_MS)
    for bar in [0, 1, 2, 4, 5, 6]:
        for beat in [1, 3]:
            t = bar * BAR_MS + beat * BEAT_MS
            add_tone(snare, t, 60, 200, volume=0.5)
            # Add noise-like component (high freq mix)
            add_tone(snare, t, 60, 3500, volume=0.15)
            add_tone(snare, t, 60, 5200, volume=0.1)
    write_wav(os.path.join(OUTPUT_DIR, "Snare.wav"), snare)

    # Bass: sustained notes bars 1-2, silent 3-4, notes 5-7, silent 8
    bass = make_samples(TOTAL_MS)
    for bar in [0, 1, 4, 5, 6]:
        for beat in [0, 2]:
            t = bar * BAR_MS + beat * BEAT_MS
            add_tone(bass, t, BEAT_MS * 2 - 50, 82, volume=0.6)
    write_wav(os.path.join(OUTPUT_DIR, "Bass.wav"), bass)

    # Hi-Hat: 8th notes, bars 1-6 only
    hihat = make_samples(TOTAL_MS)
    for bar in range(6):
        for eighth in range(8):
            t = bar * BAR_MS + eighth * (BEAT_MS // 2)
            add_tone(hihat, t, 30, 8000, volume=0.2)
            add_tone(hihat, t, 30, 10000, volume=0.15)
    write_wav(os.path.join(OUTPUT_DIR, "HiHat.wav"), hihat)

    # Synth pad: long sustained chords, bars 1-3 and 5-8
    synth = make_samples(TOTAL_MS)
    for bar_start, bar_end in [(0, 3), (4, 8)]:
        t_start = bar_start * BAR_MS
        t_dur = (bar_end - bar_start) * BAR_MS - 100
        for freq in [261.6, 329.6, 392.0]:  # C major chord
            add_tone(synth, t_start, t_dur, freq, volume=0.15)
    write_wav(os.path.join(OUTPUT_DIR, "Synth Pad.wav"), synth)

    print(f"Generated 5 test stems in {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    generate()
