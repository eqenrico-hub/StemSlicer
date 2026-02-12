# StemSlicer

Automatic silence removal for FL Studio and any DAW.

Drop in stems with long silent sections, get back clean clips that play at the correct timeline positions.

## The Problem

Mixing engineers receive stems (vocals, bass, guitar, synths, etc.) that are full-length but only contain audio during certain sections. Manually cutting silent parts takes hours. Every major DAW has "Strip Silence" built-in — except FL Studio.

## How It Works

1. **Add your stems** (WAV, MP3, FLAC, OGG, AIF, M4A) — drag & drop or use buttons
2. **Preview the waveform** — stereo display with detected regions highlighted
3. **Adjust detection settings** (threshold, min silence duration, padding)
4. **Click Process**

### Output Modes

**Pre-Positioned (default)** — Each clip has leading silence matching its original position. Drop all clips at **position 0:00** in FL Studio and they play at the correct time.

**Raw Clips** — Just the audio portions, named with their timestamps.

### REAPER Export

Optionally generates a `.rpp` project file with all clips placed on separate tracks at correct positions. Open in REAPER for a fully arranged session.

## Requirements

- Python 3.9+
- ffmpeg (install with `brew install ffmpeg` on Mac)

## Setup

```bash
cd StemSlicer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python stemslicer.py
```

## Output Structure

```
StemSlicer_Output/
├── vocals/
│   ├── vocals_001_0m30s.wav
│   ├── vocals_002_1m00s.wav
│   └── vocals_003_2m00s.wav
├── bass/
│   ├── bass_001_0m00s.wav
│   └── bass_002_1m45s.wav
├── _reaper_clips/         (raw clips for REAPER project)
│   ├── vocals/
│   └── bass/
└── StemSlicer_Session.rpp (REAPER project file)
```

## Settings Guide

| Setting | Default | What it does |
|---------|---------|-------------|
| Silence Threshold | -40 dB | Audio below this level = silence. Lower = more sensitive. |
| Min Silence Duration | 500 ms | A gap must be this long to count as silence. |
| Padding | 50 ms | Extra audio kept before/after each detected region. |
