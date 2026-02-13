#!/usr/bin/env python3
"""
StemSlicer — Works with any DAW — FL Studio, Ableton, Logic, REAPER, Cubase, Pro Tools

Detects silent regions in audio stems, splits them into separate clips,
and outputs "pre-positioned" WAV files (with leading silence) so you can
drop them all at position 0:00 in FL Studio and they play at the correct time.

Also exports a REAPER .rpp project file with all clips placed on the timeline.
"""

import os
import sys
import platform
import wave
import struct
import math
import array
import json
import zipfile
import subprocess
import hmac
import hashlib
import base64
from pathlib import Path
from datetime import timedelta

# ── Bundled ffmpeg support ──
# When frozen with PyInstaller, ffmpeg is bundled alongside the executable.
# Tell pydub where to find it so the user never needs to install ffmpeg.
def _setup_ffmpeg():
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        bundle_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
        # Check common locations PyInstaller might place binaries
        for search_dir in [bundle_dir, os.path.dirname(sys.executable)]:
            ffmpeg_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
            ffmpeg_path = os.path.join(search_dir, ffmpeg_name)
            if os.path.isfile(ffmpeg_path):
                os.environ["PATH"] = search_dir + os.pathsep + os.environ.get("PATH", "")
                return
    # Not frozen or ffmpeg not bundled — rely on system PATH
    # We'll check later and show a user-friendly message if missing

_setup_ffmpeg()

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ─────────────────────────────────────────────────────────────────────────────
# LICENSE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _license_config_path():
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "StemSlicer" / "license.json"
    return Path.home() / ".stemslicer" / "license.json"


def _get_license_secret():
    # Obfuscated secret — split across parts and base64 encoded
    _p1 = b"U3RlbVNsaWNlci1I"  # StemSlicer-H
    _p2 = b"TUFDLVBYRU1JVU0t"  # MAC-PREMIUM-
    _p3 = b"UEFZUEFMLTVFVVIt"  # PAYPAL-5EUR-
    _p4 = b"MjAyNExJQ0VOU0U="  # 2024LICENSE
    return base64.b64decode(_p1 + _p2 + _p3 + _p4)


def _generate_license_key(email):
    secret = _get_license_secret()
    digest = hmac.new(secret, email.strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()
    return f"SS-{digest[:8].upper()}"


def _validate_license(email, key):
    expected = _generate_license_key(email)
    return hmac.compare_digest(expected, key.strip().upper())


def _is_licensed():
    path = _license_config_path()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        email = data.get("email", "")
        key = data.get("key", "")
        return _validate_license(email, key)
    except Exception:
        return False


def _save_license(email, key):
    path = _license_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"email": email.strip().lower(), "key": key.strip().upper()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}

# ── Instrument Detection & Color System ──
# Keyword-based instrument recognition from filenames (like REAPER SWS auto-color).
# Each category has: keywords to match, a display color (hex), and REAPER PEAKCOL int.

import re

INSTRUMENT_CATEGORIES = [
    {
        "name": "Kick",
        "keywords": ["kick", "kck", "kik"],
        "color": "#E74C3C",       # Red
        "reaper_col": 3947580,
    },
    {
        "name": "Snare",
        "keywords": ["snare", "snr", "sn"],
        "color": "#E67E22",       # Orange
        "reaper_col": 2328630,
    },
    {
        "name": "Hi-Hat",
        "keywords": ["hihat", "hi-hat", "hh", "hat"],
        "color": "#F1C40F",       # Yellow
        "reaper_col": 1023487,
    },
    {
        "name": "Drums",
        "keywords": ["drum", "drums", "drm", "perc", "percussion", "tom", "cymbal", "cym",
                      "clap", "rim", "shaker", "tamb", "conga", "bongo", "timbale", "overhead", "oh"],
        "color": "#D35400",       # Dark orange
        "reaper_col": 55295,
    },
    {
        "name": "Bass",
        "keywords": ["bass", "bas", "sub", "808", "low"],
        "color": "#3498DB",       # Blue
        "reaper_col": 14398692,
    },
    {
        "name": "Guitar",
        "keywords": ["guitar", "gtr", "guit", "gtrs", "elec gtr", "ac gtr",
                      "acoustic guitar", "electric guitar"],
        "color": "#27AE60",       # Green
        "reaper_col": 6336352,
    },
    {
        "name": "Vocals",
        "keywords": ["vocal", "vocals", "vox", "voc", "voice", "sing", "choir", "chorus",
                      "adlib", "ad lib", "harmony", "harm", "back vox",
                      "lead vox", "bgv", "bv"],
        "color": "#E91E63",       # Pink
        "reaper_col": 6556898,
    },
    {
        "name": "Keys",
        "keywords": ["keys", "key", "piano", "pno", "organ", "rhodes", "wurli",
                      "clav", "synth", "syn", "pad", "lead", "arp", "pluck"],
        "color": "#9B59B6",       # Purple
        "reaper_col": 11960310,
    },
    {
        "name": "Strings",
        "keywords": ["string", "strings", "str", "violin", "viola", "cello", "orchestra",
                      "orch", "ensemble"],
        "color": "#1ABC9C",       # Teal
        "reaper_col": 10271228,
    },
    {
        "name": "Brass",
        "keywords": ["brass", "horn", "trumpet", "trp", "trombone", "sax",
                      "saxophone", "flute", "woodwind"],
        "color": "#F39C12",       # Amber
        "reaper_col": 1219327,
    },
    {
        "name": "FX",
        "keywords": ["fx", "sfx", "effect", "riser", "sweep", "impact", "trans",
                      "transition", "noise", "atmos", "atmosphere", "ambient",
                      "foley", "whoosh", "reverse"],
        "color": "#00BCD4",       # Cyan
        "reaper_col": 13959168,
    },
]

# Fallback for unrecognized instruments
_DEFAULT_CATEGORY = {
    "name": "Other",
    "color": "#95A5A6",  # Grey
    "reaper_col": 16576,
}


def detect_instrument(filename):
    """
    Detect instrument category from a filename using keyword matching.
    Returns the category dict (name, color, reaper_col).
    """
    # Normalize: lowercase, replace separators with spaces
    name = Path(filename).stem.lower()
    name = re.sub(r'[_\-\.]+', ' ', name)

    for cat in INSTRUMENT_CATEGORIES:
        for kw in cat["keywords"]:
            # Word boundary match — allows digits/spaces/start/end as boundaries
            if re.search(r'(?:^|[\s\d])' + re.escape(kw) + r'(?:[\s\d]|$)', name):
                return cat
    return _DEFAULT_CATEGORY


def clean_stem_name(filename):
    """
    Clean up a stem filename for export. Non-destructive — only fixes common issues.

    - Trims leading/trailing whitespace
    - Normalizes multiple spaces/underscores to single
    - Removes leading/trailing underscores and dashes
    - Fixes double extensions (.wav.wav)
    - Preserves the original name if it's already clean
    """
    name = Path(filename).stem

    # Trim whitespace
    name = name.strip()

    # Normalize multiple underscores/spaces/dashes to single
    name = re.sub(r'[_]{2,}', '_', name)
    name = re.sub(r'[ ]{2,}', ' ', name)
    name = re.sub(r'[-]{2,}', '-', name)

    # Remove leading/trailing separators
    name = name.strip('_- ')

    # If name is empty after cleanup, use original
    if not name:
        name = Path(filename).stem

    return name


def detect_regions(audio_path, threshold_db=-40, min_silence_ms=500, padding_ms=50):
    """
    Detect non-silent regions in an audio file.

    Returns list of (start_ms, end_ms) tuples for each non-silent region.
    """
    audio = AudioSegment.from_file(audio_path)
    regions = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=threshold_db,
        seek_step=10,
    )

    # Apply padding (extend each region but clamp to file bounds)
    total_ms = len(audio)
    padded = []
    for start, end in regions:
        start = max(0, start - padding_ms)
        end = min(total_ms, end + padding_ms)
        padded.append((start, end))

    # Merge overlapping regions after padding
    if not padded:
        return []

    merged = [padded[0]]
    for start, end in padded[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def format_timestamp(ms):
    """Format milliseconds as XmYYs (e.g., 1m30s, 0m00s)."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}m{seconds:02d}s"


def split_and_export(audio_path, regions, output_dir, mode="prepositioned",
                     progress_callback=None):
    """
    Split audio into clips based on detected regions and export as WAV files.

    mode:
      "prepositioned" — each clip has leading silence so it plays at the
                        correct timeline position when dropped at 0:00
      "raw"           — just the audio clips, named with timestamps
    """
    audio = AudioSegment.from_file(audio_path)
    stem_name = clean_stem_name(audio_path)
    stem_dir = os.path.join(output_dir, stem_name)
    os.makedirs(stem_dir, exist_ok=True)

    sample_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width

    output_files = []
    for i, (start_ms, end_ms) in enumerate(regions, 1):
        clip = audio[start_ms:end_ms]
        ts = format_timestamp(start_ms)
        filename = f"{stem_name}_{i:03d}_{ts}.wav"
        filepath = os.path.join(stem_dir, filename)

        if mode == "prepositioned":
            # Prepend silence matching the clip's original position
            leading_silence = AudioSegment.silent(
                duration=start_ms,
                frame_rate=sample_rate,
            )
            # Match channel count and sample width
            if leading_silence.channels != channels:
                if channels == 1:
                    leading_silence = leading_silence.set_channels(1)
                else:
                    leading_silence = leading_silence.set_channels(channels)
            leading_silence = leading_silence.set_sample_width(sample_width)

            full_clip = leading_silence + clip
            full_clip.export(filepath, format="wav")
        else:
            clip.export(filepath, format="wav")

        output_files.append({
            "path": filepath,
            "filename": filename,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
        })

        if progress_callback:
            progress_callback(i, len(regions))

    return output_files


def merge_and_export(audio_path, regions, output_dir, progress_callback=None):
    """
    Export one full-length WAV per stem with clips at their correct positions
    and silence in the gaps. Ideal for dragging into Ableton / FL Studio —
    one file per stem, one drag, each becomes its own track.
    """
    audio = AudioSegment.from_file(audio_path)
    stem_name = clean_stem_name(audio_path)
    total_ms = len(audio)

    sample_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width

    # Build a silent canvas the full length of the original
    canvas = AudioSegment.silent(
        duration=total_ms,
        frame_rate=sample_rate,
    )
    if canvas.channels != channels:
        canvas = canvas.set_channels(channels)
    canvas = canvas.set_sample_width(sample_width)

    # Overlay each detected region onto the canvas
    for i, (start_ms, end_ms) in enumerate(regions, 1):
        clip = audio[start_ms:end_ms]
        canvas = canvas.overlay(clip, position=start_ms)
        if progress_callback:
            progress_callback(i, len(regions))

    filename = f"{stem_name}.wav"
    filepath = os.path.join(output_dir, filename)
    canvas.export(filepath, format="wav")

    return [{
        "path": filepath,
        "filename": filename,
        "start_ms": 0,
        "end_ms": total_ms,
        "duration_ms": total_ms,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# REAPER .RPP EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_rpp(all_results, output_dir, sample_rate=44100):
    """
    Generate a REAPER project file (.rpp) with all clips placed on the timeline.

    all_results: dict mapping stem_name -> list of clip info dicts
    """
    rpp_path = os.path.join(output_dir, "StemSlicer_Session.rpp")

    lines = []
    lines.append("<REAPER_PROJECT 0.1 \"7.0\" 1")
    lines.append(f"  SAMPLERATE {sample_rate} 0 0")
    lines.append("  TEMPO 120 4 4")
    lines.append("")

    track_num = 0
    for stem_name, clips in all_results.items():
        track_num += 1
        cat = detect_instrument(stem_name)
        lines.append(f"  <TRACK {{{_guid()}}}")
        lines.append(f'    NAME "{stem_name}"')
        lines.append(f"    TRACKID {track_num}")
        lines.append("    VOLPAN 1 0 -1 -1 1")
        lines.append("    MUTESOLO 0 0 0")
        lines.append("    IPHASE 0")
        lines.append("    ISBUS 0 0")
        lines.append("    FX 1")
        lines.append(f"    PEAKCOL {cat['reaper_col']}")
        lines.append("    MAINSEND 1 0")

        for clip in clips:
            start_sec = clip["start_ms"] / 1000.0
            dur_sec = clip["duration_ms"] / 1000.0
            # Use the raw clip path (not pre-positioned)
            clip_path = clip["path"]

            lines.append(f"    <ITEM")
            lines.append(f"      POSITION {start_sec:.6f}")
            lines.append(f"      LENGTH {dur_sec:.6f}")
            lines.append(f'      NAME "{clip["filename"]}"')
            lines.append(f"      GUID {{{_guid()}}}")
            lines.append(f"      FADEIN 1 0.01 0 1 0 0 0")
            lines.append(f"      FADEOUT 1 0.01 0 1 0 0 0")
            lines.append(f"      VOLPAN 1 0 1 -1")
            lines.append(f"      SOFFS 0")
            lines.append(f"      PLAYRATE 1 1 0 -1 0 0.0025")
            lines.append(f"      CHANMODE 0")
            lines.append(f"      <SOURCE WAVE")
            lines.append(f'        FILE "{clip_path}"')
            lines.append(f"      >")
            lines.append(f"    >")

        lines.append("  >")
        lines.append("")

    lines.append(">")

    with open(rpp_path, "w") as f:
        f.write("\n".join(lines))

    return rpp_path


def _guid():
    """Generate a simple pseudo-GUID for REAPER project items."""
    import uuid
    return str(uuid.uuid4()).upper()


# ─────────────────────────────────────────────────────────────────────────────
# AAF EXPORT (Pro Tools, Logic, Cubase, Nuendo, REAPER)
# ─────────────────────────────────────────────────────────────────────────────

def _check_pyaaf2():
    """Check if pyaaf2 is available. Returns True if importable."""
    try:
        import aaf2  # noqa: F401
        return True
    except ImportError:
        return False


def generate_aaf(all_results, output_dir, sample_rate=44100):
    """
    Generate an AAF file with all clips placed on the timeline.
    Uses external WAV references (not embedded) to keep file small.

    all_results: dict mapping stem_name -> list of clip info dicts
    Returns path to the AAF file, or None on failure.
    """
    import aaf2

    aaf_path = os.path.join(output_dir, "StemSlicer_Session.aaf")

    with aaf2.open(aaf_path, "w") as f:
        # Find the longest clip end time to set composition length
        max_end_ms = 0
        for clips in all_results.values():
            for clip in clips:
                max_end_ms = max(max_end_ms, clip["end_ms"])

        # Main composition
        main_comp = f.create.MasterMob("StemSlicer_Session")
        f.content.mobs.append(main_comp)

        edit_rate = aaf2.rational.AAFRational(sample_rate, 1)

        for stem_name, clips in all_results.items():
            # Create a SourceMob for each clip file
            for clip in clips:
                clip_path = clip["path"]
                clip_filename = clip["filename"]

                # Create source mob (tape/file mob) for the WAV reference
                src_mob = f.create.SourceMob()
                src_mob.name = clip_filename
                f.content.mobs.append(src_mob)

                # Describe the essence — WAV file descriptor
                desc = f.create.WAVEDescriptor()
                desc.name = clip_filename
                # Store as relative path from the AAF location
                rel_path = os.path.relpath(clip_path, output_dir)
                loc = f.create.NetworkLocator()
                loc['URLString'].value = rel_path
                desc['Locator'].append(loc)
                desc['SampleRate'].value = edit_rate
                desc['Length'].value = int(clip["duration_ms"] * sample_rate / 1000)
                desc['AudioSamplingRate'].value = edit_rate
                desc['Channels'].value = 1
                desc['QuantizationBits'].value = 16
                src_mob.descriptor = desc

                # Source mob needs a timeline slot
                duration_samples = int(clip["duration_ms"] * sample_rate / 1000)
                src_slot = src_mob.create_empty_sequence_slot(edit_rate, media_kind="sound")
                src_filler = f.create.Filler("sound", duration_samples)
                src_slot.segment = src_filler

                # Add slot to main composition
                start_samples = int(clip["start_ms"] * sample_rate / 1000)

                slot = main_comp.create_empty_sequence_slot(edit_rate, media_kind="sound")
                seq = f.create.Sequence("sound")

                # Leading filler (silence before clip starts)
                if start_samples > 0:
                    filler = f.create.Filler("sound", start_samples)
                    seq.components.append(filler)

                # Source clip referencing the WAV
                src_clip = src_mob.create_source_clip(slot_id=1, length=duration_samples)
                seq.components.append(src_clip)

                slot.segment = seq
                slot.name = f"{stem_name} - {clip_filename}"

    return aaf_path


# ─────────────────────────────────────────────────────────────────────────────
# DAWPROJECT EXPORT (Bitwig Studio, Studio One)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dawproject(all_results, output_dir, sample_rate=44100):
    """
    Generate a .dawproject file (ZIP containing project.xml + audio).
    Uses stdlib only (xml.etree.ElementTree + zipfile).

    all_results: dict mapping stem_name -> list of clip info dicts
    Returns path to the .dawproject file.
    """
    import xml.etree.ElementTree as ET

    dawproject_path = os.path.join(output_dir, "StemSlicer_Session.dawproject")

    # Build project.xml
    project = ET.Element("Project", version="1.0")
    ET.SubElement(project, "Application", name="StemSlicer", version="2.0")

    # Arrangement
    arrangement = ET.SubElement(project, "Arrangement")
    tempo_node = ET.SubElement(arrangement, "Tempo")
    tempo_node.set("value", "120")
    time_sig = ET.SubElement(arrangement, "TimeSignature", numerator="4", denominator="4")

    tracks_node = ET.SubElement(arrangement, "Tracks")

    # Collect audio files to embed in the ZIP
    audio_entries = {}  # zip_path -> filesystem_path

    track_idx = 0
    for stem_name, clips in all_results.items():
        cat = detect_instrument(stem_name)
        color_hex = cat["color"]  # e.g. "#E74C3C"

        track = ET.SubElement(tracks_node, "Track",
                              id=f"track-{track_idx}",
                              name=stem_name,
                              color=color_hex)
        channel = ET.SubElement(track, "Channel")
        ET.SubElement(channel, "Volume", value="1.0")
        ET.SubElement(channel, "Pan", value="0.0")

        lanes = ET.SubElement(track, "Lanes")
        lane = ET.SubElement(lanes, "Lane")

        for clip_idx, clip in enumerate(clips):
            clip_filename = clip["filename"]
            # Path inside ZIP
            zip_audio_path = f"audio/{stem_name}/{clip_filename}"
            audio_entries[zip_audio_path] = clip["path"]

            start_sec = clip["start_ms"] / 1000.0
            dur_sec = clip["duration_ms"] / 1000.0

            clip_el = ET.SubElement(lane, "Clip",
                                    time=f"{start_sec:.6f}",
                                    duration=f"{dur_sec:.6f}",
                                    name=clip_filename)
            ET.SubElement(clip_el, "Audio",
                          file=zip_audio_path,
                          channels="2",
                          sampleRate=str(sample_rate))

        track_idx += 1

    # Build metadata.xml
    metadata = ET.Element("MetaData")
    ET.SubElement(metadata, "Title").text = "StemSlicer_Session"
    ET.SubElement(metadata, "Artist").text = ""
    ET.SubElement(metadata, "Application").text = "StemSlicer 2.0"

    # Write ZIP
    with zipfile.ZipFile(dawproject_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # project.xml
        project_xml = ET.tostring(project, encoding="unicode", xml_declaration=True)
        zf.writestr("project.xml", project_xml)

        # metadata.xml
        meta_xml = ET.tostring(metadata, encoding="unicode", xml_declaration=True)
        zf.writestr("metadata.xml", meta_xml)

        # Audio files
        for zip_path, fs_path in audio_entries.items():
            zf.write(fs_path, zip_path)

    return dawproject_path


# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_waveform_peaks_stereo(audio_path, num_points=800):
    """
    Get normalized peak values for drawing a stereo waveform.

    Returns (peaks_left, peaks_right, is_stereo, duration_ms).
      peaks_left / peaks_right: list of (min_peak, max_peak) tuples normalized to -1..1
      is_stereo: True if the source has 2+ channels
      duration_ms: total duration in milliseconds
    """
    audio = AudioSegment.from_file(audio_path)
    samples = audio.get_array_of_samples()
    is_stereo = audio.channels >= 2
    duration_ms = len(audio)

    if is_stereo:
        left_samples = samples[::2]
        right_samples = samples[1::2]
    else:
        left_samples = samples
        right_samples = samples

    total_samples = len(left_samples)
    if total_samples == 0:
        empty = [(0.0, 0.0)] * num_points
        return empty, empty, is_stereo, duration_ms

    chunk_size = max(1, total_samples // num_points)
    max_val = float(2 ** (audio.sample_width * 8 - 1))

    def extract_peaks(samps):
        peaks = []
        for i in range(num_points):
            start = i * chunk_size
            end = min(start + chunk_size, len(samps))
            if start >= len(samps):
                peaks.append((0.0, 0.0))
                continue
            chunk = samps[start:end]
            mn = min(chunk) / max_val
            mx = max(chunk) / max_val
            peaks.append((mn, mx))
        return peaks

    peaks_l = extract_peaks(left_samples)
    peaks_r = extract_peaks(right_samples) if is_stereo else extract_peaks(left_samples)

    return peaks_l, peaks_r, is_stereo, duration_ms


# ─────────────────────────────────────────────────────────────────────────────
# GUI — PySide6
# ─────────────────────────────────────────────────────────────────────────────

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QListWidget, QListWidgetItem,
    QProgressBar, QRadioButton, QCheckBox, QFrame, QFileDialog,
    QMessageBox, QSizePolicy, QGraphicsDropShadowEffect, QButtonGroup,
    QAbstractItemView, QMenu, QDialog, QLineEdit,
)
from PySide6.QtCore import (
    Qt, Signal, QObject, QThread, QTimer, QMimeData, QUrl, QSize,
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QLinearGradient, QPen, QBrush,
    QFont, QFontDatabase, QDragEnterEvent, QDropEvent, QCursor,
)

from theme import Colors, Fonts, get_stylesheet


# ─────────────────────────────────────────────────────────────────────────────
# License Dialog
# ─────────────────────────────────────────────────────────────────────────────

class LicenseDialog(QDialog):
    """License activation dialog — shown on first launch before the main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("StemSlicer — Activate License")
        self.setFixedSize(400, 320)
        self.setModal(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Gradient background matching app theme
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor(Colors.SURFACE_TOP))
        grad.setColorAt(1.0, QColor(Colors.BG))
        painter.fillRect(self.rect(), grad)
        painter.end()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(8)

        # Accent bar at top
        accent = QFrame()
        accent.setFixedHeight(3)
        accent.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {Colors.ACCENT}, stop:1 {Colors.ACCENT_DIM});"
        )
        layout.addWidget(accent)

        layout.addSpacing(8)

        # Title
        title = QLabel("StemSlicer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: #FFFFFF; background: transparent;"
        )
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Enter your license to get started")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: 11px; color: {Colors.TEXT_MUTED_SOLID}; background: transparent;"
        )
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # Email field
        email_label = QLabel("Email")
        email_label.setStyleSheet(
            f"font-size: 10px; font-weight: 600; color: {Colors.TEXT_MUTED_SOLID}; "
            f"text-transform: uppercase; letter-spacing: 1px; background: transparent;"
        )
        layout.addWidget(email_label)

        self._email_input = QLineEdit()
        self._email_input.setPlaceholderText("your@email.com")
        self._email_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.WAVEFORM_BG};
                border: 1px solid {Colors.BORDER_SOLID};
                border-radius: 6px;
                padding: 8px 12px;
                color: {Colors.TEXT_SOLID};
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT};
            }}
        """)
        layout.addWidget(self._email_input)

        layout.addSpacing(4)

        # License key field
        key_label = QLabel("License Key")
        key_label.setStyleSheet(
            f"font-size: 10px; font-weight: 600; color: {Colors.TEXT_MUTED_SOLID}; "
            f"text-transform: uppercase; letter-spacing: 1px; background: transparent;"
        )
        layout.addWidget(key_label)

        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("SS-XXXXXXXX")
        self._key_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.WAVEFORM_BG};
                border: 1px solid {Colors.BORDER_SOLID};
                border-radius: 6px;
                padding: 8px 12px;
                color: {Colors.TEXT_SOLID};
                font-size: 13px;
                font-family: monospace;
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT};
            }}
        """)
        layout.addWidget(self._key_input)

        # Error message (hidden by default)
        self._error_label = QLabel("")
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setStyleSheet(
            f"font-size: 11px; color: {Colors.DANGER}; background: transparent;"
        )
        self._error_label.hide()
        layout.addWidget(self._error_label)

        layout.addSpacing(8)

        # Activate button (same style as PROCESS STEMS)
        self._activate_btn = QPushButton("ACTIVATE")
        self._activate_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._activate_btn.setFixedHeight(36)
        self._activate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.ACCENT}, stop:1 {Colors.ACCENT_DIM});
                color: #0D0F14;
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1AFFDE, stop:1 {Colors.ACCENT});
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_DIM};
            }}
        """)
        self._activate_btn.clicked.connect(self._on_activate)
        layout.addWidget(self._activate_btn)

        # Allow Enter key to activate
        self._key_input.returnPressed.connect(self._on_activate)
        self._email_input.returnPressed.connect(lambda: self._key_input.setFocus())

        layout.addStretch()

    def _on_activate(self):
        email = self._email_input.text().strip()
        key = self._key_input.text().strip()

        if not email:
            self._show_error("Please enter your email address")
            return
        if not key:
            self._show_error("Please enter your license key")
            return

        if _validate_license(email, key):
            _save_license(email, key)
            self.accept()
        else:
            self._show_error("Invalid license key. Please check your email and key.")

    def _show_error(self, msg):
        self._error_label.setText(msg)
        self._error_label.show()

    @classmethod
    def activate(cls, parent=None):
        """Show the dialog. Returns True if license was activated, False if cancelled."""
        dlg = cls(parent)
        dlg._build_ui()
        return dlg.exec() == QDialog.DialogCode.Accepted


# ─────────────────────────────────────────────────────────────────────────────
# Custom Widgets
# ─────────────────────────────────────────────────────────────────────────────

class AccentBar(QFrame):
    """3px teal gradient bar at the top of the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(3)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QColor(Colors.ACCENT))
        grad.setColorAt(1.0, QColor(Colors.ACCENT_DIM))
        painter.fillRect(self.rect(), grad)
        painter.end()


class GlowPanel(QFrame):
    """Panel with gradient background, rounded corners, and subtle shadow."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = title

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(6)

        if title:
            heading = QLabel(title)
            heading.setObjectName("panelHeading")
            self._layout.addWidget(heading)

    def contentLayout(self):
        return self._layout

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Gradient fill
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor(Colors.SURFACE_TOP))
        grad.setColorAt(1.0, QColor(Colors.SURFACE))
        path = QPainterPath()
        path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, 10, 10)
        painter.fillPath(path, grad)

        # Subtle border
        painter.setPen(QPen(QColor(255, 255, 255, 13), 1))
        painter.drawPath(path)
        painter.end()


class SliderGroup(QWidget):
    """Label + QSlider + value readout + hint text."""

    valueChanged = Signal(int)

    def __init__(self, label, min_val, max_val, default, suffix="",
                 hint="", parent=None):
        super().__init__(parent)
        self._suffix = suffix
        self._label_text = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Top row: label + value
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {Colors.TEXT_SOLID}; font-size: {Fonts.BODY_SIZE}px;")
        top_row.addWidget(self._label)

        top_row.addStretch()

        self._value_label = QLabel(f"{default}{suffix}")
        self._value_label.setObjectName("sliderValue")
        top_row.addWidget(self._value_label)

        layout.addLayout(top_row)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(min_val)
        self._slider.setMaximum(max_val)
        self._slider.setValue(default)
        self._slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._slider.valueChanged.connect(self._on_change)
        layout.addWidget(self._slider)

        # Hint
        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("sliderHint")
            layout.addWidget(hint_label)

    def _on_change(self, val):
        self._value_label.setText(f"{val}{self._suffix}")
        self.valueChanged.emit(val)

    def value(self):
        return self._slider.value()


class FileListWidget(QListWidget):
    """Styled file list with drag-and-drop from Finder."""

    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setMinimumHeight(50)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    paths.append(path)
            if paths:
                self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class PlacesWidget(QListWidget):
    """Pinned folder list (Ableton-style Places) with drag-drop and right-click remove."""

    placeClicked = Signal(str)   # emits folder path when a place is clicked
    placesChanged = Signal()     # emitted when places are added/removed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("placesWidget")
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setFixedWidth(180)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemClicked.connect(self._on_item_clicked)
        self._places = []  # list of {"name": str, "path": str}

    def set_places(self, places):
        self._places = list(places)
        self._rebuild()

    def get_places(self):
        return list(self._places)

    def add_place(self, folder_path):
        folder_path = str(folder_path)
        for p in self._places:
            if p["path"] == folder_path:
                return
        name = Path(folder_path).name
        self._places.append({"name": name, "path": folder_path})
        self._rebuild()
        self.placesChanged.emit()

    def remove_place(self, index):
        if 0 <= index < len(self._places):
            self._places.pop(index)
            self._rebuild()
            self.placesChanged.emit()

    def _rebuild(self):
        self.clear()
        # First row is always the "+ Place" action
        add_item = QListWidgetItem("+ Place")
        add_item.setForeground(QColor(Colors.ACCENT))
        add_item.setToolTip("Pin a folder for quick access")
        self.addItem(add_item)
        for place in self._places:
            exists = Path(place["path"]).is_dir()
            label = f"\U0001F4C1 {place['name']}"
            item = QListWidgetItem(label)
            item.setToolTip(place["path"])
            if not exists:
                item.setForeground(QColor(Colors.TEXT_MUTED_SOLID))
                item.setToolTip(f"{place['path']}\n(folder not found)")
            self.addItem(item)

    def _on_item_clicked(self, item):
        row = self.row(item)
        if row == 0:
            # "+ Place" row — open folder picker
            folder = QFileDialog.getExistingDirectory(self, "Pin a folder to Places")
            if folder:
                self.add_place(folder)
            return
        place_idx = row - 1  # offset for the "+ Place" row
        if place_idx < 0 or place_idx >= len(self._places):
            return
        place = self._places[place_idx]
        folder = Path(place["path"])
        if not folder.is_dir():
            QMessageBox.warning(
                self, "Folder Not Found",
                f"The folder is not available:\n{place['path']}\n\n"
                "It may be on a disconnected drive."
            )
            return
        self.placeClicked.emit(place["path"])

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        row = self.row(item)
        if row == 0:
            return  # no context menu on "+ Place" row
        place_idx = row - 1
        if place_idx < 0 or place_idx >= len(self._places):
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove")
        open_action = menu.addAction("Open in Finder")
        action = menu.exec(self.mapToGlobal(pos))
        if action == remove_action:
            self.remove_place(place_idx)
        elif action == open_action:
            folder = self._places[place_idx]["path"]
            if platform.system() == "Windows":
                os.startfile(folder)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and Path(path).is_dir():
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and Path(path).is_dir():
                    self.add_place(path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class _DropOverlay(QWidget):
    """Translucent overlay shown when dragging files over the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark background
        painter.fillRect(self.rect(), QColor(13, 15, 20, 200))

        # Dashed teal border inset
        pen = QPen(QColor(Colors.ACCENT), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        margin = 20
        painter.drawRoundedRect(margin, margin,
                                self.width() - margin * 2,
                                self.height() - margin * 2, 12, 12)

        # Icon + text
        painter.setPen(Qt.PenStyle.NoPen)
        cx = self.width() / 2
        cy = self.height() / 2 - 16

        # Down-arrow icon (simple triangle)
        arrow = QPainterPath()
        arrow.moveTo(cx - 16, cy - 8)
        arrow.lineTo(cx + 16, cy - 8)
        arrow.lineTo(cx, cy + 12)
        arrow.closeSubpath()
        painter.setBrush(QColor(Colors.ACCENT))
        painter.drawPath(arrow)

        # Text
        painter.setPen(QColor(Colors.ACCENT))
        f = painter.font()
        f.setPixelSize(16)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(self.rect().adjusted(0, 30, 0, 0),
                         Qt.AlignmentFlag.AlignCenter,
                         "Drop audio files or folders here")
        painter.end()


class ZoomBar(QWidget):
    """Draggable/resizable zoom bar — shows visible range of the waveform."""

    viewChanged = Signal(float, float)  # view_start, view_end (0..1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        self._view_start = 0.0
        self._view_end = 1.0
        self._dragging = None  # None, "move", "left", "right"
        self._drag_origin_x = 0
        self._drag_origin_start = 0.0
        self._drag_origin_end = 1.0
        self._has_data = False
        self._mini_peaks = None  # mini waveform for overview

    def set_has_data(self, has_data, peaks=None):
        self._has_data = has_data
        self._mini_peaks = peaks
        if not has_data:
            self._view_start = 0.0
            self._view_end = 1.0
        self.update()

    def set_view(self, start, end):
        self._view_start = max(0.0, min(start, 1.0))
        self._view_end = max(0.0, min(end, 1.0))
        if self._view_end - self._view_start < 0.02:
            self._view_end = self._view_start + 0.02
        self.update()

    def view_start(self):
        return self._view_start

    def view_end(self):
        return self._view_end

    def reset_zoom(self):
        self._view_start = 0.0
        self._view_end = 1.0
        self.viewChanged.emit(0.0, 1.0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(self.rect(), QColor(Colors.WAVEFORM_BG))

        if not self._has_data:
            painter.end()
            return

        # Mini waveform overview
        if self._mini_peaks:
            n = len(self._mini_peaks)
            mid_y = h / 2.0
            usable = mid_y - 1
            x_step = w / max(n - 1, 1)
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
            for i in range(n):
                x = int(i * x_step)
                mn, mx = self._mini_peaks[i]
                y1 = int(mid_y - mx * usable)
                y2 = int(mid_y - mn * usable)
                painter.drawLine(x, y1, x, y2)

        # Dim areas outside the handle
        dim = QColor(0, 0, 0, 120)
        hx1 = int(self._view_start * w)
        hx2 = int(self._view_end * w)
        if hx1 > 0:
            painter.fillRect(0, 0, hx1, h, dim)
        if hx2 < w:
            painter.fillRect(hx2, 0, w - hx2, h, dim)

        # Handle
        handle_path = QPainterPath()
        handle_path.addRoundedRect(float(hx1), 0.5, float(hx2 - hx1), h - 1, 3, 3)
        painter.setPen(QPen(QColor(Colors.ACCENT), 1))
        painter.setBrush(QColor(0, 229, 204, 25))
        painter.drawPath(handle_path)

        # Resize grip lines on edges
        grip_pen = QPen(QColor(Colors.ACCENT), 1)
        painter.setPen(grip_pen)
        for gx in [hx1 + 3, hx2 - 3]:
            painter.drawLine(gx, 4, gx, h - 4)

        painter.end()

    def mousePressEvent(self, event):
        if not self._has_data or event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        w = self.width()
        hx1 = self._view_start * w
        hx2 = self._view_end * w
        edge = 8  # grab zone for resize

        self._drag_origin_x = x
        self._drag_origin_start = self._view_start
        self._drag_origin_end = self._view_end

        if abs(x - hx1) < edge:
            self._dragging = "left"
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        elif abs(x - hx2) < edge:
            self._dragging = "right"
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        elif hx1 <= x <= hx2:
            self._dragging = "move"
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        else:
            # Click outside handle — center handle on click position
            span = self._view_end - self._view_start
            center = x / w
            new_start = center - span / 2
            new_end = center + span / 2
            if new_start < 0:
                new_start = 0; new_end = span
            if new_end > 1:
                new_end = 1; new_start = 1 - span
            self._view_start = new_start
            self._view_end = new_end
            self._drag_origin_start = new_start
            self._drag_origin_end = new_end
            self._dragging = "move"
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            self.viewChanged.emit(self._view_start, self._view_end)
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            # Just update cursor
            x = event.position().x()
            w = self.width()
            hx1 = self._view_start * w
            hx2 = self._view_end * w
            if abs(x - hx1) < 8 or abs(x - hx2) < 8:
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
            elif hx1 <= x <= hx2:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            return

        dx = (event.position().x() - self._drag_origin_x) / self.width()
        min_span = 0.02

        if self._dragging == "move":
            span = self._drag_origin_end - self._drag_origin_start
            new_start = self._drag_origin_start + dx
            new_end = self._drag_origin_end + dx
            if new_start < 0:
                new_start = 0; new_end = span
            if new_end > 1:
                new_end = 1; new_start = 1 - span
            self._view_start = new_start
            self._view_end = new_end
        elif self._dragging == "left":
            new_start = self._drag_origin_start + dx
            new_start = max(0.0, min(new_start, self._view_end - min_span))
            self._view_start = new_start
        elif self._dragging == "right":
            new_end = self._drag_origin_end + dx
            new_end = max(self._view_start + min_span, min(new_end, 1.0))
            self._view_end = new_end

        self.viewChanged.emit(self._view_start, self._view_end)
        self.update()

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom."""
        if self._has_data:
            self.reset_zoom()


class WaveformWidget(QWidget):
    """Antialiased stereo waveform display with region overlays and zoom."""

    zoomChanged = Signal(float, float)  # view_start, view_end

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._peaks_left = None
        self._peaks_right = None
        self._is_stereo = False
        self._duration_ms = 0
        self._regions = []

        # Zoom state (0..1 range of the full waveform)
        self._view_start = 0.0
        self._view_end = 1.0

        # Cached paths — rebuilt on resize, new data, or zoom change
        self._path_left_fill = None
        self._path_right_fill = None
        self._cache_width = 0
        self._cache_height = 0
        self._cache_vs = -1.0
        self._cache_ve = -1.0

    def set_waveform(self, peaks_left, peaks_right, is_stereo, duration_ms):
        self._peaks_left = peaks_left
        self._peaks_right = peaks_right
        self._is_stereo = is_stereo
        self._duration_ms = duration_ms
        self._view_start = 0.0
        self._view_end = 1.0
        self._invalidate_cache()
        self.update()

    def set_regions(self, regions):
        self._regions = regions
        self.update()

    def set_view(self, start, end):
        self._view_start = start
        self._view_end = end
        self._invalidate_cache()
        self.update()

    def clear(self):
        self._peaks_left = None
        self._peaks_right = None
        self._regions = []
        self._view_start = 0.0
        self._view_end = 1.0
        self._invalidate_cache()
        self.update()

    def _invalidate_cache(self):
        self._path_left_fill = None
        self._path_right_fill = None
        self._cache_width = 0
        self._cache_height = 0

    def _visible_peaks(self, peaks):
        """Return the slice of peaks visible in the current view."""
        if peaks is None:
            return None
        n = len(peaks)
        i_start = int(self._view_start * n)
        i_end = int(self._view_end * n)
        i_start = max(0, min(i_start, n))
        i_end = max(i_start + 1, min(i_end, n))
        return peaks[i_start:i_end]

    def _build_paths(self, w, h):
        if self._peaks_left is None:
            return

        vis_l = self._visible_peaks(self._peaks_left)
        vis_r = self._visible_peaks(self._peaks_right) if self._is_stereo else None

        if self._is_stereo:
            half_h = h / 2.0
            self._path_left_fill = self._make_waveform_path(vis_l, w, 0, half_h, margin=4)
            self._path_right_fill = self._make_waveform_path(vis_r, w, half_h, half_h, margin=4)
        else:
            self._path_left_fill = self._make_waveform_path(vis_l, w, 0, h, margin=6)
            self._path_right_fill = None

        self._cache_width = w
        self._cache_height = h
        self._cache_vs = self._view_start
        self._cache_ve = self._view_end

    def _make_waveform_path(self, peaks, w, y_offset, h, margin=4):
        """Build a filled polygon path for a waveform channel."""
        if peaks is None:
            return None
        n = len(peaks)
        if n == 0:
            return None

        mid_y = y_offset + h / 2.0
        usable_h = (h / 2.0) - margin

        path = QPainterPath()
        x_step = w / max(n - 1, 1)

        first_y = mid_y - peaks[0][1] * usable_h
        path.moveTo(0, first_y)
        for i in range(1, n):
            x = i * x_step
            y = mid_y - peaks[i][1] * usable_h
            path.lineTo(x, y)

        for i in range(n - 1, -1, -1):
            x = i * x_step
            y = mid_y - peaks[i][0] * usable_h
            path.lineTo(x, y)

        path.closeSubpath()
        return path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), QColor(Colors.WAVEFORM_BG))

        if self._peaks_left is None:
            painter.setPen(QColor(Colors.TEXT_MUTED_SOLID))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Select a file to preview waveform")
            painter.end()
            return

        # Rebuild cached paths if needed
        if (self._cache_width != w or self._cache_height != h
                or self._cache_vs != self._view_start
                or self._cache_ve != self._view_end):
            self._build_paths(w, h)

        # Draw region overlays mapped to current view
        if self._duration_ms > 0:
            view_start_ms = self._view_start * self._duration_ms
            view_span_ms = (self._view_end - self._view_start) * self._duration_ms
            if view_span_ms > 0:
                for start_ms, end_ms in self._regions:
                    x1 = int((start_ms - view_start_ms) / view_span_ms * w)
                    x2 = int((end_ms - view_start_ms) / view_span_ms * w)
                    if x2 < 0 or x1 > w:
                        continue
                    x1 = max(0, x1)
                    x2 = min(w, x2)
                    painter.fillRect(x1, 0, x2 - x1, h, QColor(0, 229, 204, 20))
                    pen = QPen(QColor(Colors.ACCENT), 1, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    if 0 <= x1 <= w:
                        painter.drawLine(x1, 0, x1, h)
                    if 0 <= x2 <= w:
                        painter.drawLine(x2, 0, x2, h)

        # Center line(s)
        center_pen = QPen(QColor(255, 255, 255, 25), 1)
        painter.setPen(center_pen)
        if self._is_stereo:
            half_h = h / 2.0
            painter.drawLine(0, int(half_h / 2), w, int(half_h / 2))
            painter.drawLine(0, int(half_h + half_h / 2), w, int(half_h + half_h / 2))
            painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
            painter.drawLine(0, int(half_h), w, int(half_h))
        else:
            painter.drawLine(0, h // 2, w, h // 2)

        # Draw waveform
        if self._path_left_fill:
            grad_l = self._make_wave_gradient(0, h / 2 if self._is_stereo else h)
            painter.setBrush(QBrush(grad_l))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(self._path_left_fill)

        if self._path_right_fill:
            grad_r = self._make_wave_gradient(h / 2, h / 2)
            painter.setBrush(QBrush(grad_r))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(self._path_right_fill)

        # Channel labels
        if self._is_stereo:
            painter.setPen(QColor(255, 255, 255, 60))
            f = painter.font()
            f.setPixelSize(9)
            painter.setFont(f)
            painter.drawText(6, 14, "L")
            painter.drawText(6, int(h / 2) + 14, "R")

        painter.end()

    def _make_wave_gradient(self, y_offset, section_height):
        grad = QLinearGradient(0, y_offset, 0, y_offset + section_height)
        grad.setColorAt(0.0, QColor(0, 229, 204, 30))
        grad.setColorAt(0.35, QColor(0, 229, 204, 130))
        grad.setColorAt(0.5, QColor(0, 229, 204, 180))
        grad.setColorAt(0.65, QColor(0, 229, 204, 130))
        grad.setColorAt(1.0, QColor(0, 229, 204, 30))
        return grad

    def wheelEvent(self, event):
        """Scroll wheel / trackpad: vertical = zoom, horizontal = pan."""
        if self._peaks_left is None:
            return

        dx = event.angleDelta().x()
        dy = event.angleDelta().y()

        # Horizontal scroll (two-finger swipe left/right) → pan
        if abs(dx) > abs(dy) and dx != 0:
            span = self._view_end - self._view_start
            # Scroll amount proportional to visible span
            shift = -dx / 1200.0 * span
            new_start = self._view_start + shift
            new_end = self._view_end + shift

            # Clamp
            if new_start < 0:
                new_start = 0; new_end = span
            if new_end > 1:
                new_end = 1; new_start = 1 - span

            self._view_start = new_start
            self._view_end = new_end
            self._invalidate_cache()
            self.update()
            self.zoomChanged.emit(self._view_start, self._view_end)
            return

        # Vertical scroll → zoom centered on mouse position
        if dy == 0:
            return

        mouse_frac = event.position().x() / self.width()
        anchor = self._view_start + mouse_frac * (self._view_end - self._view_start)

        zoom_factor = 0.85 if dy > 0 else 1.0 / 0.85
        span = self._view_end - self._view_start
        new_span = max(0.02, min(1.0, span * zoom_factor))

        # Keep anchor point at same mouse position
        new_start = anchor - mouse_frac * new_span
        new_end = new_start + new_span

        # Clamp
        if new_start < 0:
            new_start = 0; new_end = new_span
        if new_end > 1:
            new_end = 1; new_start = 1 - new_span

        self._view_start = new_start
        self._view_end = new_end
        self._invalidate_cache()
        self.update()
        self.zoomChanged.emit(self._view_start, self._view_end)

    def resizeEvent(self, event):
        self._invalidate_cache()
        super().resizeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Worker (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class AudioWorker(QObject):
    """Handles all audio processing off the main thread."""

    waveformReady = Signal(list, list, bool, int)  # peaks_l, peaks_r, is_stereo, duration_ms
    regionsReady = Signal(list)
    progressUpdate = Signal(float, str)  # 0..1, status text
    processingDone = Signal(str, str)  # message, output_dir
    processingError = Signal(str)

    def load_waveform(self, audio_path, num_points=1200):
        try:
            pl, pr, stereo, dur = get_waveform_peaks_stereo(audio_path, num_points)
            self.waveformReady.emit(pl, pr, stereo, dur)
        except Exception as e:
            self.processingError.emit(f"Waveform error: {e}")

    def detect(self, audio_path, threshold_db, min_silence_ms, padding_ms):
        try:
            regions = detect_regions(audio_path, threshold_db, min_silence_ms, padding_ms)
            self.regionsReady.emit(regions)
        except Exception as e:
            self.regionsReady.emit([])

    def process_all(self, input_files, threshold_db, min_silence_ms, padding_ms,
                    mode, export_formats):
        try:
            total_files = len(input_files)
            all_results = {}
            all_raw_results = {}

            any_daw_export = (export_formats.get("rpp") or
                              export_formats.get("aaf") or
                              export_formats.get("dawproject"))
            want_raw_clips = export_formats.get("raw_clips", False)

            from datetime import datetime
            first_parent = Path(input_files[0]).parent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(str(first_parent), f"StemSlicer_Output_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            if (any_daw_export or want_raw_clips) and mode != "raw":
                raw_folder_name = "Raw_Clips" if want_raw_clips else "_daw_clips"
                raw_output_dir = os.path.join(output_dir, raw_folder_name)
                os.makedirs(raw_output_dir, exist_ok=True)

            for file_idx, audio_path in enumerate(input_files):
                stem_name = clean_stem_name(audio_path)
                self.progressUpdate.emit(
                    file_idx / total_files,
                    f"Analyzing {stem_name}... ({file_idx + 1}/{total_files})"
                )

                regions = detect_regions(audio_path, threshold_db, min_silence_ms, padding_ms)
                if not regions:
                    self.progressUpdate.emit(
                        (file_idx + 0.5) / total_files,
                        f"No audio detected in {stem_name}, skipping..."
                    )
                    continue

                self.progressUpdate.emit(
                    file_idx / total_files,
                    f"{'Merging' if mode == 'merged' else 'Splitting'} {stem_name}: {len(regions)} regions... ({file_idx + 1}/{total_files})"
                )

                def progress_cb(current, total, fi=file_idx):
                    overall = (fi + current / total) / total_files
                    self.progressUpdate.emit(overall, "")

                if mode == "merged":
                    clips = merge_and_export(
                        audio_path, regions, output_dir,
                        progress_callback=progress_cb,
                    )
                else:
                    clips = split_and_export(
                        audio_path, regions, output_dir, mode=mode,
                        progress_callback=progress_cb,
                    )
                all_results[stem_name] = clips

                if any_daw_export or want_raw_clips:
                    if mode == "raw":
                        all_raw_results[stem_name] = clips
                    else:
                        raw_clips = split_and_export(
                            audio_path, regions, raw_output_dir, mode="raw",
                        )
                        all_raw_results[stem_name] = raw_clips

            # Detect sample rate from first input file
            sr = 44100
            if any_daw_export and all_raw_results:
                try:
                    audio = AudioSegment.from_file(input_files[0])
                    sr = audio.frame_rate
                except Exception:
                    pass

            # Generate DAW projects
            exported_projects = []

            if export_formats.get("rpp") and all_raw_results:
                self.progressUpdate.emit(0.92, "Generating REAPER project...")
                generate_rpp(all_raw_results, output_dir, sample_rate=sr)
                exported_projects.append("REAPER")

            if export_formats.get("aaf") and all_raw_results:
                self.progressUpdate.emit(0.94, "Generating AAF project...")
                try:
                    generate_aaf(all_raw_results, output_dir, sample_rate=sr)
                    exported_projects.append("AAF")
                except Exception as e:
                    self.progressUpdate.emit(0.94, f"AAF export failed: {e}")

            if export_formats.get("dawproject") and all_raw_results:
                self.progressUpdate.emit(0.96, "Generating DAWproject...")
                generate_dawproject(all_raw_results, output_dir, sample_rate=sr)
                exported_projects.append("DAWproject")

            total_clips = sum(len(c) for c in all_results.values())
            msg = f"Done! {total_clips} clips from {len(all_results)} stems"
            if exported_projects:
                msg += " + " + ", ".join(exported_projects)

            self.progressUpdate.emit(1.0, msg)
            self.processingDone.emit(msg, output_dir)

        except Exception as e:
            self.processingError.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class StemSlicerApp(QMainWindow):
    # Signals to trigger worker methods from main thread
    _triggerLoadWaveform = Signal(str, int)
    _triggerDetect = Signal(str, int, int, int)
    _triggerProcessAll = Signal(list, int, int, int, str, dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("StemSlicer")
        self.setFixedSize(860, 720)
        self.setAcceptDrops(True)

        self.input_files = []
        self.current_preview_file = None
        self.is_processing = False
        self._drag_active = False

        # Debounce timer for slider changes
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._run_detection)

        # Setup worker thread
        self._worker_thread = QThread(self)
        self._worker = AudioWorker()
        self._worker.moveToThread(self._worker_thread)

        # Connect trigger signals to worker slots
        self._triggerLoadWaveform.connect(self._worker.load_waveform)
        self._triggerDetect.connect(self._worker.detect)
        self._triggerProcessAll.connect(self._worker.process_all)

        # Connect worker result signals
        self._worker.waveformReady.connect(self._on_waveform_ready)
        self._worker.regionsReady.connect(self._on_regions_ready)
        self._worker.progressUpdate.connect(self._on_progress)
        self._worker.processingDone.connect(self._on_done)
        self._worker.processingError.connect(self._on_error)

        self._worker_thread.start()

        self._build_ui()
        self._places_widget.set_places(self._load_places())

    # ── Places config persistence ──

    @staticmethod
    def _places_config_path():
        if platform.system() == "Windows":
            base = Path(os.environ.get("APPDATA", Path.home()))
            return base / "StemSlicer" / "places.json"
        return Path.home() / ".stemslicer" / "places.json"

    def _load_places(self):
        path = self._places_config_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("places", [])
        except Exception:
            return []

    def _save_places(self):
        path = self._places_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"places": self._places_widget.get_places()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Drop overlay (hidden by default, shown during drag)
        self._drop_overlay = _DropOverlay(central)
        self._drop_overlay.hide()

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 0, 16, 12)
        main_layout.setSpacing(6)

        # Accent bar
        accent = AccentBar()
        main_layout.addWidget(accent)

        # Header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 4, 0, 0)

        title = QLabel("StemSlicer")
        title.setObjectName("title")
        header_layout.addWidget(title)

        subtitle = QLabel("Works with any DAW — FL Studio, Ableton, Logic, REAPER, Cubase, Pro Tools")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignBottom)
        header_layout.addWidget(subtitle)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # ── INPUT panel ──
        input_panel = GlowPanel("INPUT")
        inp_layout = input_panel.contentLayout()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        add_files_btn = QPushButton("+ Files")
        add_files_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        add_files_btn.clicked.connect(self._add_files)
        btn_row.addWidget(add_files_btn)

        add_folder_btn = QPushButton("+ Folder")
        add_folder_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        add_folder_btn.clicked.connect(self._add_folder)
        btn_row.addWidget(add_folder_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        clear_btn.clicked.connect(self._clear_files)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()

        self._file_count_label = QLabel("No files")
        self._file_count_label.setObjectName("fileCount")
        btn_row.addWidget(self._file_count_label)

        inp_layout.addLayout(btn_row)

        # Two-column body: Places (left) + File list (right)
        columns = QHBoxLayout()
        columns.setSpacing(8)

        # Left column — Places
        places_col = QVBoxLayout()
        places_col.setSpacing(2)

        places_label = QLabel("PLACES")
        places_label.setObjectName("panelHeading")
        places_col.addWidget(places_label)

        self._places_widget = PlacesWidget()
        self._places_widget.setMaximumHeight(110)
        self._places_widget.placeClicked.connect(self._on_place_clicked)
        self._places_widget.placesChanged.connect(self._save_places)
        places_col.addWidget(self._places_widget)
        columns.addLayout(places_col)

        # Right column — Loaded Files
        files_col = QVBoxLayout()
        files_col.setSpacing(2)
        files_label = QLabel("LOADED FILES")
        files_label.setObjectName("panelHeading")
        files_col.addWidget(files_label)

        self._file_list = FileListWidget()
        self._file_list.setMaximumHeight(110)
        self._file_list.filesDropped.connect(self._on_files_dropped)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        files_col.addWidget(self._file_list)
        columns.addLayout(files_col, stretch=1)

        inp_layout.addLayout(columns)

        input_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(input_panel)

        # ── DETECTION panel ──
        detect_panel = GlowPanel("DETECTION")
        det_layout = detect_panel.contentLayout()

        sliders_row = QHBoxLayout()
        sliders_row.setSpacing(24)

        self._thresh_slider = SliderGroup(
            "Silence Threshold", -70, -15, -40, suffix=" dB",
            hint="Lower = more sensitive"
        )
        self._thresh_slider.valueChanged.connect(self._on_slider_changed)
        sliders_row.addWidget(self._thresh_slider)

        self._silence_slider = SliderGroup(
            "Min Silence", 100, 3000, 500, suffix=" ms",
            hint="Gap length to count as silence"
        )
        self._silence_slider.valueChanged.connect(self._on_slider_changed)
        sliders_row.addWidget(self._silence_slider)

        self._pad_slider = SliderGroup(
            "Padding", 0, 500, 50, suffix=" ms",
            hint="Extra audio kept per clip edge"
        )
        self._pad_slider.valueChanged.connect(self._on_slider_changed)
        sliders_row.addWidget(self._pad_slider)

        det_layout.addLayout(sliders_row)
        detect_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(detect_panel)

        # ── PREVIEW panel ──
        preview_panel = GlowPanel("PREVIEW")
        prev_layout = preview_panel.contentLayout()

        self._preview_info = QLabel("Select a file to preview")
        self._preview_info.setObjectName("previewInfo")
        prev_layout.addWidget(self._preview_info)

        self._zoom_bar = ZoomBar()
        prev_layout.addWidget(self._zoom_bar)

        self._waveform = WaveformWidget()
        prev_layout.addWidget(self._waveform, stretch=1)

        # Wire zoom bar ↔ waveform
        self._zoom_bar.viewChanged.connect(self._waveform.set_view)
        self._waveform.zoomChanged.connect(self._zoom_bar.set_view)

        main_layout.addWidget(preview_panel, stretch=1)

        # ── OUTPUT panel ──
        output_panel = GlowPanel("OUTPUT")
        out_layout = output_panel.contentLayout()

        # Row 1: Always-included output + raw clips extra
        exports_row = QHBoxLayout()
        exports_row.setSpacing(16)

        exports_label = QLabel("Exports:")
        exports_label.setStyleSheet(f"color: {Colors.TEXT_SOLID}; font-weight: 600;")
        exports_label.setFixedWidth(90)
        exports_row.addWidget(exports_label)

        always_label = QLabel("One WAV per stem (always included)")
        always_label.setStyleSheet(f"color: {Colors.TEXT_DIM};")
        exports_row.addWidget(always_label)

        self._raw_clips_check = QCheckBox("+ Raw clips (audio only, no silence)")
        self._raw_clips_check.setToolTip("Also export trimmed clips — just the audio segments, no silence. Useful for sampling or sound design.")
        self._raw_clips_check.setChecked(False)
        exports_row.addWidget(self._raw_clips_check)

        exports_row.addStretch()
        out_layout.addLayout(exports_row)

        # Row 2: DAW project export
        daw_row = QHBoxLayout()
        daw_row.setSpacing(16)

        daw_label = QLabel("DAW Project:")
        daw_label.setStyleSheet(f"color: {Colors.TEXT_SOLID}; font-weight: 600;")
        daw_label.setFixedWidth(90)
        daw_row.addWidget(daw_label)

        self._reaper_check = QCheckBox("REAPER")
        self._reaper_check.setToolTip("Generate a .rpp project — open it in REAPER and all clips are on the timeline, color-coded")
        self._reaper_check.setChecked(False)
        daw_row.addWidget(self._reaper_check)

        self._aaf_check = QCheckBox("Pro Tools / Logic / Cubase")
        self._aaf_check.setToolTip("Generate an AAF session — opens in Pro Tools, Logic, Cubase, and Nuendo")
        self._aaf_check.setChecked(False)
        if not _check_pyaaf2():
            self._aaf_check.setEnabled(False)
            self._aaf_check.setToolTip("AAF export requires pyaaf2 — install with: pip install pyaaf2")
        daw_row.addWidget(self._aaf_check)

        self._dawproject_check = QCheckBox("Bitwig / Studio One")
        self._dawproject_check.setToolTip("Generate a .dawproject file — opens in Bitwig Studio and Studio One")
        self._dawproject_check.setChecked(False)
        daw_row.addWidget(self._dawproject_check)

        daw_row.addStretch()
        out_layout.addLayout(daw_row)

        # Hidden radio buttons — kept for internal processing logic
        self._mode_group = QButtonGroup(self)
        self._radio_merged = QRadioButton()
        self._radio_merged.setChecked(True)
        self._mode_group.addButton(self._radio_merged)
        self._radio_prepos = QRadioButton()
        self._mode_group.addButton(self._radio_prepos)
        self._radio_raw = QRadioButton()
        self._mode_group.addButton(self._radio_raw)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        out_layout.addWidget(self._progress_bar)

        # Bottom row: status + process button
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("statusLabel")
        bottom_row.addWidget(self._status_label, stretch=1)

        self._process_btn = QPushButton("PROCESS STEMS")
        self._process_btn.setObjectName("processBtn")
        self._process_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._process_btn.setFixedHeight(36)
        self._process_btn.setMinimumWidth(160)
        self._process_btn.clicked.connect(self._start_processing)
        bottom_row.addWidget(self._process_btn)

        out_layout.addLayout(bottom_row)
        output_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(output_panel)

        # ── INFO BAR (Ableton-style) ──
        self._info_bar = QLabel("Hover over any control to learn what it does")
        self._info_bar.setObjectName("infoBar")
        self._info_bar.setFixedHeight(22)
        main_layout.addWidget(self._info_bar)

        # Register hover descriptions for all interactive widgets
        self._hover_descriptions = {
            add_files_btn: "Add individual audio files (WAV, MP3, FLAC, OGG, AIF, M4A)",
            add_folder_btn: "Add all audio files from a folder at once",
            clear_btn: "Remove all files from the list and reset the preview",
            self._places_widget: "Places — pinned folders for quick access. Click to load, right-click to remove, drag folders here to pin",
            self._file_list: "Your loaded audio files — click one to preview its waveform",
            self._thresh_slider: "Silence Threshold — how quiet audio must be to count as silence. Lower values catch quieter sounds",
            self._silence_slider: "Min Silence Duration — how long a quiet gap must last before it's treated as a break between clips",
            self._pad_slider: "Padding — extra milliseconds of audio kept at the start and end of each clip to preserve transients",
            self._zoom_bar: "Zoom bar — drag edges to zoom in, drag middle to scroll, double-click to reset. You can also scroll-wheel on the waveform",
            self._waveform: "Waveform preview — teal regions show detected audio clips. Scroll to zoom, drag zoom bar above to navigate",
            self._raw_clips_check: "Also exports trimmed clips (just audio, no silence) into a Raw_Clips folder. Useful for sampling or sound design.",
            self._reaper_check: "REAPER — generates a .rpp project with all clips placed on the timeline, color-coded by instrument",
            self._aaf_check: "Pro Tools / Logic / Cubase — generates an AAF session file that opens in any of these DAWs",
            self._dawproject_check: "Bitwig / Studio One — generates a .dawproject file with all clips and audio bundled together",
            self._progress_bar: "Processing progress",
            self._process_btn: "Start processing — splits all loaded files, removes silence, and exports clips to a folder next to your originals",
        }
        for widget, desc in self._hover_descriptions.items():
            widget.setMouseTracking(True)
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == event.Type.Enter:
            desc = self._hover_descriptions.get(obj)
            if desc:
                self._info_bar.setText(desc)
        elif event.type() == event.Type.Leave:
            self._info_bar.setText("Hover over any control to learn what it does")
        return super().eventFilter(obj, event)

    # ── Window-level drag-and-drop ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drop_overlay.setGeometry(self.centralWidget().rect())
            self._drop_overlay.raise_()
            self._drop_overlay.show()

    def dragLeaveEvent(self, event):
        self._drop_overlay.hide()

    def dropEvent(self, event: QDropEvent):
        self._drop_overlay.hide()
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
            if paths:
                self._on_files_dropped(paths)
            event.acceptProposedAction()

    # ── File Management ──

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select audio stems", "",
            "Audio files (*.wav *.mp3 *.flac *.ogg *.aif *.aiff *.m4a);;All files (*.*)"
        )
        for f in files:
            if f not in self.input_files:
                self.input_files.append(f)
        self._refresh_file_list()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder of stems")
        if not folder:
            return
        for f in sorted(Path(folder).iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                path = str(f)
                if path not in self.input_files:
                    self.input_files.append(path)
        self._refresh_file_list()

    def _on_place_clicked(self, folder_path):
        self.input_files = []
        self.current_preview_file = None
        for f in sorted(Path(folder_path).iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.input_files.append(str(f))
        self._refresh_file_list()
        if not self.input_files:
            self._waveform.clear()
            self._zoom_bar.set_has_data(False)
            self._preview_info.setText("No audio files in this folder")

    def _clear_files(self):
        self.input_files = []
        self.current_preview_file = None
        self._refresh_file_list()
        self._waveform.clear()
        self._zoom_bar.set_has_data(False)
        self._preview_info.setText("Select a file to preview")

    def _on_files_dropped(self, paths):
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                for f in sorted(pp.iterdir()):
                    if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                        path = str(f)
                        if path not in self.input_files:
                            self.input_files.append(path)
            elif pp.suffix.lower() in SUPPORTED_EXTENSIONS:
                if p not in self.input_files:
                    self.input_files.append(p)
        self._refresh_file_list()

    def _refresh_file_list(self):
        self._file_list.clear()
        for f in self.input_files:
            name = Path(f).name
            cat = detect_instrument(name)
            item = QListWidgetItem(name)
            item.setToolTip(f"{f}\nDetected: {cat['name']}")
            item.setForeground(QColor(cat["color"]))
            self._file_list.addItem(item)

        count = len(self.input_files)
        if count == 0:
            self._file_count_label.setText("No files")
        else:
            self._file_count_label.setText(f"{count} file{'s' if count != 1 else ''}")

        # Auto-select first file if nothing selected
        if self.input_files and not self.current_preview_file:
            self._file_list.setCurrentRow(0)

    def _on_file_selected(self, row):
        if 0 <= row < len(self.input_files):
            self.current_preview_file = self.input_files[row]
            self._load_preview()

    # ── Preview ──

    def _load_preview(self):
        if not self.current_preview_file:
            return
        name = Path(self.current_preview_file).name
        self._preview_info.setText(f"Loading {name}...")
        self._triggerLoadWaveform.emit(self.current_preview_file, 1200)

    def _on_waveform_ready(self, peaks_l, peaks_r, is_stereo, duration_ms):
        self._waveform.set_waveform(peaks_l, peaks_r, is_stereo, duration_ms)
        self._zoom_bar.set_has_data(True, peaks=peaks_l)
        self._zoom_bar.reset_zoom()
        # Run detection immediately
        self._run_detection()

    def _on_slider_changed(self, val):
        if self.current_preview_file and not self.is_processing:
            self._debounce_timer.start()

    def _run_detection(self):
        if not self.current_preview_file:
            return
        self._triggerDetect.emit(
            self.current_preview_file,
            self._thresh_slider.value(),
            self._silence_slider.value(),
            self._pad_slider.value(),
        )

    def _on_regions_ready(self, regions):
        self._waveform.set_regions(regions)
        if self.current_preview_file:
            name = Path(self.current_preview_file).name
            dur_ms = self._waveform._duration_ms
            if dur_ms > 0:
                duration = str(timedelta(milliseconds=dur_ms)).split(".")[0]
            else:
                duration = "0:00:00"
            n = len(regions)
            ch = "stereo" if self._waveform._is_stereo else "mono"
            self._preview_info.setText(
                f"{name}  |  {duration}  |  {ch}  |  {n} region{'s' if n != 1 else ''}"
            )

    # ── Processing ──

    def _start_processing(self):
        if not self.input_files:
            QMessageBox.warning(self, "No Files", "Please add audio files first.")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self._process_btn.setEnabled(False)
        self._process_btn.setText("Processing...")
        self._progress_bar.setValue(0)
        self._status_label.setText("Starting...")

        mode = "merged"  # Always one file per stem
        export_formats = {
            "rpp": self._reaper_check.isChecked(),
            "aaf": self._aaf_check.isChecked(),
            "dawproject": self._dawproject_check.isChecked(),
            "raw_clips": self._raw_clips_check.isChecked(),
        }

        self._triggerProcessAll.emit(
            list(self.input_files),
            self._thresh_slider.value(),
            self._silence_slider.value(),
            self._pad_slider.value(),
            mode,
            export_formats,
        )

    def _on_progress(self, value, status):
        self._progress_bar.setValue(int(value * 1000))
        if status:
            self._status_label.setText(status)

    def _on_done(self, message, output_dir):
        self.is_processing = False
        self._process_btn.setEnabled(True)
        self._process_btn.setText("PROCESS STEMS")
        self._progress_bar.setValue(1000)
        self._status_label.setText(message)

        result = QMessageBox.question(
            self, "Processing Complete",
            f"{message}\n\nOutput: {output_dir}\n\nOpen output folder?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result == QMessageBox.StandardButton.Yes:
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])

    def _on_error(self, error_msg):
        self.is_processing = False
        self._process_btn.setEnabled(True)
        self._process_btn.setText("PROCESS STEMS")
        self._status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

    def closeEvent(self, event):
        self._worker_thread.quit()
        self._worker_thread.wait(2000)
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet())

    # License check — show activation dialog if not licensed
    if not _is_licensed():
        if not LicenseDialog.activate():
            sys.exit(0)

    window = StemSlicerApp()
    window.show()
    sys.exit(app.exec())
