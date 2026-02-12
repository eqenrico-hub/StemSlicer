#!/usr/bin/env python3
"""
StemSlicer — Automatic Silence Removal for FL Studio / Any DAW

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
import subprocess
from pathlib import Path
from datetime import timedelta

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}


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
    stem_name = Path(audio_path).stem
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
        lines.append(f"  <TRACK {{{_guid()}}}")
        lines.append(f'    NAME "{stem_name}"')
        lines.append(f"    TRACKID {track_num}")
        lines.append("    VOLPAN 1 0 -1 -1 1")
        lines.append("    MUTESOLO 0 0 0")
        lines.append("    IPHASE 0")
        lines.append("    ISBUS 0 0")
        lines.append("    FX 1")
        lines.append("    PEAKCOL 16576")
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
    QScrollArea, QAbstractItemView,
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
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(10)

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
        layout.setSpacing(4)

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
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)

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


class WaveformWidget(QWidget):
    """Antialiased stereo waveform display with region overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._peaks_left = None
        self._peaks_right = None
        self._is_stereo = False
        self._duration_ms = 0
        self._regions = []

        # Cached paths — rebuilt on resize or new data
        self._path_left_fill = None
        self._path_right_fill = None
        self._cache_width = 0
        self._cache_height = 0

    def set_waveform(self, peaks_left, peaks_right, is_stereo, duration_ms):
        self._peaks_left = peaks_left
        self._peaks_right = peaks_right
        self._is_stereo = is_stereo
        self._duration_ms = duration_ms
        self._invalidate_cache()
        self.update()

    def set_regions(self, regions):
        self._regions = regions
        self.update()

    def clear(self):
        self._peaks_left = None
        self._peaks_right = None
        self._regions = []
        self._invalidate_cache()
        self.update()

    def _invalidate_cache(self):
        self._path_left_fill = None
        self._path_right_fill = None
        self._cache_width = 0
        self._cache_height = 0

    def _build_paths(self, w, h):
        if self._peaks_left is None:
            return

        if self._is_stereo:
            half_h = h / 2.0
            self._path_left_fill = self._make_waveform_path(
                self._peaks_left, w, 0, half_h, margin=4)
            self._path_right_fill = self._make_waveform_path(
                self._peaks_right, w, half_h, half_h, margin=4)
        else:
            self._path_left_fill = self._make_waveform_path(
                self._peaks_left, w, 0, h, margin=6)
            self._path_right_fill = None

        self._cache_width = w
        self._cache_height = h

    def _make_waveform_path(self, peaks, w, y_offset, h, margin=4):
        """Build a filled polygon path for a waveform channel."""
        n = len(peaks)
        if n == 0:
            return None

        mid_y = y_offset + h / 2.0
        usable_h = (h / 2.0) - margin

        # Build top edge (max peaks) left-to-right, then bottom edge (min peaks) right-to-left
        path = QPainterPath()

        # x step
        x_step = w / max(n - 1, 1)

        # Top edge (max values)
        x = 0.0
        first_y = mid_y - peaks[0][1] * usable_h
        path.moveTo(x, first_y)
        for i in range(1, n):
            x = i * x_step
            y = mid_y - peaks[i][1] * usable_h
            path.lineTo(x, y)

        # Bottom edge (min values) right-to-left
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

        # Background
        painter.fillRect(self.rect(), QColor(Colors.WAVEFORM_BG))

        if self._peaks_left is None:
            # Empty state
            painter.setPen(QColor(Colors.TEXT_MUTED_SOLID))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Select a file to preview waveform")
            painter.end()
            return

        # Rebuild cached paths if needed
        if self._cache_width != w or self._cache_height != h:
            self._build_paths(w, h)

        # Draw region overlays (behind waveform)
        if self._duration_ms > 0:
            for start_ms, end_ms in self._regions:
                x1 = int(start_ms / self._duration_ms * w)
                x2 = int(end_ms / self._duration_ms * w)
                painter.fillRect(x1, 0, x2 - x1, h, QColor(0, 229, 204, 20))
                # Dashed boundary lines
                pen = QPen(QColor(Colors.ACCENT), 1, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(x1, 0, x1, h)
                painter.drawLine(x2, 0, x2, h)

        # Center line(s)
        center_pen = QPen(QColor(255, 255, 255, 25), 1)
        painter.setPen(center_pen)
        if self._is_stereo:
            half_h = h / 2.0
            painter.drawLine(0, int(half_h / 2), w, int(half_h / 2))
            painter.drawLine(0, int(half_h + half_h / 2), w, int(half_h + half_h / 2))
            # Channel divider
            painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
            painter.drawLine(0, int(half_h), w, int(half_h))
        else:
            painter.drawLine(0, h // 2, w, h // 2)

        # Draw waveform with gradient fill
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

        # Channel labels for stereo
        if self._is_stereo:
            painter.setPen(QColor(255, 255, 255, 60))
            f = painter.font()
            f.setPixelSize(9)
            painter.setFont(f)
            painter.drawText(6, 14, "L")
            painter.drawText(6, int(h / 2) + 14, "R")

        painter.end()

    def _make_wave_gradient(self, y_offset, section_height):
        mid = y_offset + section_height / 2.0
        grad = QLinearGradient(0, y_offset, 0, y_offset + section_height)
        grad.setColorAt(0.0, QColor(0, 229, 204, 30))
        grad.setColorAt(0.35, QColor(0, 229, 204, 130))
        grad.setColorAt(0.5, QColor(0, 229, 204, 180))
        grad.setColorAt(0.65, QColor(0, 229, 204, 130))
        grad.setColorAt(1.0, QColor(0, 229, 204, 30))
        return grad

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
                    mode, export_rpp):
        try:
            total_files = len(input_files)
            all_results = {}
            all_raw_results = {}

            first_parent = Path(input_files[0]).parent
            output_dir = os.path.join(str(first_parent), "StemSlicer_Output")
            os.makedirs(output_dir, exist_ok=True)

            if export_rpp and mode == "prepositioned":
                raw_output_dir = os.path.join(output_dir, "_reaper_clips")
                os.makedirs(raw_output_dir, exist_ok=True)

            for file_idx, audio_path in enumerate(input_files):
                stem_name = Path(audio_path).stem
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
                    f"Splitting {stem_name}: {len(regions)} regions... ({file_idx + 1}/{total_files})"
                )

                def progress_cb(current, total, fi=file_idx):
                    overall = (fi + current / total) / total_files
                    self.progressUpdate.emit(overall, "")

                clips = split_and_export(
                    audio_path, regions, output_dir, mode=mode,
                    progress_callback=progress_cb,
                )
                all_results[stem_name] = clips

                if export_rpp:
                    if mode == "prepositioned":
                        raw_clips = split_and_export(
                            audio_path, regions, raw_output_dir, mode="raw",
                        )
                        all_raw_results[stem_name] = raw_clips
                    else:
                        all_raw_results[stem_name] = clips

            # Generate REAPER project
            rpp_path = None
            if export_rpp and all_raw_results:
                self.progressUpdate.emit(0.95, "Generating REAPER project...")
                try:
                    audio = AudioSegment.from_file(input_files[0])
                    sr = audio.frame_rate
                except Exception:
                    sr = 44100
                rpp_path = generate_rpp(all_raw_results, output_dir, sample_rate=sr)

            total_clips = sum(len(c) for c in all_results.values())
            msg = f"Done! {total_clips} clips from {len(all_results)} stems"
            if rpp_path:
                msg += " + REAPER project"

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
    _triggerProcessAll = Signal(list, int, int, int, str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("StemSlicer")
        self.resize(960, 780)
        self.setMinimumSize(800, 650)
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

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Scroll area for small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(24, 0, 24, 20)
        main_layout.setSpacing(12)

        # Accent bar
        accent = AccentBar()
        main_layout.addWidget(accent)

        # Header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 8, 0, 0)

        title = QLabel("StemSlicer")
        title.setObjectName("title")
        header_layout.addWidget(title)

        subtitle = QLabel("Automatic Silence Removal for FL Studio / Any DAW")
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

        self._file_list = FileListWidget()
        self._file_list.filesDropped.connect(self._on_files_dropped)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        inp_layout.addWidget(self._file_list)

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
        main_layout.addWidget(detect_panel)

        # ── PREVIEW panel ──
        preview_panel = GlowPanel("PREVIEW")
        prev_layout = preview_panel.contentLayout()

        self._preview_info = QLabel("Select a file to preview")
        self._preview_info.setObjectName("previewInfo")
        prev_layout.addWidget(self._preview_info)

        self._waveform = WaveformWidget()
        prev_layout.addWidget(self._waveform, stretch=1)

        main_layout.addWidget(preview_panel, stretch=1)

        # ── OUTPUT panel ──
        output_panel = GlowPanel("OUTPUT")
        out_layout = output_panel.contentLayout()

        options_row = QHBoxLayout()
        options_row.setSpacing(16)

        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet(f"color: {Colors.TEXT_SOLID}; font-weight: 600;")
        options_row.addWidget(mode_label)

        self._mode_group = QButtonGroup(self)
        self._radio_prepos = QRadioButton("Pre-Positioned (FL Studio)")
        self._radio_prepos.setChecked(True)
        self._mode_group.addButton(self._radio_prepos)
        options_row.addWidget(self._radio_prepos)

        self._radio_raw = QRadioButton("Raw Clips")
        self._mode_group.addButton(self._radio_raw)
        options_row.addWidget(self._radio_raw)

        options_row.addStretch()

        self._reaper_check = QCheckBox("REAPER Export (.rpp)")
        self._reaper_check.setChecked(True)
        options_row.addWidget(self._reaper_check)

        out_layout.addLayout(options_row)

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
        self._process_btn.setFixedHeight(44)
        self._process_btn.setMinimumWidth(180)
        self._process_btn.clicked.connect(self._start_processing)
        bottom_row.addWidget(self._process_btn)

        out_layout.addLayout(bottom_row)
        main_layout.addWidget(output_panel)

        scroll.setWidget(scroll_content)

        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    # ── Window-level drag-and-drop ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag_active = True
            self.update()

    def dragLeaveEvent(self, event):
        self._drag_active = False
        self.update()

    def dropEvent(self, event: QDropEvent):
        self._drag_active = False
        self.update()
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

    def _clear_files(self):
        self.input_files = []
        self.current_preview_file = None
        self._refresh_file_list()
        self._waveform.clear()
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
            item = QListWidgetItem(Path(f).name)
            item.setToolTip(f)
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

        mode = "prepositioned" if self._radio_prepos.isChecked() else "raw"
        export_rpp = self._reaper_check.isChecked()

        self._triggerProcessAll.emit(
            list(self.input_files),
            self._thresh_slider.value(),
            self._silence_slider.value(),
            self._pad_slider.value(),
            mode,
            export_rpp,
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
    window = StemSlicerApp()
    window.show()
    sys.exit(app.exec())
