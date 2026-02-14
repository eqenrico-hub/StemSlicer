# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for StemSlicer — works on macOS and Windows.
Bundles ffmpeg so the user doesn't need to install it separately.
"""

import sys
import os

block_cipher = None

# ── Detect ffmpeg binaries to bundle ──
ffmpeg_binaries = []
if sys.platform == 'win32':
    for name in ['ffmpeg.exe', 'ffprobe.exe']:
        if os.path.isfile(name):
            ffmpeg_binaries.append((name, '.'))
elif sys.platform == 'darwin':
    for name in ['ffmpeg', 'ffprobe']:
        if os.path.isfile(name):
            ffmpeg_binaries.append((name, '.'))

a = Analysis(
    ['stemslicer.py'],
    pathex=[],
    binaries=ffmpeg_binaries,
    datas=[
        ('theme.py', '.'),
    ],
    hiddenimports=[
        'pydub',
        'audioop_lts',
        'aaf2',
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'customtkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'PIL',
        'cv2',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='StemSlicer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name='StemSlicer',
    )
    app = BUNDLE(
        coll,
        name='StemSlicer.app',
        icon='StemSlicer.icns',
        bundle_identifier='com.stemslicer.app',
        info_plist={
            'CFBundleShortVersionString': '2.0.0',
            'CFBundleName': 'StemSlicer',
            'CFBundleIconFile': 'StemSlicer.icns',
            'NSHighResolutionCapable': True,
        },
    )
else:
    # Windows: single-file .exe with ffmpeg bundled inside
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='StemSlicer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='StemSlicer.ico',
    )
