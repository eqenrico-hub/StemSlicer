"""
StemSlicer — Premium Dark Theme (PySide6)

Colors, fonts, and QSS stylesheet for the premium GUI.
"""


class Colors:
    BG = "#0D0F14"
    SURFACE = "#151921"
    SURFACE_TOP = "#1C2030"
    ACCENT = "#00E5CC"
    ACCENT_DIM = "#00B8A3"
    SECONDARY = "#6C63FF"
    TEXT = "rgba(255, 255, 255, 0.9)"
    TEXT_MUTED = "rgba(255, 255, 255, 0.5)"
    DANGER = "#FF4757"
    WAVEFORM_BG = "#0A0C12"
    BORDER = "rgba(255, 255, 255, 0.05)"

    # Solid equivalents for contexts that don't support rgba
    TEXT_SOLID = "#E6E6E6"
    TEXT_MUTED_SOLID = "#808080"
    BORDER_SOLID = "#1A1C22"


class Fonts:
    FAMILY = "Inter"
    FALLBACK = "Segoe UI, Helvetica Neue, Arial, sans-serif"
    TITLE_SIZE = 26
    SUBTITLE_SIZE = 12
    HEADING_SIZE = 11
    BODY_SIZE = 12
    SMALL_SIZE = 10
    BUTTON_SIZE = 12
    PROCESS_BUTTON_SIZE = 14


def get_stylesheet():
    return f"""
    /* ── Global ── */
    QMainWindow {{
        background-color: {Colors.BG};
    }}

    QWidget {{
        color: {Colors.TEXT_SOLID};
        font-family: "{Fonts.FAMILY}", {Fonts.FALLBACK};
        font-size: {Fonts.BODY_SIZE}px;
    }}

    /* ── Labels ── */
    QLabel {{
        background: transparent;
        border: none;
    }}

    QLabel#title {{
        font-size: {Fonts.TITLE_SIZE}px;
        font-weight: 700;
        color: #FFFFFF;
    }}

    QLabel#subtitle {{
        font-size: {Fonts.SUBTITLE_SIZE}px;
        color: {Colors.TEXT_MUTED_SOLID};
    }}

    QLabel#panelHeading {{
        font-size: {Fonts.HEADING_SIZE}px;
        font-weight: 600;
        color: {Colors.TEXT_MUTED_SOLID};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    QLabel#sliderValue {{
        font-size: {Fonts.BODY_SIZE}px;
        font-weight: 600;
        color: {Colors.ACCENT};
    }}

    QLabel#sliderHint {{
        font-size: {Fonts.SMALL_SIZE}px;
        color: {Colors.TEXT_MUTED_SOLID};
    }}

    QLabel#fileCount {{
        font-size: {Fonts.SMALL_SIZE}px;
        color: {Colors.TEXT_MUTED_SOLID};
    }}

    QLabel#statusLabel {{
        font-size: {Fonts.BODY_SIZE}px;
        color: {Colors.TEXT_MUTED_SOLID};
    }}

    QLabel#previewInfo {{
        font-size: {Fonts.SMALL_SIZE}px;
        color: {Colors.TEXT_MUTED_SOLID};
    }}

    /* ── Buttons ── */
    QPushButton {{
        background-color: transparent;
        color: {Colors.ACCENT};
        border: 1px solid {Colors.ACCENT};
        border-radius: 6px;
        padding: 7px 18px;
        font-size: {Fonts.BUTTON_SIZE}px;
        font-weight: 600;
    }}

    QPushButton:hover {{
        background-color: rgba(0, 229, 204, 0.1);
    }}

    QPushButton:pressed {{
        background-color: rgba(0, 229, 204, 0.2);
    }}

    QPushButton#dangerBtn {{
        color: {Colors.DANGER};
        border-color: {Colors.DANGER};
    }}

    QPushButton#dangerBtn:hover {{
        background-color: rgba(255, 71, 87, 0.1);
    }}

    QPushButton#dangerBtn:pressed {{
        background-color: rgba(255, 71, 87, 0.2);
    }}

    QPushButton#processBtn {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.ACCENT}, stop:1 {Colors.ACCENT_DIM});
        color: #0D0F14;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: {Fonts.PROCESS_BUTTON_SIZE}px;
        font-weight: 700;
    }}

    QPushButton#processBtn:hover {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #1AFFDE, stop:1 {Colors.ACCENT});
    }}

    QPushButton#processBtn:pressed {{
        background-color: {Colors.ACCENT_DIM};
    }}

    QPushButton#processBtn:disabled {{
        background-color: #1A2A28;
        color: #4A5A58;
    }}

    /* ── Sliders ── */
    QSlider::groove:horizontal {{
        height: 4px;
        background: #1A1E2A;
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.ACCENT}, stop:1 {Colors.ACCENT_DIM});
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: {Colors.ACCENT};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}

    QSlider::handle:horizontal:hover {{
        background: #1AFFDE;
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
    }}

    /* ── List Widget ── */
    QListWidget {{
        background-color: {Colors.WAVEFORM_BG};
        border: 1px solid {Colors.BORDER_SOLID};
        border-radius: 6px;
        padding: 4px;
        outline: none;
    }}

    QListWidget::item {{
        color: {Colors.TEXT_SOLID};
        padding: 5px 8px;
        border-radius: 4px;
    }}

    QListWidget::item:selected {{
        background-color: rgba(0, 229, 204, 0.15);
        color: {Colors.ACCENT};
    }}

    QListWidget::item:hover {{
        background-color: rgba(0, 229, 204, 0.07);
    }}

    /* ── Scrollbar ── */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: rgba(255, 255, 255, 0.12);
        border-radius: 4px;
        min-height: 30px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: rgba(255, 255, 255, 0.2);
    }}

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: none;
        height: 0;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background: rgba(255, 255, 255, 0.12);
        border-radius: 4px;
        min-width: 30px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: rgba(255, 255, 255, 0.2);
    }}

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {{
        background: none;
        width: 0;
    }}

    /* ── Progress Bar ── */
    QProgressBar {{
        background-color: #1A1E2A;
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
        font-size: 0px;
    }}

    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.ACCENT}, stop:1 {Colors.SECONDARY});
        border-radius: 4px;
    }}

    /* ── Radio Buttons ── */
    QRadioButton {{
        color: {Colors.TEXT_SOLID};
        spacing: 8px;
        font-size: {Fonts.BODY_SIZE}px;
    }}

    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 2px solid {Colors.TEXT_MUTED_SOLID};
        border-radius: 10px;
        background: transparent;
    }}

    QRadioButton::indicator:checked {{
        border-color: {Colors.ACCENT};
        background: {Colors.ACCENT};
    }}

    QRadioButton::indicator:hover {{
        border-color: {Colors.ACCENT};
    }}

    /* ── Check Box ── */
    QCheckBox {{
        color: {Colors.TEXT_SOLID};
        spacing: 8px;
        font-size: {Fonts.BODY_SIZE}px;
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 2px solid {Colors.TEXT_MUTED_SOLID};
        border-radius: 4px;
        background: transparent;
    }}

    QCheckBox::indicator:checked {{
        border-color: {Colors.ACCENT};
        background: {Colors.ACCENT};
    }}

    QCheckBox::indicator:hover {{
        border-color: {Colors.ACCENT};
    }}

    /* ── Tooltip ── */
    QToolTip {{
        background-color: {Colors.SURFACE_TOP};
        color: {Colors.TEXT_SOLID};
        border: 1px solid {Colors.BORDER_SOLID};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: {Fonts.SMALL_SIZE}px;
    }}
    """
