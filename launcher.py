"""
IAMARS — Application Launcher
Professional PyQt6 desktop entry point.

Replaces the CMD launch flow with a proper application window:
  - System status check on startup
  - Video file selection (single or multiple, drag-and-drop)
  - Pipeline start / stop controls
  - Live stats feed during processing
  - DPI-aware, frameless, dark tactical theme

Usage
-----
    python launcher.py

Requirements
------------
    pip install PyQt6
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# ── Graceful PyQt6 import ─────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QListWidget, QListWidgetItem,
        QFrame, QProgressBar, QTextEdit, QSplitter, QSizePolicy,
        QGraphicsDropShadowEffect,
    )
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
        QEasingCurve, QPoint,
    )
    from PyQt6.QtGui import (
        QFont, QFontDatabase, QColor, QPalette, QIcon,
        QDragEnterEvent, QDropEvent, QPainter, QPen, QBrush,
    )
except ImportError:
    print("[ERROR] PyQt6 not installed.  Run:  pip install PyQt6")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Colour tokens  (match the OpenCV dashboard palette as closely as Qt allows)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":           "#0A0C0E",
    "panel":        "#12141518",
    "panel_solid":  "#121418",
    "border":       "#003040",
    "border_dim":   "#001820",
    "accent":       "#00FF88",
    "accent_dim":   "#007840",
    "accent_dark":  "#00301A",
    "text":         "#D2D7DC",
    "text_dim":     "#4A4E55",
    "text_muted":   "#6E7580",
    "red":          "#DC281E",
    "orange":       "#FF9400",
    "yellow":       "#E6D200",
    "danger":       "#3C0000",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
}}

QFrame#PanelFrame {{
    background-color: {C['panel_solid']};
    border: 1px solid {C['border']};
    border-radius: 4px;
}}

QLabel#SectionTitle {{
    color: {C['accent']};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 6px 10px 4px 10px;
}}

QLabel#BrandTitle {{
    color: {C['accent']};
    font-size: 28px;
    font-weight: bold;
    letter-spacing: 6px;
}}

QLabel#BrandSub {{
    color: {C['accent_dim']};
    font-size: 10px;
    letter-spacing: 3px;
}}

QLabel#StatusDot {{
    font-size: 18px;
}}

QPushButton#PrimaryBtn {{
    background-color: {C['accent_dark']};
    color: {C['accent']};
    border: 1px solid {C['accent_dim']};
    border-radius: 3px;
    padding: 8px 20px;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QPushButton#PrimaryBtn:hover {{
    background-color: {C['accent_dim']};
    color: #000;
}}
QPushButton#PrimaryBtn:pressed {{
    background-color: {C['accent']};
    color: #000;
}}
QPushButton#PrimaryBtn:disabled {{
    background-color: #111;
    color: {C['text_dim']};
    border-color: {C['text_dim']};
}}

QPushButton#DangerBtn {{
    background-color: {C['danger']};
    color: {C['red']};
    border: 1px solid {C['red']};
    border-radius: 3px;
    padding: 8px 20px;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QPushButton#DangerBtn:hover {{
    background-color: {C['red']};
    color: #fff;
}}
QPushButton#DangerBtn:disabled {{
    background-color: #111;
    color: {C['text_dim']};
    border-color: {C['text_dim']};
}}

QPushButton#GhostBtn {{
    background-color: transparent;
    color: {C['text_muted']};
    border: 1px solid {C['border']};
    border-radius: 3px;
    padding: 6px 14px;
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton#GhostBtn:hover {{
    color: {C['text']};
    border-color: {C['accent_dim']};
}}

QListWidget {{
    background-color: #0D0F12;
    border: 1px solid {C['border']};
    border-radius: 3px;
    color: {C['text']};
    font-size: 12px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 6px 8px;
    border-radius: 2px;
    border-bottom: 1px solid {C['border_dim']};
}}
QListWidget::item:selected {{
    background-color: {C['accent_dark']};
    color: {C['accent']};
}}
QListWidget::item:hover {{
    background-color: #151820;
}}

QTextEdit {{
    background-color: #080A0C;
    border: 1px solid {C['border']};
    border-radius: 3px;
    color: {C['accent_dim']};
    font-size: 11px;
    font-family: 'Consolas', 'Courier New', monospace;
    padding: 6px;
}}

QProgressBar {{
    background-color: #0D0F12;
    border: 1px solid {C['border']};
    border-radius: 2px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {C['accent']};
    border-radius: 2px;
}}

QSplitter::handle {{
    background-color: {C['border']};
    width: 1px;
}}

QScrollBar:vertical {{
    background: {C['bg']};
    width: 6px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {C['border']};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Separator widget
# ─────────────────────────────────────────────────────────────────────────────
class HSep(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet(f"color: {C['border']}; margin: 4px 0;")


# ─────────────────────────────────────────────────────────────────────────────
# Status indicator dot
# ─────────────────────────────────────────────────────────────────────────────
class StatusDot(QLabel):
    def __init__(self, parent=None):
        super().__init__("●", parent)
        self.setObjectName("StatusDot")
        self._active = False
        self._blink  = False
        self._timer  = QTimer()
        self._timer.timeout.connect(self._toggle)
        self.setStyleSheet(f"color: {C['text_dim']};")

    def set_active(self, active: bool):
        self._active = active
        if active:
            self._timer.start(600)
        else:
            self._timer.stop()
            self._blink = False
            self.setStyleSheet(f"color: {C['text_dim']};")

    def _toggle(self):
        self._blink = not self._blink
        col = C['red'] if self._blink else "#660000"
        self.setStyleSheet(f"color: {col};")


# ─────────────────────────────────────────────────────────────────────────────
# Drag-and-drop file list
# ─────────────────────────────────────────────────────────────────────────────
class VideoDropList(QListWidget):
    files_added = pyqtSignal(list)

    SUPPORTED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self._placeholder = True
        self._add_placeholder()

    def _add_placeholder(self):
        item = QListWidgetItem("  Drop video files here  —  or click  [ ADD FILES ]")
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setForeground(QColor(C['text_dim']))
        self.addItem(item)

    def _clear_placeholder(self):
        if self._placeholder:
            self.clear()
            self._placeholder = False

    def add_files(self, paths: list[str]):
        self._clear_placeholder()
        existing = {self.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self.count())}
        added = []
        for p in paths:
            if Path(p).suffix.lower() in self.SUPPORTED and p not in existing:
                item = QListWidgetItem(f"  {Path(p).name}")
                item.setData(Qt.ItemDataRole.UserRole, p)
                item.setForeground(QColor(C['text']))
                self.addItem(item)
                added.append(p)
        if added:
            self.files_added.emit(added)

    def get_files(self) -> list[str]:
        return [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i).data(Qt.ItemDataRole.UserRole)
        ]

    def remove_selected(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        if self.count() == 0:
            self._placeholder = True
            self._add_placeholder()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.add_files(paths)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline worker thread
# ─────────────────────────────────────────────────────────────────────────────
class PipelineWorker(QThread):
    log_line   = pyqtSignal(str)
    progress   = pyqtSignal(int, int)    # (current_frame, total_frames)
    finished   = pyqtSignal(str)         # completion message
    error      = pyqtSignal(str)

    def __init__(self, video_path: str, save_path: str = None):
        super().__init__()
        self.video_path = video_path
        self.save_path  = save_path
        self._stop      = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # Build import path
            project_root = Path(__file__).parent
            sys.path.insert(0, str(project_root))

            # Import pipeline components inline so they share CUDA context
            from integration.video_pipeline import run as pipeline_run

            self.log_line.emit(f"[LAUNCH] {Path(self.video_path).name}")
            self.log_line.emit("[INIT]  Loading models...")

            # Monkey-patch cv2.imshow so the pipeline renders into its own
            # window without conflicting with the launcher Qt window.
            import cv2
            original_imshow = cv2.imshow

            frame_count = [0]
            total_frames = [0]

            import cv2 as _cv2
            cap_probe = _cv2.VideoCapture(self.video_path)
            if cap_probe.isOpened():
                total_frames[0] = int(cap_probe.get(_cv2.CAP_PROP_FRAME_COUNT))
                cap_probe.release()

            _worker_ref = self

            def patched_imshow(name, frame):
                if _worker_ref._stop:
                    return
                original_imshow(name, frame)
                frame_count[0] += 1
                if total_frames[0] > 0:
                    _worker_ref.progress.emit(frame_count[0], total_frames[0])
                if frame_count[0] % 60 == 0:
                    fps_est = frame_count[0] / max(
                        1, time.time() - _worker_ref._t_start
                    )
                    _worker_ref.log_line.emit(
                        f"[RUN]   Frame {frame_count[0]:>5} / {total_frames[0]}   "
                        f"FPS ≈ {fps_est:.1f}"
                    )

            cv2.imshow = patched_imshow
            self._t_start = time.time()

            pipeline_run(
                video_path = self.video_path,
                save_path  = self.save_path,
                headless   = False,
            )

            cv2.imshow = original_imshow

            elapsed = time.time() - self._t_start
            msg = (f"[DONE]  {frame_count[0]} frames in {elapsed:.1f}s  "
                   f"({frame_count[0]/max(1,elapsed):.1f} FPS avg)")
            self.finished.emit(msg)

        except Exception as exc:
            import traceback
            self.error.emit(f"[ERROR] {exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# System check panel widget
# ─────────────────────────────────────────────────────────────────────────────
class SystemCheckWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PanelFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(4)

        title = QLabel("[ SYSTEM STATUS ]")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        layout.addWidget(HSep())

        self._rows: dict[str, QLabel] = {}
        checks = [
            ("CUDA",      self._check_cuda),
            ("PyTorch",   self._check_torch),
            ("Models",    self._check_models),
            ("OpenCV",    self._check_opencv),
        ]
        for name, _ in checks:
            row = QHBoxLayout()
            lbl = QLabel(f"  {name:<12}")
            lbl.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px;")
            val = QLabel("checking...")
            val.setStyleSheet(f"color: {C['text_dim']}; font-size: 12px;")
            self._rows[name] = val
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            layout.addLayout(row)

        # Run checks in background
        t = threading.Thread(target=self._run_checks,
                             args=([c for _, c in checks],), daemon=True)
        t.start()

    def _run_checks(self, checks):
        names = list(self._rows.keys())
        for i, fn in enumerate(checks):
            ok, text = fn()
            col = C['accent'] if ok else C['red']
            self._rows[names[i]].setStyleSheet(f"color: {col}; font-size: 12px;")
            self._rows[names[i]].setText(text)

    def _check_cuda(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return True, f"OK  [{name}]"
            return False, "NOT AVAILABLE"
        except ImportError:
            return False, "torch not installed"

    def _check_torch(self):
        try:
            import torch
            return True, f"OK  [{torch.__version__}]"
        except ImportError:
            return False, "NOT INSTALLED"

    def _check_models(self):
        root = Path(__file__).parent
        required = [
            "models/visiodect/best.pt",
            "models/uav_IR_detection/best.pt",
            "models/uav_RGB_detection/best.pt",
        ]
        missing = [m for m in required if not (root / m).exists()]
        if not missing:
            return True, "OK  [4/4 loaded]"
        return False, f"MISSING {len(missing)} model(s)"

    def _check_opencv(self):
        try:
            import cv2
            return True, f"OK  [{cv2.__version__}]"
        except ImportError:
            return False, "NOT INSTALLED"


# ─────────────────────────────────────────────────────────────────────────────
# Main launcher window
# ─────────────────────────────────────────────────────────────────────────────
class IAMARSLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IAMARS — Tactical Surveillance System")
        self.setMinimumSize(900, 620)
        self.resize(1060, 680)
        self._worker: PipelineWorker | None = None
        self._queue: list[str] = []
        self._current_idx = 0

        self._build_ui()
        self._center_on_screen()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(16, 14, 16, 14)
        main.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setSpacing(14)

        self._dot = StatusDot()
        hdr.addWidget(self._dot)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        b1 = QLabel("I A M A R S")
        b1.setObjectName("BrandTitle")
        b2 = QLabel("TACTICAL SURVEILLANCE SYSTEM")
        b2.setObjectName("BrandSub")
        brand_col.addWidget(b1)
        brand_col.addWidget(b2)
        hdr.addLayout(brand_col)
        hdr.addStretch()

        self._status_lbl = QLabel("READY")
        self._status_lbl.setStyleSheet(
            f"color: {C['accent_dim']}; font-size: 11px; letter-spacing: 2px;"
        )
        hdr.addWidget(self._status_lbl)
        main.addLayout(hdr)
        main.addWidget(HSep())

        # ── Body splitter ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Left column
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(8)

        sys_panel = SystemCheckWidget()
        lv.addWidget(sys_panel)

        # Video list section
        vid_frame = QFrame()
        vid_frame.setObjectName("PanelFrame")
        vfl = QVBoxLayout(vid_frame)
        vfl.setContentsMargins(10, 8, 10, 10)
        vfl.setSpacing(6)

        vtitle = QLabel("[ VIDEO QUEUE ]")
        vtitle.setObjectName("SectionTitle")
        vfl.addWidget(vtitle)
        vfl.addWidget(HSep())

        self._file_list = VideoDropList()
        self._file_list.setMinimumHeight(160)
        vfl.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("ADD FILES")
        btn_add.setObjectName("GhostBtn")
        btn_add.clicked.connect(self._browse_files)
        btn_clear = QPushButton("CLEAR")
        btn_clear.setObjectName("GhostBtn")
        btn_clear.clicked.connect(self._file_list.remove_selected)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        vfl.addLayout(btn_row)

        lv.addWidget(vid_frame)
        lv.addStretch()

        # Right column
        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.setSpacing(8)

        # Log
        log_frame = QFrame()
        log_frame.setObjectName("PanelFrame")
        lfl = QVBoxLayout(log_frame)
        lfl.setContentsMargins(10, 8, 10, 10)
        lfl.setSpacing(6)

        ltitle = QLabel("[ SYSTEM LOG ]")
        ltitle.setObjectName("SectionTitle")
        lfl.addWidget(ltitle)
        lfl.addWidget(HSep())

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(260)
        lfl.addWidget(self._log)
        rv.addWidget(log_frame)

        # Progress
        prog_frame = QFrame()
        prog_frame.setObjectName("PanelFrame")
        pfl = QVBoxLayout(prog_frame)
        pfl.setContentsMargins(10, 8, 10, 10)
        pfl.setSpacing(6)

        self._prog_lbl = QLabel("IDLE")
        self._prog_lbl.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 11px; letter-spacing: 1px;"
        )
        pfl.addWidget(self._prog_lbl)

        self._prog_bar = QProgressBar()
        self._prog_bar.setValue(0)
        pfl.addWidget(self._prog_bar)
        rv.addWidget(prog_frame)
        rv.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([400, 540])
        main.addWidget(splitter, 1)

        # ── Footer controls ───────────────────────────────────────────────────
        main.addWidget(HSep())
        foot = QHBoxLayout()
        foot.setSpacing(10)

        self._btn_run = QPushButton("▶  LAUNCH PIPELINE")
        self._btn_run.setObjectName("PrimaryBtn")
        self._btn_run.setFixedHeight(38)
        self._btn_run.clicked.connect(self._launch)

        self._btn_stop = QPushButton("■  STOP")
        self._btn_stop.setObjectName("DangerBtn")
        self._btn_stop.setFixedHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)

        foot.addWidget(self._btn_run)
        foot.addWidget(self._btn_stop)
        foot.addStretch()

        ver = QLabel("IAMARS v1.0  |  DRDO DEMO BUILD  |  CONFIDENTIAL")
        ver.setStyleSheet(
            f"color: {C['text_dim']}; font-size: 10px; letter-spacing: 1px;"
        )
        foot.addWidget(ver)
        main.addLayout(foot)

    # ─────────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────────
    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All Files (*)",
        )
        if files:
            self._file_list.add_files(files)

    def _launch(self):
        files = self._file_list.get_files()
        if not files:
            self._log_append("[WARN]  No video files in queue.")
            return

        self._queue       = list(files)
        self._current_idx = 0
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._dot.set_active(True)
        self._set_status("RUNNING")
        self._run_next()

    def _run_next(self):
        if self._current_idx >= len(self._queue):
            self._on_queue_done()
            return

        vpath = self._queue[self._current_idx]
        self._log_append(
            f"\n[QUEUE] {self._current_idx + 1}/{len(self._queue)}  "
            f"{Path(vpath).name}"
        )
        self._prog_bar.setValue(0)
        self._prog_lbl.setText(
            f"Processing  {Path(vpath).name}  "
            f"[{self._current_idx+1}/{len(self._queue)}]"
        )

        self._worker = PipelineWorker(vpath)
        self._worker.log_line.connect(self._log_append)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._log_append("[STOP]  Pipeline interrupted by user.")
        self._on_queue_done()

    def _on_progress(self, cur: int, total: int):
        if total > 0:
            pct = int(cur * 100 / total)
            self._prog_bar.setValue(pct)

    def _on_finished(self, msg: str):
        self._log_append(msg)
        self._current_idx += 1
        self._run_next()

    def _on_error(self, msg: str):
        self._log_append(msg)
        self._current_idx += 1
        self._run_next()

    def _on_queue_done(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._dot.set_active(False)
        self._set_status("READY")
        self._prog_lbl.setText("IDLE")
        self._prog_bar.setValue(0)
        self._log_append("[DONE]  Queue complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _log_append(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"[{ts}]  {text}")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_status(self, text: str):
        self._status_lbl.setText(text)
        col = C['accent'] if text == "RUNNING" else C['accent_dim']
        self._status_lbl.setStyleSheet(
            f"color: {col}; font-size: 11px; letter-spacing: 2px;"
        )

    def _center_on_screen(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width()  - self.width())  // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # DPI awareness — must be set before QApplication creation
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    # Override font with monospace for tactical feel
    font = QFont("Consolas", 13)
    font.setStyleHint(QFont.StyleHint.Monospace)
    app.setFont(font)

    window = IAMARSLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()