"""Shot-tracking window: the documented primary entry point.

The GUI is a thin view over the same kernel the CLI drives. It lets the user

  * pick a scenario and the declared kernel priors (RANSAC threshold, label
    margin, anchor span, feature budget) and run the decomposition,
  * select which motion layer's observations/trajectories are shown,
  * click an observation to inspect exactly which track/layer/evidence backs it,
  * compare the region-motion decomposition against the virtual-model composite
    frame by frame,
  * read the kernel's reliability verdict and, on failure, adjust the controls
    and re-run (the failure feeds straight back into a new kernel invocation).

Run with  ./run_gui.sh  or  python -m dynamicsolver.gui.app
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from ..decomposition import BACKGROUND, BOX, DEFERRED
from ..priors import Priors
from ..synth import SCENARIOS
from .worker import start_worker

LAYER_COLORS = {
    BACKGROUND: (255, 170, 60),    # orange/blue in BGR
    BOX: (60, 120, 255),
    DEFERRED: (150, 150, 150),
}


def _bgr_to_qpix(frame_bgr: np.ndarray) -> QPixmap:
    rgb = frame_bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动态场景镜头求解器 · Dynamic-scene shot solver")
        self.resize(1500, 900)
        self.result = None
        self.thread = None
        self.frame_index = 0
        self.playing = False
        self.selected_layers = {BACKGROUND, BOX, DEFERRED}
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        root = QSplitter(Qt.Horizontal)

        # Left: controls + observation inspector.
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("场景 Scenario"))
        self.scenario_box = QComboBox()
        self.scenario_box.addItems(SCENARIOS)
        lv.addWidget(self.scenario_box)

        lv.addWidget(QLabel("RANSAC 阈值 Threshold (px)"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.2, 8.0)
        self.threshold.setSingleStep(0.1)
        self.threshold.setValue(1.5)
        lv.addWidget(self.threshold)

        lv.addWidget(QLabel("标签裕度 Label margin"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.01, 1.0)
        self.margin.setSingleStep(0.01)
        self.margin.setValue(0.15)
        lv.addWidget(self.margin)

        lv.addWidget(QLabel("锚点跨度 Anchor span (frames)"))
        self.anchor = QSpinBox()
        self.anchor.setRange(2, 20)
        self.anchor.setValue(8)
        lv.addWidget(self.anchor)

        lv.addWidget(QLabel("特征上限 Max corners"))
        self.max_corners = QSpinBox()
        self.max_corners.setRange(100, 2000)
        self.max_corners.setSingleStep(50)
        self.max_corners.setValue(600)
        lv.addWidget(self.max_corners)

        self.run_btn = QPushButton("运行分解 Run decomposition")
        self.run_btn.clicked.connect(self._run)
        lv.addWidget(self.run_btn)

        self.verdict_label = QLabel("内核状态 Kernel: 未运行 idle")
        self.verdict_label.setWordWrap(True)
        lv.addWidget(self.verdict_label)

        lv.addWidget(QLabel("运动层 Motion layers"))
        self.layer_list = QListWidget()
        for key, name in ((BACKGROUND, "背景 Background"),
                          (BOX, "前景箱体 Box"),
                          (DEFERRED, "挂起 Deferred")):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, key)
            self.layer_list.addItem(item)
        self.layer_list.itemChanged.connect(self._on_layer_toggle)
        lv.addWidget(self.layer_list)

        lv.addWidget(QLabel("观测检查 Observation inspector (click a row)"))
        self.inspector = QListWidget()
        self.inspector.currentItemChanged.connect(self._on_inspect)
        lv.addWidget(self.inspector, 1)
        root.addWidget(left)

        # Right: two image panels + transport.
        right = QWidget()
        rv = QVBoxLayout(right)
        panels = QSplitter(Qt.Horizontal)
        self.region_view = QLabel("区域运动 Region motion")
        self.region_view.setAlignment(Qt.AlignCenter)
        self.region_view.setMinimumSize(640, 480)
        self.composite_view = QLabel("虚拟模型合成 Composite")
        self.composite_view.setAlignment(Qt.AlignCenter)
        self.composite_view.setMinimumSize(640, 480)
        panels.addWidget(self.region_view)
        panels.addWidget(self.composite_view)
        rv.addWidget(panels, 1)

        bar = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        bar.addWidget(self.play_btn)
        self.frame_slider = QSpinBox()
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_spin)
        bar.addWidget(self.frame_slider)
        self.frame_info = QLabel("/ -")
        bar.addWidget(self.frame_info)
        rv.addLayout(bar)
        root.addWidget(right)
        root.setSizes([360, 1140])
        self.setCentralWidget(root)

    # ---------------------------------------------------------------- kernel run
    def _run(self):
        self.run_btn.setEnabled(False)
        self.verdict_label.setText("内核运行中 Kernel running …")
        priors = Priors(
            ransac_threshold_px=self.threshold.value(),
            label_margin=self.margin.value(),
        )
        self.thread, worker = start_worker(
            self.scenario_box.currentText(), priors,
            self.max_corners.value(), self.anchor.value(), seed=0,
            on_finished=self._on_finished, on_failed=self._on_failed,
        )
        self.thread.start()

    def _on_finished(self, result):
        self.result = result
        self.run_btn.setEnabled(True)
        if result.stage_errors:
            self.verdict_label.setText(
                "阶段错误 Stage errors: " + "; ".join(
                    f"{k}: {v}" for k, v in result.stage_errors.items()))
        else:
            dec = result.decomposition
            self.verdict_label.setText(
                f"结论 Verdict: {dec.reason} | "
                f"背景 {len(dec.bg_tracks)}  前景 {len(dec.box_tracks)}  "
                f"挂起 {len(dec.deferred_tracks)}"
                + (f"\n警告 {dec.warnings}" if dec.warnings else ""))
        n = len(result.frames)
        self.frame_slider.setRange(0, max(n - 1, 0))
        self.frame_info.setText(f"/ {n - 1}")
        self.frame_index = 0
        self._populate_inspector()
        self._render_frame()

    def _on_failed(self, message):
        self.run_btn.setEnabled(True)
        self.verdict_label.setText("失败 Failure: " + message)
        QMessageBox.critical(self, "Kernel failure", message)

    # ------------------------------------------------------------- layer/inspect
    def _on_layer_toggle(self, item):
        key = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self.selected_layers.add(key)
        else:
            self.selected_layers.discard(key)
        self._render_frame()

    def _populate_inspector(self):
        self.inspector.clear()
        if self.result is None:
            return
        dec = self.result.decomposition
        for tr in self.result.tracks:
            lab = dec.labels.get(tr.track_id)
            if lab is None:
                continue
            item = QListWidgetItem(
                f"#{tr.track_id:03d} {lab.layer:10s} conf={lab.confidence:.2f} "
                f"frames {tr.start}-{tr.end}")
            item.setData(Qt.UserRole, tr.track_id)
            self.inspector.addItem(item)

    def _on_inspect(self, current, _previous):
        if current is None or self.result is None:
            return
        tid = current.data(Qt.UserRole)
        self._render_frame(highlight_tid=tid)

    # ------------------------------------------------------------------- frames
    def _toggle_play(self):
        if self.result is None:
            return
        self.playing = not self.playing
        self.play_btn.setText("⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._timer.start(200)
        else:
            self._timer.stop()

    def _advance_frame(self):
        n = len(self.result.frames) if self.result else 1
        self.frame_index = (self.frame_index + 1) % max(n, 1)
        self.frame_slider.setValue(self.frame_index)

    def _on_frame_spin(self, value):
        self.frame_index = value
        self._render_frame()

    def _render_frame(self, highlight_tid=None):
        if self.result is None:
            return
        f = self.frame_index
        frame = self.result.frames[f]
        dec = self.result.decomposition

        region = frame.copy()
        for tr in self.result.tracks:
            lab = dec.labels.get(tr.track_id)
            if lab is None or lab.layer not in self.selected_layers:
                continue
            uv = tr.at(f)
            if uv is None:
                continue
            color = LAYER_COLORS[lab.layer]
            radius = 6 if tr.track_id == highlight_tid else 3
            cv_circle(region, uv, color, radius)
        if highlight_tid is not None:
            self._draw_track_path(region, highlight_tid)
        self.region_view.setPixmap(_bgr_to_qpix(region).scaled(
            self.region_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        comp = self._composite_for_frame(f)
        self.composite_view.setPixmap(_bgr_to_qpix(comp).scaled(
            self.composite_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _draw_track_path(self, img, tid):
        tr = next((t for t in self.result.tracks if t.track_id == tid), None)
        if tr is None:
            return
        pts = [tuple(np.round(uv).astype(int)) for uv in tr.uvs]
        import cv2
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (255, 255, 255), 1, cv2.LINE_AA)

    def _composite_for_frame(self, f):
        if self.result.composite:
            # composite frames already concatenate region view + gap + overlay;
            # show only the overlay (right-hand) half for the composite panel.
            full = self.result.composite[f]
            return full[:, full.shape[1] // 2 + 6:]
        return self.result.frames[f]


def cv_circle(img, uv, color_bgr, radius):
    import cv2
    cv2.circle(img, tuple(np.round(uv).astype(int)), radius,
               color_bgr, -1, cv2.LINE_AA)


def main():
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
