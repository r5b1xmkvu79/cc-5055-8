"""Background worker that runs the blocking pipeline off the GUI thread."""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from ..pipeline import PipelineResult, run_pipeline
from ..priors import Priors


class PipelineWorker(QObject):
    finished = Signal(object)   # PipelineResult
    failed = Signal(str)

    def __init__(self, scenario: str, priors: Priors, max_corners: int,
                 anchor_span: int, seed: int):
        super().__init__()
        self.scenario = scenario
        self.priors = priors
        self.max_corners = max_corners
        self.anchor_span = anchor_span
        self.seed = seed

    def run(self):
        try:
            result: PipelineResult = run_pipeline(
                scenario=self.scenario,
                priors=self.priors,
                max_corners=self.max_corners,
                anchor_span=self.anchor_span,
                seed=self.seed,
            )
            self.finished.emit(result)
        except Exception as exc:  # keep the GUI alive on any kernel failure
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def start_worker(scenario, priors, max_corners, anchor_span, seed,
                 on_finished, on_failed) -> tuple[QThread, PipelineWorker]:
    thread = QThread()
    worker = PipelineWorker(scenario, priors, max_corners, anchor_span, seed)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker
