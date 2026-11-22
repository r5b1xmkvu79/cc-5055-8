"""End-to-end pipeline shared by the GUI and the CLI.

Stages: load/synth clip -> KLT tracks -> motion decomposition -> background
camera solve -> foreground box solve -> composite frames. Every stage returns
structured failure information so callers (notably the GUI) can feed it back
into the kernel with adjusted parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camerasolve import solve_cameras
from .decomposition import BACKGROUND, BOX, DEFERRED, decompose, layer_track_data
from .features import extract_tracks
from .layer_drift import DriftReport, diagnose_drift
from .layer_tracker import LayerTrackerResult, track_layers
from .objectpose import render_overlay, solve_box
from .priors import Priors
from .synth import SCENARIOS, SceneConfig, generate


@dataclass
class PipelineResult:
    ok: bool
    frames: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    decomposition: object = None
    layer_tracking: LayerTrackerResult | None = None
    drift_report: DriftReport | None = None
    camera_solve: object = None
    box_solve: object = None
    composite: list = field(default_factory=list)
    K: np.ndarray | None = None
    width: int = 0
    height: int = 0
    priors: Priors | None = None
    paths: dict = field(default_factory=dict)
    stage_errors: dict = field(default_factory=dict)


def read_clip(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _apply_tracking(dec, tracking) -> None:
    """Commit reversible tracker decisions onto the decomposition label map.

    The decomposition's own label objects are mutated in place so every
    downstream stage (camera solve, box solve, composite, evaluation) reads one
    consistent layer partition. The original layer is preserved on every
    correction's ``inverse`` and recoverable via ``tracking.rollback()``.
    """
    for tid, layer in tracking.assignments.items():
        lab = dec.labels.get(tid)
        if lab is None or lab.layer == layer:
            continue
        lab.layer = layer
        if layer == DEFERRED:
            lab.confidence = min(lab.confidence, 0.5)
    dec.bg_tracks = sorted(tid for tid, lab in dec.labels.items()
                           if lab.layer == BACKGROUND)
    dec.box_tracks = sorted(tid for tid, lab in dec.labels.items()
                            if lab.layer == BOX)
    dec.deferred_tracks = sorted(tid for tid, lab in dec.labels.items()
                                 if lab.layer == DEFERRED)


def run_pipeline(scenario: str | None = None,
                 video_path: str | None = None,
                 gt_dir: str | None = None,
                 priors: Priors | None = None,
                 max_corners: int = 600,
                 min_active: int = 120,
                 anchor_span: int = 8,
                 seed: int = 0) -> PipelineResult:
    priors = priors or Priors()
    res = PipelineResult(ok=False, priors=priors)
    try:
        if scenario is not None:
            cfg = SceneConfig(scenario=scenario)
            if priors.box_size is None:
                priors.box_size = cfg.box_size.copy()
            gen = generate(cfg)
            video_path = gen["video_path"]
            gt_dir = gen["out_dir"]
            res.paths = dict(video=video_path, gt_dir=gt_dir,
                             gt_bundle=str(Path(gt_dir) / "ground_truth.npz"),
                             gt_obs=str(Path(gt_dir) / "observations.npz"))
        frames = read_clip(video_path)
        if not frames:
            res.stage_errors["io"] = f"no frames decoded from {video_path}"
            return res
        res.frames = frames
        h, w = frames[0].shape[:2]
        res.width, res.height = w, h
        if gt_dir and Path(gt_dir, "ground_truth.npz").exists():
            gt = np.load(Path(gt_dir, "ground_truth.npz"))
            res.K = gt["K"]
        else:
            from .geometry import make_calibration
            res.K = make_calibration(w, h)

        tracks = extract_tracks(frames, min_active=min_active, max_corners=max_corners)
        res.tracks = tracks
        if len(tracks) < 2 * priors.min_structure_tracks:
            res.stage_errors["tracking"] = (
                f"only {len(tracks)} tracks; need >= {2 * priors.min_structure_tracks}")
            return res

        dec = decompose(tracks, res.K, w, h, priors=priors, seed=seed,
                        anchor_span=anchor_span)
        res.decomposition = dec

        # Multi-hypothesis identity reconnection across same-speed windows.
        # Only decided corrections change a layer; observationally unresolved
        # tracks stay deferred with both hypotheses retained, and the full
        # correction log is reversible (LayerTrackerResult.rollback).
        tracking = track_layers(tracks, dec, strategy="offline")
        res.layer_tracking = tracking
        if tracking.corrections:
            _apply_tracking(dec, tracking)
        res.drift_report = diagnose_drift(tracks, dec, tracking, w, h)

        bg_data = layer_track_data(tracks, dec, BACKGROUND)
        cam = solve_cameras(bg_data, res.K, len(frames),
                            depth_range=priors.background_depth_range,
                            min_tracks=priors.min_structure_tracks)
        res.camera_solve = cam
        if not cam.ok:
            res.stage_errors["camera"] = cam.reason
            return res

        box = None
        if dec.box_tracks and priors.box_size is not None:
            box = solve_box(tracks, dec.box_tracks, cam.R, cam.t, res.K,
                            priors.box_size, frames=frames, seed=seed)
            res.box_solve = box
            if not box.ok:
                res.stage_errors["box"] = box.reason
        else:
            res.stage_errors["box"] = "no foreground layer or box size prior"

        res.composite = build_composite(frames, tracks, dec, cam, box, res.K, priors)
        res.ok = cam.ok
        return res
    except Exception as exc:  # surface kernel failures instead of crashing the GUI
        res.stage_errors["exception"] = f"{type(exc).__name__}: {exc}"
        return res


def build_composite(frames, tracks, decomposition, cam, box, K, priors: Priors):
    """Side-by-side region-motion layer view and virtual-box overlay."""
    out = []
    color_bg, color_box, color_defer = (255, 170, 60), (60, 120, 255), (120, 120, 120)
    for f, frame in enumerate(frames):
        layer_view = frame.copy()
        for tr in tracks:
            lab = decomposition.labels.get(tr.track_id)
            if lab is None:
                continue
            uv = tr.at(f)
            if uv is None:
                continue
            if lab.layer == BACKGROUND:
                c = color_bg
            elif lab.layer == BOX:
                c = color_box
            else:
                c = color_defer
            cv2.circle(layer_view, tuple(np.round(uv).astype(int)), 3, c, -1, cv2.LINE_AA)
        comp = frame.copy()
        if box is not None and box.ok:
            comp = render_overlay(comp, priors.box_size, box.R[f], box.t[f],
                                  cam.R[f], cam.t[f], K)
        gap = np.full((frame.shape[0], 12, 3), 40, dtype=np.uint8)
        out.append(np.hstack([layer_view, gap, comp]))
    return out


def save_composite(result: PipelineResult, out_path: str, fps: int = 15):
    if not result.composite:
        return None
    h, w = result.composite[0].shape[:2]
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for fr in result.composite:
        writer.write(fr)
    writer.release()
    return str(path)
