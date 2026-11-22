"""Command-line helper for the motion decomposition kernel.

The GUI (dynamicsolver/gui/app.py, launched via run_gui.sh) is the documented
primary entry point. This CLI exists for headless runs, batch scenario tests
and CI; every option mirrors a GUI control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate
from .priors import Priors
from .pipeline import run_pipeline, save_composite
from .synth import SCENARIOS


def _summarize(result, gt_bundle=None, gt_obs=None, K=None):
    dec, cam, box = result.decomposition, result.camera_solve, result.box_solve
    summary = {
        "ok": result.ok,
        "stage_errors": result.stage_errors,
        "n_tracks": len(result.tracks),
        "background_tracks": len(dec.bg_tracks) if dec else 0,
        "box_tracks": len(dec.box_tracks) if dec else 0,
        "deferred_tracks": len(dec.deferred_tracks) if dec else 0,
        "crossing_windows": [list(w) for w in dec.crossing_windows] if dec else [],
        "internal_label_flips": dec.internal_label_flips if dec else 0,
        "verdict": dec.reason if dec else "no decomposition",
        "warnings": dec.warnings if dec else [],
        "camera": {"ok": cam.ok, "reason": cam.reason,
                   "reproj_px": None if not np.isfinite(cam.mean_reproj_px) else cam.mean_reproj_px,
                   "n_tracks": cam.n_tracks} if cam else None,
        "box": ({"ok": box.ok, "reason": box.reason,
                 "reproj_px": None if not np.isfinite(box.reproj_px) else box.reproj_px,
                 "n_assigned": box.n_assigned} if box else None),
    }
    if gt_bundle and gt_obs and cam and cam.ok:
        report = evaluate(result.tracks, dec, cam, box, K, gt_bundle, gt_obs)
        summary["evaluation"] = {
            "layer_f1": report.layer_f1,
            "identity_swaps": report.identity_swaps,
            "deferred_count": report.deferred_count,
            "camera_rot_med_deg": report.camera_rot_deg,
            "camera_t_dir_med_deg": report.camera_t_dir_deg,
            "camera_pos_med_m": report.camera_pos_m,
            "box_rot_med_deg": report.box_rot_deg,
            "box_pos_med_m": report.box_pos_m,
            "bg_reproj_rmse_px": report.bg_reproj_px,
            "box_reproj_rmse_px": report.box_reproj_px,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Headless helper for the shot decomposition kernel.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    parser.add_argument("--video", help="external clip instead of a synthetic scenario")
    parser.add_argument("--gt-dir", help="directory with ground_truth.npz/observations.npz")
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--label-margin", type=float, default=0.15)
    parser.add_argument("--max-corners", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output/run")
    args = parser.parse_args(argv)

    priors = Priors(ransac_threshold_px=args.threshold, label_margin=args.label_margin)
    result = run_pipeline(
        scenario=None if args.video else args.scenario,
        video_path=args.video, gt_dir=args.gt_dir,
        priors=priors, max_corners=args.max_corners, seed=args.seed,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.composite:
        comp_path = save_composite(result, str(out_dir / "composite.mp4"))
    else:
        comp_path = None
    gt_bundle = result.paths.get("gt_bundle")
    gt_obs = result.paths.get("gt_obs")
    summary = _summarize(result, gt_bundle, gt_obs, result.K)
    summary["composite_path"] = comp_path
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    print(json.dumps(summary, indent=2, default=str))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
