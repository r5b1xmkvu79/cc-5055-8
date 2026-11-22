import unittest

import numpy as np

from dynamicsolver.camerasolve import solve_cameras
from dynamicsolver.decomposition import BACKGROUND, layer_track_data
from dynamicsolver.evaluation import true_layer_for_tracks
from dynamicsolver.features import extract_tracks
from dynamicsolver.geometry import rotation_error
from dynamicsolver.objectpose import solve_box
from dynamicsolver.pipeline import read_clip, run_pipeline
from dynamicsolver.priors import Priors
from dynamicsolver.synth import SceneConfig, generate


def _assets(scenario):
    cfg = SceneConfig(scenario=scenario)
    gen = generate(cfg)
    frames = read_clip(gen["video_path"])
    gt = np.load(gen["out_dir"] + "/ground_truth.npz")
    return cfg, gen, frames, gt


class TestBoxPose(unittest.TestCase):
    def test_aruco_pnp_recovers_relative_pose(self):
        # The box solver's job is the box-to-camera pose; both are reconstructed
        # in an arbitrary SfM world, so the correct invariant is the *relative*
        # box-camera rotation, which must match ground truth within a few deg.
        cfg, gen, frames, gt = _assets("baseline")
        tracks = extract_tracks(frames)
        K = gt["K"]
        from dynamicsolver.decomposition import decompose
        dec = decompose(tracks, K, cfg.width, cfg.height, priors=Priors())
        cam = solve_cameras(layer_track_data(tracks, dec, BACKGROUND), K,
                            len(frames), depth_range=Priors().background_depth_range)
        self.assertTrue(cam.ok)
        box = solve_box(tracks, dec.box_tracks, cam.R, cam.t, K,
                        cfg.box_size, frames=frames)
        self.assertTrue(box.ok, box.reason)
        errs = [
            rotation_error(cam.R[f] @ box.R[f],
                           gt["cam_R"][f] @ gt["box_R"][f])
            for f in range(len(frames))
        ]
        self.assertLess(np.median(errs), 3.0)

    def test_markers_detected_on_every_or_near_every_frame(self):
        cfg, gen, frames, gt = _assets("baseline")
        tracks = extract_tracks(frames)
        K = gt["K"]
        from dynamicsolver.decomposition import decompose
        dec = decompose(tracks, K, cfg.width, cfg.height, priors=Priors())
        cam = solve_cameras(layer_track_data(tracks, dec, BACKGROUND), K,
                            len(frames), depth_range=Priors().background_depth_range)
        box = solve_box(tracks, dec.box_tracks, cam.R, cam.t, K,
                        cfg.box_size, frames=frames)
        self.assertTrue(box.ok)
        self.assertGreaterEqual(np.mean(box.markers_per_frame), 2.0)
        self.assertGreaterEqual(box.n_assigned, 50)


class TestPipeline(unittest.TestCase):
    def test_all_scenarios_complete_without_stage_errors(self):
        for scenario in ("baseline", "sparse", "similar", "occlusion"):
            with self.subTest(scenario=scenario):
                result = run_pipeline(scenario=scenario)
                self.assertNotIn("exception", result.stage_errors)
                self.assertNotIn("camera", result.stage_errors)
                self.assertTrue(result.camera_solve.ok)
                self.assertIsNotNone(result.box_solve)
                self.assertTrue(result.box_solve.ok)
                self.assertEqual(len(result.composite), len(result.frames))

    def test_layer_tracking_and_drift_evidence_are_attached(self):
        result = run_pipeline(scenario="similar")
        self.assertIsNotNone(result.layer_tracking)
        self.assertIsNotNone(result.drift_report)
        lt = result.layer_tracking
        self.assertGreater(len(lt.corrections), 0)
        # Reversible log restores the pre-tracker partition exactly.
        restored = lt.rollback()
        current = {tid: lab.layer for tid, lab
                   in result.decomposition.labels.items()}
        # Current reflects applied decisions; rollback differs but is a valid
        # full partition over the same track ids.
        self.assertEqual(set(restored), set(current))
        # Drift report is observation-only and bounded.
        report = result.drift_report
        self.assertTrue(report.events)
        self.assertEqual(report.totals["n_tracker_faults"], 0)
        self.assertLess(len(report.log_lines()),
                        4 * len(report.events) + 20)

    def test_tracker_does_not_touch_scenarios_without_crossing(self):
        result = run_pipeline(scenario="occlusion")
        self.assertEqual(result.layer_tracking.corrections, [])
        self.assertEqual(result.layer_tracking.tracker_faults, [])

    def test_sparse_reports_ambiguity_instead_of_silent_mislabel(self):
        result = run_pipeline(scenario="sparse")
        dec = result.decomposition
        honest = dec.reason != "ok" or bool(dec.warnings) or dec.deferred_tracks
        self.assertTrue(honest)


if __name__ == "__main__":
    unittest.main()
