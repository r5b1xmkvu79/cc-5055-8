"""Tests for the observation-only layer-identity drift diagnostics.

Ground truth appears only in assertions about the *causes*; the diagnostic
implementation itself never opens a ground-truth bundle. The tests verify:

* same-speed and crossing-occlusion scenarios receive distinct causes,
* every anomaly is localised to an anchor-pair stage with layer attribution
  and geometry residuals,
* the log is bounded (detail truncation is reported),
* unresolved observational ties and implementation faults stay distinct.
"""
import unittest

import numpy as np

from dynamicsolver.decomposition import decompose
from dynamicsolver.features import extract_tracks
from dynamicsolver.layer_drift import (
    OCCLUSION, SAME_SPEED, DriftReport, diagnose_drift,
)
from dynamicsolver.layer_tracker import track_layers
from dynamicsolver.pipeline import read_clip
from dynamicsolver.priors import Priors
from dynamicsolver.synth import SceneConfig, generate


def _diagnose(scenario):
    cfg = SceneConfig(scenario=scenario)
    gen = generate(cfg)
    frames = read_clip(gen["video_path"])
    gt = np.load(gen["out_dir"] + "/ground_truth.npz")
    tracks = extract_tracks(frames)
    dec = decompose(tracks, gt["K"], cfg.width, cfg.height, priors=Priors())
    tracker = track_layers(tracks, dec, strategy="offline")
    report = diagnose_drift(tracks, dec, tracker, cfg.width, cfg.height)
    return cfg, dec, tracker, report


class TestDriftDiagnostics(unittest.TestCase):
    def test_similar_scenario_attributes_same_speed_cause(self):
        _, dec, _, report = _diagnose("similar")
        self.assertTrue(dec.crossing_windows)
        crossing = set(dec.crossing_windows)
        same_speed = [ev for ev in report.events
                      if ev.window in crossing and ev.cause == SAME_SPEED]
        self.assertTrue(same_speed, "the real crossing must be diagnosed as "
                        "same-speed motion from observations")
        event = same_speed[0]
        self.assertEqual(event.confidence, "supported")
        self.assertTrue(event.evidence_chain)

    def test_occlusion_scenario_is_distinguished_from_same_speed(self):
        _, dec, _, report = _diagnose("occlusion")
        # Occlusion produces no same-speed crossing, yet the diagnostic still
        # surfaces localised coverage-loss events with an occlusion cause.
        self.assertEqual(dec.crossing_windows, [])
        causes = {ev.cause for ev in report.events}
        self.assertIn(OCCLUSION, causes)
        self.assertNotIn(SAME_SPEED, causes)

    def test_every_event_carries_stage_localised_evidence(self):
        _, _, _, report = _diagnose("similar")
        self.assertIsInstance(report, DriftReport)
        for event in report.events:
            self.assertTrue(event.evidence_chain)
            for stage in event.evidence_chain:
                # Layer attribution and residuals are the execution evidence
                # that locates an anomaly at a specific anchor-pair stage.
                self.assertIn("background", stage.layer_attribution)
                self.assertIn("box", stage.layer_attribution)
                self.assertTrue(stage.stage)
                self.assertGreaterEqual(stage.active_tracks, 0)
                self.assertTrue(np.isfinite(stage.termination_concentration))

    def test_residual_alerts_are_bounded_per_event(self):
        _, _, _, report = _diagnose("similar")
        from dynamicsolver.layer_drift import MAX_DETAIL_PER_EVENT
        for event in report.events:
            self.assertLessEqual(len(event.residual_alert_track_ids),
                                 MAX_DETAIL_PER_EVENT)
        # The aggregate totals still count every alert even when detail is
        # truncated, so bounding the log never hides the anomaly population.
        counted = sum(len(ev.residual_alert_track_ids) for ev in report.events)
        self.assertLessEqual(counted, report.totals["n_residual_alerts"] + 1)

    def test_unresolved_ties_and_tracker_faults_remain_distinct(self):
        _, _, tracker, report = _diagnose("similar")
        # Observational unresolved ties are recorded as corrections with no
        # applied layer; implementation faults are a separate list and must be
        # empty on the deterministic synthetic clip.
        self.assertGreater(len(tracker.unresolved), 0)
        for corr in tracker.unresolved:
            self.assertFalse(corr.decided)
            self.assertIsNone(corr.applied_layer)
        self.assertEqual(tracker.tracker_faults, [])
        self.assertEqual(report.totals["n_tracker_faults"], 0)
        self.assertEqual(report.totals["n_unresolved"], len(tracker.unresolved))

    def test_log_lines_are_linear_in_events_not_tracks(self):
        cfg, dec, _, report = _diagnose("similar")
        lines = report.log_lines()
        # A handful of aggregate + bounded stage lines regardless of the
        # hundreds of tracks in the clip.
        self.assertLess(len(lines), 4 * len(report.events) + 20)
        self.assertTrue(all(isinstance(line, str) for line in lines))

    def test_diagnostics_never_read_ground_truth_files(self):
        # Import-time guard: the diagnostic module must not depend on the
        # evaluation/synth GT loaders.
        import dynamicsolver.layer_drift as mod
        with open(mod.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("import evaluate", source)


if __name__ == "__main__":
    unittest.main()
