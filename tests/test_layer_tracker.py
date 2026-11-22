"""Tests for the multi-hypothesis motion-layer tracker.

The tracker itself reads only observations (KLT trajectories and the
decomposition's window labels). Ground truth is used solely in these tests to
check that the reconnected identities and the honest deferrals match reality;
nothing under ``dynamicsolver/`` imports GT for the decision.
"""
import unittest

import numpy as np

from dynamicsolver.decomposition import (
    BACKGROUND, BOX, DEFERRED, decompose,
)
from dynamicsolver.evaluation import identity_swaps, true_layer_for_tracks
from dynamicsolver.features import Track, extract_tracks
from dynamicsolver.layer_tracker import (
    LayerHypothesis, benchmark_strategies, track_layers,
)
from dynamicsolver.pipeline import read_clip
from dynamicsolver.priors import Priors
from dynamicsolver.synth import SceneConfig, generate


def _load(scenario):
    cfg = SceneConfig(scenario=scenario)
    gen = generate(cfg)
    frames = read_clip(gen["video_path"])
    gt = np.load(gen["out_dir"] + "/ground_truth.npz")
    tracks = extract_tracks(frames)
    dec = decompose(tracks, gt["K"], cfg.width, cfg.height, priors=Priors())
    truth = true_layer_for_tracks(
        tracks, gen["out_dir"] + "/ground_truth.npz", gt["K"])
    return cfg, gen, tracks, dec, truth


class TestLayerTracker(unittest.TestCase):
    def test_same_speed_window_produces_replayable_hypotheses(self):
        _, _, tracks, dec, _ = _load("similar")
        result = track_layers(tracks, dec, strategy="offline")
        self.assertTrue(dec.crossing_windows)
        self.assertTrue(result.corrections)
        for corr in result.corrections:
            self.assertEqual(len(corr.hypotheses), 2)
            layers = {h.layer for h in corr.hypotheses}
            self.assertEqual(layers, {BACKGROUND, BOX})
            for hyp in corr.hypotheses:
                self.assertIsInstance(hyp, LayerHypothesis)
                self.assertTrue(hyp.trajectory)
                # Replay must be deterministic and frame-sorted.
                replayed = hyp.replay()
                frames = [f for f, _ in replayed]
                self.assertEqual(frames, sorted(frames))

    def test_decided_corrections_match_ground_truth(self):
        _, _, tracks, dec, truth = _load("similar")
        result = track_layers(tracks, dec, strategy="offline")
        decided = [c for c in result.corrections if c.decided]
        self.assertGreater(len(decided), 0)
        for corr in decided:
            self.assertEqual(corr.applied_layer, truth[corr.track_id])

    def test_unresolved_tracks_keep_both_hypotheses_and_stay_deferred(self):
        _, _, tracks, dec, _ = _load("similar")
        result = track_layers(tracks, dec, strategy="offline")
        self.assertGreater(len(result.unresolved), 0)
        for corr in result.unresolved:
            self.assertFalse(corr.decided)
            self.assertIsNone(corr.applied_layer)
            self.assertEqual(len(corr.hypotheses), 2)
            self.assertEqual(result.assignments[corr.track_id], DEFERRED)

    def test_correction_log_is_fully_reversible(self):
        _, _, tracks, dec, _ = _load("similar")
        original = {tid: lab.layer for tid, lab in dec.labels.items()}
        result = track_layers(tracks, dec, strategy="offline")
        self.assertNotEqual(result.assignments, original)
        restored = result.rollback()
        self.assertEqual(restored, original)

    def test_offline_and_online_share_observation_counters(self):
        _, _, tracks, dec, _ = _load("similar")
        report = benchmark_strategies(tracks, dec, repeats=3)
        for strategy in ("offline", "online"):
            self.assertIn(strategy, report)
            self.assertEqual(report[strategy]["tracker_faults"], 0)
            self.assertGreater(report[strategy]["corrections"], 0)
        # Both strategies consume the identical observations, so their decided
        # counts must agree; only commit timing differs.
        self.assertEqual(report["offline"]["decided"],
                         report["online"]["decided"])
        self.assertEqual(report["offline"]["unresolved"],
                         report["online"]["unresolved"])

    def test_tracker_removes_reconnected_flips_from_swap_metric(self):
        _, _, tracks, dec, truth = _load("similar")
        raw_swaps, _ = identity_swaps(dec, None, truth, dec.crossing_windows)
        tracked = track_layers(tracks, dec, strategy="offline")
        # Reconnect without mutating dec: emulate the output partition.
        from dynamicsolver.pipeline import _apply_tracking
        _apply_tracking(dec, tracked)
        fixed_swaps, _ = identity_swaps(dec, None, truth, dec.crossing_windows,
                                        tracker_result=tracked)
        self.assertGreaterEqual(raw_swaps, fixed_swaps)
        self.assertEqual(fixed_swaps, 0)

    def test_no_crossing_means_identity_is_untouched(self):
        _, _, tracks, dec, _ = _load("occlusion")
        self.assertEqual(dec.crossing_windows, [])
        result = track_layers(tracks, dec, strategy="offline")
        self.assertEqual(result.corrections, [])
        original = {tid: lab.layer for tid, lab in dec.labels.items()}
        self.assertEqual(result.assignments, original)

    def test_repeated_runs_produce_identical_evidence(self):
        # Motion decomposition reruns must be bit-consistent: same hypotheses,
        # supports, flow distances and final assignments on every repeat.
        _, _, tracks, dec, _ = _load("similar")
        first = track_layers(tracks, dec, strategy="offline")
        for _ in range(3):
            again = track_layers(tracks, dec, strategy="offline")
            self.assertEqual(again.assignments, first.assignments)
            for c1, c2 in zip(first.corrections, again.corrections):
                self.assertEqual(c1.track_id, c2.track_id)
                self.assertEqual(c1.decided, c2.decided)
                self.assertEqual(c1.applied_layer, c2.applied_layer)
                self.assertEqual(c1.flow_distance_px, c2.flow_distance_px)
                for h1, h2 in zip(c1.hypotheses, c2.hypotheses):
                    self.assertEqual(h1.layer, h2.layer)
                    self.assertAlmostEqual(h1.support, h2.support, places=12)
                    self.assertEqual(set(h1.trajectory), set(h2.trajectory))
                    for f in h1.trajectory:
                        np.testing.assert_allclose(
                            h1.trajectory[f], h2.trajectory[f], rtol=0, atol=0)

    def test_synthetic_swap_pair_is_resolved_by_flow_fields(self):
        """Controlled observation-only fixture: two tracks exchange headings at
        a same-speed window. The layer flow fields built from each cohort must
        send each post-window fragment back to its own cohort."""
        rng = np.random.default_rng(3)
        n = 36
        frames = list(range(n))
        # Background cohort: steady rightward drift; box cohort: steady downward.
        bg_tracks, box_tracks = [], []
        for k in range(8):
            base = np.array([100.0 + 30 * k, 120.0 + 20 * k])
            uvs = [base + np.array([3.0 * f, 0.5 * f]) + rng.normal(0, 0.15, 2)
                   for f in frames]
            bg_tracks.append(Track(100 + k, list(frames), uvs))
        for k in range(8):
            base = np.array([300.0 + 12 * k, 300.0 + 8 * k])
            uvs = [base + np.array([0.4 * f, 7.0 * f]) + rng.normal(0, 0.15, 2)
                   for f in frames]
            box_tracks.append(Track(200 + k, list(frames), uvs))
        # Two swapped chains: bg-origin fragment that follows the box field
        # after frame 12, and vice versa.
        swap_a = Track(300, list(frames),
                       [np.array([200.0, 200.0]) + np.array([3.0 * f, 0.5 * f])
                        if f < 12 else
                        np.array([200.0, 200.0]) + np.array([36.0, 6.0])
                        + np.array([0.4 * (f - 12), 7.0 * (f - 12)])
                        for f in frames])
        swap_b = Track(301, list(frames),
                       [np.array([340.0, 340.0]) + np.array([0.4 * f, 7.0 * f])
                        if f < 12 else
                        np.array([340.0, 340.0]) + np.array([4.8, 84.0])
                        + np.array([3.0 * (f - 12), 0.5 * (f - 12)])
                        for f in frames])
        tracks = bg_tracks + box_tracks + [swap_a, swap_b]
        # Build a minimal decomposition-shaped object from the cohorts.
        from dynamicsolver.decomposition import (
            DecompositionResult, PairEvidence, TrackLabel,
        )
        labels = {}
        for t in bg_tracks:
            labels[t.track_id] = TrackLabel(
                t.track_id, BACKGROUND, 0.9, {},
                window_labels={(0, 12): BACKGROUND, (12, 24): BACKGROUND,
                               (24, 35): BACKGROUND})
        for t in box_tracks:
            labels[t.track_id] = TrackLabel(
                t.track_id, BOX, 0.9, {},
                window_labels={(0, 12): BOX, (12, 24): BOX, (24, 35): BOX})
        labels[300] = TrackLabel(
            300, DEFERRED, 0.5, {},
            window_labels={(0, 12): BACKGROUND, (12, 24): BOX,
                           (24, 35): BOX})
        labels[301] = TrackLabel(
            301, DEFERRED, 0.5, {},
            window_labels={(0, 12): BOX, (12, 24): BACKGROUND,
                           (24, 35): BACKGROUND})
        pairs = [PairEvidence(0, 12, len(tracks), [], False, [], False),
                 PairEvidence(12, 24, len(tracks), [], False, [], False),
                 PairEvidence(24, 35, len(tracks), [], False, [], False)]
        dec = DecompositionResult(
            ok=True, reason="ok", labels=labels,
            pair_evidence=pairs, crossing_windows=[(12, 24)],
            internal_label_flips=0,
            bg_tracks=[t.track_id for t in bg_tracks],
            box_tracks=[t.track_id for t in box_tracks],
            deferred_tracks=[300, 301], coverage_evidence={},
            warnings=[], params={})
        result = track_layers(tracks, dec, strategy="offline")
        self.assertEqual(result.assignments[300], BOX)
        self.assertEqual(result.assignments[301], BACKGROUND)

    def test_thin_cohorts_raise_fault_instead_of_default_label(self):
        """When neither layer has enough pre-window tracks to fit an admissible
        flow field, the tracker must report an implementation-side fault and
        leave every label untouched rather than masking the state loss."""
        rng = np.random.default_rng(11)
        n = 36
        frames = list(range(n))
        windows = [(0, 12), (12, 24), (24, 35)]
        bg_tracks, box_tracks = [], []
        for k in range(3):
            base = np.array([100.0 + 30 * k, 120.0 + 20 * k])
            uvs = [base + np.array([3.0 * f, 0.5 * f]) + rng.normal(0, 0.15, 2)
                   for f in frames]
            bg_tracks.append(Track(100 + k, list(frames), uvs))
        for k in range(3):
            base = np.array([300.0 + 12 * k, 300.0 + 8 * k])
            uvs = [base + np.array([0.4 * f, 7.0 * f]) + rng.normal(0, 0.15, 2)
                   for f in frames]
            box_tracks.append(Track(200 + k, list(frames), uvs))
        swap_a = Track(
            300, list(frames),
            [np.array([200.0, 200.0]) + np.array([3.0 * f, 0.5 * f])
             if f < 12 else
             np.array([200.0, 200.0]) + np.array([36.0, 6.0])
             + np.array([0.4 * (f - 12), 7.0 * (f - 12)])
             for f in frames])
        tracks = bg_tracks + box_tracks + [swap_a]
        from dynamicsolver.decomposition import (
            DecompositionResult, PairEvidence, TrackLabel,
        )
        labels = {}
        for t in bg_tracks:
            labels[t.track_id] = TrackLabel(
                t.track_id, BACKGROUND, 0.9, {},
                window_labels={w: BACKGROUND for w in windows})
        for t in box_tracks:
            labels[t.track_id] = TrackLabel(
                t.track_id, BOX, 0.9, {},
                window_labels={w: BOX for w in windows})
        labels[300] = TrackLabel(
            300, DEFERRED, 0.5, {},
            window_labels={windows[0]: BACKGROUND, windows[1]: BOX,
                           windows[2]: BOX})
        pairs = [PairEvidence(f0, f1, len(tracks), [], False, [], False)
                 for f0, f1 in windows]
        dec = DecompositionResult(
            ok=True, reason="ok", labels=labels, pair_evidence=pairs,
            crossing_windows=[(12, 24)], internal_label_flips=1,
            bg_tracks=[t.track_id for t in bg_tracks],
            box_tracks=[t.track_id for t in box_tracks],
            deferred_tracks=[300], coverage_evidence={}, warnings=[], params={})
        original = {tid: lab.layer for tid, lab in dec.labels.items()}
        result = track_layers(tracks, dec, strategy="offline",
                              min_field_cohort=6)
        self.assertTrue(result.tracker_faults)
        self.assertEqual(result.corrections, [])
        self.assertEqual(result.unresolved, [])
        self.assertEqual(result.assignments, original)
        self.assertEqual(result.assignments[300], DEFERRED)


if __name__ == "__main__":
    unittest.main()
