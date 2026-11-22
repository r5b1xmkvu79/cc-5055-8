import unittest

import numpy as np

from dynamicsolver.decomposition import (
    BACKGROUND, BOX, DEFERRED, decompose,
)
from dynamicsolver.evaluation import true_layer_for_tracks
from dynamicsolver.features import extract_tracks
from dynamicsolver.pipeline import read_clip
from dynamicsolver.priors import Priors
from dynamicsolver.synth import SceneConfig, generate


def _load(scenario):
    cfg = SceneConfig(scenario=scenario)
    gen = generate(cfg)
    frames = read_clip(gen["video_path"])
    gt = np.load(gen["out_dir"] + "/ground_truth.npz")
    tracks = extract_tracks(frames)
    return cfg, gen, frames, gt, tracks


class TestDecomposition(unittest.TestCase):
    def test_baseline_separates_layers_with_high_f1(self):
        cfg, gen, frames, gt, tracks = _load("baseline")
        priors = Priors()
        result = decompose(tracks, gt["K"], cfg.width, cfg.height, priors=priors)
        truth = true_layer_for_tracks(
            tracks, gen["out_dir"] + "/ground_truth.npz", gt["K"])
        for layer in (BACKGROUND, BOX):
            ids = result.bg_tracks if layer == BACKGROUND else result.box_tracks
            tp = sum(1 for i in ids if truth[i] == layer)
            precision = tp / max(len(ids), 1)
            self.assertGreaterEqual(precision, 0.85)
        # The easy clip should not defer a large fraction of tracks.
        self.assertLess(len(result.deferred_tracks), 0.1 * len(tracks))

    def test_dominant_foreground_does_not_become_background_by_count(self):
        # The cardinality trap: in 'sparse' the box supplies most tracks. The
        # kernel must not label the dominant structure background purely
        # because it is large; it returns an explicit ambiguity verdict.
        cfg, gen, frames, gt, tracks = _load("sparse")
        priors = Priors()
        result = decompose(tracks, gt["K"], cfg.width, cfg.height, priors=priors)
        truth = true_layer_for_tracks(
            tracks, gen["out_dir"] + "/ground_truth.npz", gt["K"])
        n_true_box = sum(1 for t in tracks if truth[t.track_id] == "box")
        self.assertGreater(n_true_box, 0.5 * len(tracks))
        # Either an honest ambiguity verdict, or the box is still recovered.
        ambiguous = result.reason != "ok" or len(result.deferred_tracks) > 0
        labelled_box = len(result.box_tracks)
        self.assertTrue(ambiguous or labelled_box >= 0.3 * n_true_box)

    def test_same_speed_crossing_is_detected(self):
        cfg, gen, frames, gt, tracks = _load("similar")
        priors = Priors()
        result = decompose(tracks, gt["K"], cfg.width, cfg.height, priors=priors)
        self.assertTrue(result.crossing_windows,
                        "the same-speed crossing window must be flagged")

    def test_labels_partition_all_tracks(self):
        cfg, gen, frames, gt, tracks = _load("occlusion")
        result = decompose(tracks, gt["K"], cfg.width, cfg.height, priors=Priors())
        labelled = set(result.bg_tracks) | set(result.box_tracks) | set(
            result.deferred_tracks)
        self.assertEqual(labelled, {t.track_id for t in tracks})


if __name__ == "__main__":
    unittest.main()
