"""Declared priors used by the decomposition kernel.

Priors are *declared* inputs (documented assumptions), not hidden tuning
magic. The kernel reports how strongly each prior contributed and refuses to
label a layer when the evidence contradicts the priors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Priors:
    expected_layers: int = 2
    ransac_threshold_px: float = 1.5
    sigma_px: float = 0.75
    min_structure_tracks: int = 8
    max_ransac_iters: int = 400
    # Depth range in which static background is expected (metres). Used for
    # metric scale and as a soft check, never as a hard filter on features.
    # The midpoint anchors the SfM free scale to the same metric frame as the
    # declared foreground box size, so overlay projection is consistent.
    # Planar-dominant triangulation inflates raw depths, so this midpoint is
    # the true camera-to-background distance rather than the wall's z value.
    background_depth_range: tuple = (3.5, 4.5)
    # Declared foreground body size (metres). None = purely geometry-driven;
    # when given it is used only to audit/metric-scale the rigid box solve.
    box_size: np.ndarray | None = None
    # Evidence margins below which a frame pair is declared ambiguous instead
    # of being force-labelled.
    label_margin: float = 0.15
    gric_margin: float = 0.05
    # If a single RANSAC structure explains at least this fraction of a pair's
    # correspondences, the pair has no static/moving contrast and its layer
    # identity is deferred instead of being decided by cardinality (the
    # dominant-foreground trap).
    single_structure_defer_frac: float = 0.85
    # A track is reported deferred when its accumulated geometric deferral
    # weight exceeds this multiple of its combined background+box votes, so a
    # dominant geometric "unknown" is never overridden by supplementary flow.
    deferral_dominance: float = 1.2
    # Minimum accumulated background+box geometric weight for a track to be
    # considered "geometrically informed"; below this its deferral reflects
    # lack of anchor coverage rather than a genuine same-speed/dominant-layer
    # ambiguity, so supplementary priors are still allowed to label it.
    min_geometry_weight: float = 1.5
    # Crossing / deferral. The centroid bound tolerates estimation noise in the
    # cluster centre: a genuine same-speed overlap sits at ~0.12-0.13, while
    # truly separated layers stay beyond ~0.18 even when their speeds match.
    crossing_centroid_frac: float = 0.15
    crossing_velocity_frac: float = 0.25
    max_pair_gap: int = 6
    # Motion-magnitude prior (pixels/frame). A close rigid foreground object
    # moves substantially more in the image than the distant static background
    # under the same camera motion. The thresholds bracket the empirically
    # separable region: the central band is left uncommitted so overlap-zone
    # tracks are not force-labelled by flow alone. Declared rather than hidden.
    motion_flow_slow: float = 4.0
    motion_flow_fast: float = 10.0
    motion_flow_weight: float = 1.6
    # Depth evidence (metres, camera-frame z after metric scaling). After the
    # background camera is solved, every track is triangulated against it. A
    # static point lands near the declared background depth; a moving box point
    # triangulates to an inconsistent, generally nearer and high-residual 3D
    # point. This is the only evidence that survives the "same image speed"
    # case, where flow magnitude deliberately cannot separate the layers.
    depth_evidence_weight: float = 1.4
    depth_near_frac: float = 0.75       # below this x background depth -> foreground
    depth_far_frac: float = 0.9         # above this x background depth -> background
    depth_residual_px: float = 4.0      # multi-view reproj above this -> moving point

    def __post_init__(self):
        if self.box_size is not None:
            self.box_size = np.asarray(self.box_size, dtype=np.float64)
