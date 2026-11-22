"""Layer-identity drift diagnostics: an observation-only execution evidence chain.

The motion-layer tracker in :mod:`dynamicsolver.layer_tracker` decides what to
do when foreground and background share image speed. This module answers the
diagnostic question: *why* did layer identity drift at a given stage, and where
in the pipeline did the evidence become unreliable?

Hard constraints (from the brief):

* **No hidden labels.** Everything here is computed from the rendered clip's
  own observations: KLT tracks, decomposition structures/window labels and
  tracker flow-field distances. Ground-truth bundles are never opened, so the
  identical diagnostics run on real footage.
* **Causes are distinguished from evidence, not assumed.** Same-speed motion
  and crossing occlusion produce different observable signatures — continuous
  tracks with converging flow versus interrupted tracks with coverage loss —
  and a window is only given a cause when its observations support one.
* **Anomalies are localised to a stage.** Each event names the anchor-pair
  stage, the layer attribution recorded there and the geometry residuals, so a
  residual spike can be traced to the exact window that produced it.
* **Bounded log cost.** Per-window detail is capped; healthy windows contribute
  one aggregate line, and only anomalous tracks are expanded. The report size
  is O(number of crossing windows + anomalous tracks), never O(all tracks).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decomposition import BACKGROUND, BOX, DEFERRED, DecompositionResult
from .features import Track
from .layer_tracker import IdentityCorrection, LayerTrackerResult

SAME_SPEED = "same_speed_motion"
OCCLUSION = "crossing_occlusion"
MIXED = "mixed_same_speed_and_occlusion"
UNSUPPORTED = "cause_observationally_unsupported"

# Bounds keeping the diagnostic log linear in the number of *anomalies*, not in
# the (potentially large) track population.
MAX_DETAIL_PER_EVENT = 12
RESIDUAL_ALERT_px = 6.0
# A 4x4 image tile holding at least this share of a span's track terminations is
# a spatially localised coverage loss — the occlusion signature. Uniform KLT
# expiry spreads over all 16 tiles (~0.06 each); measured occlusion spans reach
# 0.22+, same-speed spans stay below 0.20.
OCCLUSION_TILE_FRAC = 0.21


@dataclass
class StageEvidence:
    """Observation record for one anchor-pair stage inside a drift event."""

    window: tuple
    stage: str
    n_correspondences: int
    layer_attribution: dict               # layer -> track count at this stage
    centroid_separation_frac: float
    speed_ratio: float
    geometry_residual_px: dict            # layer -> median inlier residual
    active_tracks: int
    interrupted_tracks: int               # reappearing tracks with an internal gap
    new_tracks: int                       # tracks first observed in this span
    termination_concentration: float      # densest 4x4 image tile's share of ends
    notes: list = field(default_factory=list)


@dataclass
class DriftEvent:
    """One bounded, reversible-linkable drift diagnosis for a crossing window."""

    window: tuple
    cause: str
    confidence: str                       # "supported" | "weak" | "unsupported"
    evidence_chain: list                  # list[StageEvidence]
    affected_track_ids: list
    correction_track_ids: list
    unresolved_track_ids: list
    residual_alert_track_ids: list
    summary: str

    def as_log_line(self) -> str:
        """Single-line aggregate; the cheap default logging path."""
        return (f"drift window {self.window} cause={self.cause} "
                f"confidence={self.confidence} affected={len(self.affected_track_ids)} "
                f"corrected={len(self.correction_track_ids)} "
                f"unresolved={len(self.unresolved_track_ids)} "
                f"residual_alerts={len(self.residual_alert_track_ids)} :: {self.summary}")


@dataclass
class DriftReport:
    events: list
    tracker_faults: list
    totals: dict
    bounded: bool                         # True when detail truncation occurred

    def log_lines(self) -> list:
        """Aggregate lines plus bounded detail; safe for CI/GUI text panels."""
        lines = [ev.as_log_line() for ev in self.events]
        for ev in self.events:
            for stage in ev.evidence_chain:
                lines.append(
                    f"  stage {stage.stage} {stage.window}: attr="
                    f"{stage.layer_attribution} sep={stage.centroid_separation_frac:.3f} "
                    f"speed_ratio={stage.speed_ratio:.3f} "
                    f"resid_px={ {k: round(v, 2) for k, v in stage.geometry_residual_px.items()} } "
                    f"interrupted={stage.interrupted_tracks} new={stage.new_tracks} "
                    f"end_tile={stage.termination_concentration:.2f}")
            for tid in ev.residual_alert_track_ids[:MAX_DETAIL_PER_EVENT]:
                lines.append(f"  residual-alert track {tid} (detail capped at "
                             f"{MAX_DETAIL_PER_EVENT}/event)")
        if self.tracker_faults:
            lines.append("tracker_faults: " + "; ".join(self.tracker_faults))
        return lines


# --------------------------------------------------------------------------- #
# Observation-only statistics
# --------------------------------------------------------------------------- #
def _structure_stats(pair) -> tuple[float, float, dict, dict]:
    """Centroid separation, speed ratio, attribution and residuals from a pair.

    All quantities come from the decomposition's own fitted structures — no
    scene ground truth is consulted.
    """
    structures = pair.structures
    attribution = {BACKGROUND: 0, BOX: 0, DEFERRED: 0}
    residuals: dict[str, list[float]] = {BACKGROUND: [], BOX: []}
    for s in structures:
        attribution[s.label] = attribution.get(s.label, 0) + len(s.track_ids)
        if s.label in residuals and len(s.residuals):
            residuals[s.label].extend(np.asarray(s.residuals, dtype=float).tolist())
    if len(structures) >= 2:
        a, b = structures[:2]
        # Width normalisation is unavailable on the pair alone; callers pass
        # image width through the stage builder, so store a raw separation and
        # normalise there.
        sep = float(np.linalg.norm(a.centroid - b.centroid))
        va = float(np.linalg.norm(a.mean_flow))
        vb = float(np.linalg.norm(b.mean_flow))
        speed_ratio = min(va, vb) / max(max(va, vb), 1e-9)
    else:
        sep, speed_ratio = np.nan, np.nan
    med_res = {k: float(np.median(v)) for k, v in residuals.items() if v}
    return sep, speed_ratio, attribution, med_res


def _continuity_stats(tracks: list[Track], f0: int, f1: int,
                      width: int, height: int) -> tuple[int, int, int, float]:
    """Active / interrupted / newly-born tracks plus termination concentration.

    An interruption is a track that disappears *inside* the span and reappears
    later (a frame gap in an otherwise surviving chain) — the observable
    signature of occlusion. Tracks that simply end are ordinary KLT expiry and
    are not counted as interruptions, since they dominate every long window.
    Their *spatial* concentration is still reported: occlusion ends cluster in
    one image tile, ordinary expiry spreads across the frame.
    """
    active, interrupted, new = 0, 0, 0
    end_pts = []
    for t in tracks:
        frames = t.frames
        visible = [f for f in frames if f0 <= f <= f1]
        if not visible:
            continue
        active += 1
        if t.start >= f0:
            new += 1
        if f0 <= t.end <= f1:
            end_pts.append(np.asarray(t.uvs[-1], dtype=np.float64))
        for i in range(len(frames) - 1):
            if f0 <= frames[i] <= f1 and frames[i + 1] - frames[i] > 1:
                interrupted += 1
                break
    concentration = 0.0
    if end_pts:
        pts = np.asarray(end_pts)
        gx = np.clip((pts[:, 0] / max(width / 4.0, 1.0)).astype(int), 0, 3)
        gy = np.clip((pts[:, 1] / max(height / 4.0, 1.0)).astype(int), 0, 3)
        grid = np.zeros((4, 4), dtype=int)
        for x, y in zip(gx, gy):
            grid[y, x] += 1
        concentration = float(grid.max() / len(end_pts))
    return active, interrupted, new, concentration


def _pair_by_window(decomp: DecompositionResult) -> dict:
    return {(p.f0, p.f1): p for p in decomp.pair_evidence}


def _classify_cause(speed_ratio: float, sep_frac: float,
                    interrupt_frac: float,
                    termination_concentration: float) -> tuple[str, str, str]:
    """Return (cause, confidence, summary) from observation thresholds."""
    speed_locked = np.isfinite(speed_ratio) and speed_ratio > 0.75
    overlap = np.isfinite(sep_frac) and sep_frac < 0.20
    occlusion = (interrupt_frac > 0.10
                 or termination_concentration >= OCCLUSION_TILE_FRAC)
    if speed_locked and overlap and occlusion:
        return MIXED, "supported", (
            f"matched speeds (ratio {speed_ratio:.2f}) and spatial overlap "
            f"(sep {sep_frac:.2f}) with localised coverage loss "
            f"(tile concentration {termination_concentration:.2f}) indicate "
            "same-speed motion compounded by occlusion")
    if speed_locked and overlap:
        return SAME_SPEED, "supported", (
            f"matched speeds (ratio {speed_ratio:.2f}) under spatial overlap "
            f"(sep {sep_frac:.2f}) with continuous, spatially spread tracks; "
            "identity is observationally indistinguishable while the lock holds")
    if occlusion:
        return OCCLUSION, "supported", (
            f"localised coverage loss (tile concentration "
            f"{termination_concentration:.2f}, interrupted {interrupt_frac:.0%}) "
            "drives the drift; speed matching is not observed")
    return UNSUPPORTED, "weak", (
        "neither speed locking nor occlusion coverage loss reaches the "
        "observation thresholds; cause cannot be asserted from observations")


# --------------------------------------------------------------------------- #
# Report construction
# --------------------------------------------------------------------------- #
def diagnose_drift(tracks: list[Track], decomp: DecompositionResult,
                   tracker: LayerTrackerResult, width: int,
                   height: int) -> DriftReport:
    """Build the bounded execution evidence chain from observations only."""
    pair_index = _pair_by_window(decomp)
    windows = sorted(pair_index, key=lambda w: w[0])
    corrections_by_window: dict[tuple, list[IdentityCorrection]] = {}
    for corr in tracker.corrections:
        corrections_by_window.setdefault(corr.window, []).append(corr)

    events: list[DriftEvent] = []
    residual_alerts_all: list[int] = []
    bounded = False
    handled_windows = set()

    for window in list(decomp.crossing_windows):
        handled_windows.add(window)
        idx = windows.index(window) if window in windows else -1
        chain: list[StageEvidence] = []
        # The event covers the stage before, the crossing stage and the stage
        # after, so drift can be localised rather than blamed on the clip.
        span_windows = [w for w in windows[max(0, idx - 1): idx + 2]]
        for stage_idx, w in enumerate(span_windows):
            pair = pair_index[w]
            sep_raw, speed_ratio, attribution, med_res = _structure_stats(pair)
            active, interrupted, new, concentration = _continuity_stats(
                tracks, *w, width, height)
            chain.append(StageEvidence(
                window=w,
                stage=("pre-crossing" if w == span_windows[0] and w != window
                       else "crossing" if w == window else "post-crossing"),
                n_correspondences=pair.n_corr,
                layer_attribution=attribution,
                centroid_separation_frac=sep_raw / max(width, 1),
                speed_ratio=speed_ratio,
                geometry_residual_px=med_res,
                active_tracks=active,
                interrupted_tracks=interrupted,
                new_tracks=new,
                termination_concentration=concentration,
                notes=list(pair.reasons[:3]),
            ))
        crossing_stage = next((s for s in chain if s.window == window), None)
        if crossing_stage is None:
            continue
        interrupt_frac = (crossing_stage.interrupted_tracks
                          / max(crossing_stage.active_tracks, 1))
        cause, confidence, summary = _classify_cause(
            crossing_stage.speed_ratio,
            crossing_stage.centroid_separation_frac,
            interrupt_frac,
            crossing_stage.termination_concentration,
        )

        corrs = corrections_by_window.get(window, [])
        affected = sorted(tid for tid, lab in decomp.labels.items()
                          if window in lab.window_labels)
        decided_ids = [c.track_id for c in corrs if c.decided]
        unresolved_ids = [c.track_id for c in corrs if not c.decided]
        # Residual alerts: tracker flow-field distances exceeding the bounded
        # pixel threshold on either layer. Only these tracks justify detail.
        alert_ids = sorted({
            c.track_id for c in corrs
            if max(c.flow_distance_px.values(), default=0.0) >= RESIDUAL_ALERT_px
        })
        if len(alert_ids) > MAX_DETAIL_PER_EVENT:
            bounded = True
        residual_alerts_all.extend(alert_ids)

        events.append(DriftEvent(
            window=window, cause=cause, confidence=confidence,
            evidence_chain=chain,
            affected_track_ids=affected,
            correction_track_ids=decided_ids,
            unresolved_track_ids=unresolved_ids,
            residual_alert_track_ids=alert_ids[:MAX_DETAIL_PER_EVENT],
            summary=summary,
        ))

    # Occlusion-only scan: a clip can lose layer coverage behind an occluder
    # without any same-speed crossing (the pillar scenario). The same bounded
    # stage evidence is built from anchor windows whose termination
    # concentration crosses the occlusion threshold. The clip's first anchor
    # window is excluded: initial redetection produces a burst of co-located
    # births whose maturity mimics concentration but is ordinary turnover.
    first_window = windows[0] if windows else None
    for w in windows:
        if w in handled_windows or w == first_window:
            continue
        active, interrupted, new, concentration = _continuity_stats(
            tracks, *w, width, height)
        birth_dominated = new > 0.5 * max(active, 1)
        if birth_dominated or concentration < OCCLUSION_TILE_FRAC:
            continue
        pair = pair_index[w]
        sep_raw, speed_ratio, attribution, med_res = _structure_stats(pair)
        stage = StageEvidence(
            window=w, stage="occlusion-scan",
            n_correspondences=pair.n_corr,
            layer_attribution=attribution,
            centroid_separation_frac=sep_raw / max(width, 1),
            speed_ratio=speed_ratio,
            geometry_residual_px=med_res,
            active_tracks=active, interrupted_tracks=interrupted,
            new_tracks=new, termination_concentration=concentration,
            notes=list(pair.reasons[:3]))
        cause, confidence, summary = _classify_cause(
            speed_ratio, stage.centroid_separation_frac,
            interrupted / max(active, 1), concentration)
        events.append(DriftEvent(
            window=w, cause=OCCLUSION if cause == UNSUPPORTED else cause,
            confidence=confidence, evidence_chain=[stage],
            affected_track_ids=sorted(
                t.track_id for t in tracks if w[0] <= t.end <= w[1]),
            correction_track_ids=[], unresolved_track_ids=[],
            residual_alert_track_ids=[],
            summary=summary))

    totals = {
        "n_events": len(events),
        "n_corrections": len(tracker.corrections),
        "n_decided": sum(1 for c in tracker.corrections if c.decided),
        "n_unresolved": len(tracker.unresolved),
        "n_residual_alerts": len(residual_alerts_all),
        "n_tracker_faults": len(tracker.tracker_faults),
        "causes": {cause: sum(1 for e in events if e.cause == cause)
                   for cause in (SAME_SPEED, OCCLUSION, MIXED, UNSUPPORTED)},
    }
    return DriftReport(events=events, tracker_faults=list(tracker.tracker_faults),
                       totals=totals, bounded=bounded)
