"""Motion-layer tracker: multi-hypothesis identity chains across same-speed windows.

The anchor-pair decomposition in :mod:`dynamicsolver.decomposition` votes a
layer label per track, but during a same-speed crossing two physical entities
(the moving box and a static landmark/background patch) have indistinguishable
image motion, so their *identity chains* can be exchanged. This module does not
re-derive geometry; it reuses the decomposition's per-window hard evidence and
adds the one thing a single best label cannot represent: competing hypotheses
with explicit support, kept until later observations make them separable.

Design rules (from the brief):

* Only observations are read here: KLT pixel trajectories and the
  decomposition's own window labels / structures. No ground-truth or hidden
  scene labels are ever imported, so the same code runs on real clips.
* Two swap hypotheses are *replayed from the frame immediately before the
  same-speed window*. Each hypothesis carries a replayable trajectory fragment
  and a support score. The wrong explanation is rejected by observation-only
  evidence: the two layers' shared per-frame flow fields over the crossing
  span. A track belonging to one layer follows that layer's median step
  vectors; a stitched identity does not, so its RMS distance to the wrong
  field is larger.
* Every identity change is a :class:`IdentityCorrection`, i.e. reversible: the
  pre-correction assignment, both hypotheses, the decision evidence and the
  inverse operation are all recorded. Applying ``inverse`` in reverse order
  restores the original decomposition labels exactly.
* "Observations cannot decide" (both flow-field distances stay within the
  decision margin) is kept distinct from "the implementation dropped its
  state" (a required hypothesis or fragment is missing). The former is an
  unresolved ambiguity, the latter a tracker fault — never papered over with a
  default label.

The strategy choice (offline chain reconnection vs. online multiple-hypothesis
retention) is measured, not assumed: :func:`benchmark_strategies` runs both on
the same observations and reports decision counts, unresolved ambiguities and
runtime. ``docs/layer_identity_drift.md`` records the empirical outcome.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .decomposition import BACKGROUND, BOX, DEFERRED, DecompositionResult
from .features import Track

HARD_LAYERS = (BACKGROUND, BOX)


@dataclass
class LayerHypothesis:
    """One candidate identity for a track fragment across a crossing.

    ``layer`` is the layer the fragment would be assigned under this
    hypothesis; ``support`` is the accumulated observation score; ``trajectory``
    is the replayable fragment (frame -> pixel position) the score was computed
    from, so every decision can be re-run rather than trusted.
    """

    layer: str
    support: float
    trajectory: dict                      # frame -> (u, v)
    geometry_residual_px: float
    evidence: dict = field(default_factory=dict)

    def replay(self):
        """Replay the stored fragment deterministically (sorted positions)."""
        return [(f, np.asarray(self.trajectory[f], dtype=np.float64))
                for f in sorted(self.trajectory)]


@dataclass
class IdentityCorrection:
    """Reversible record of one identity decision at a same-speed window.

    ``applied_layer`` is ``None`` when the observations could not separate the
    hypotheses: the track stays deferred and the record is marked unresolved.
    ``inverse`` restores the exact pre-event assignment, so the whole correction
    log can be rolled back.
    """

    track_id: int
    window: tuple                         # (f0, f1) same-speed anchor window
    original_layer: str
    applied_layer: str | None
    hypotheses: list                      # list[LayerHypothesis]
    reason: str
    bend_deg: float
    flow_distance_px: dict                # layer -> RMS distance to that layer field
    decided: bool
    inverse: dict

    def rollback(self, assignments: dict) -> None:
        assignments[self.track_id] = self.inverse["layer"]


@dataclass
class LayerTrackerResult:
    assignments: dict                     # track_id -> layer (may stay DEFERRED)
    corrections: list                     # list[IdentityCorrection]
    chains: dict                          # track_id -> ordered per-window layer chain
    candidates: dict                      # track_id -> list[LayerHypothesis] kept
    unresolved: list                      # corrections whose evidence was insufficient
    tracker_faults: list                  # missing-state events, distinct from ambiguity
    strategy: str
    elapsed_s: float
    n_windows: int

    def rollback(self) -> dict:
        """Return the exact pre-tracker assignment map by applying inverses."""
        restored = dict(self.assignments)
        for corr in reversed(self.corrections):
            corr.rollback(restored)
        return restored


# --------------------------------------------------------------------------- #
# Observation-only geometry helpers
# --------------------------------------------------------------------------- #
def _track_fragment(track: Track, f0: int, f1: int) -> dict:
    """Observed positions of one track on the inclusive frame span [f0, f1]."""
    return {f: np.asarray(uv, dtype=np.float64)
            for f, uv in zip(track.frames, track.uvs) if f0 <= f <= f1}


def _step_vectors(track: Track, f0: int, f1: int) -> dict:
    """Per-frame 2D displacement vectors for consecutive frames in [f0, f1)."""
    out = {}
    for i in range(len(track.frames) - 1):
        a, b = track.frames[i], track.frames[i + 1]
        if b == a + 1 and f0 <= a < f1:
            out[a] = np.asarray(track.uvs[i + 1] - track.uvs[i], dtype=np.float64)
    return out


def _mean_direction(positions: list[np.ndarray]) -> np.ndarray | None:
    dirs = []
    for a, b in zip(positions[:-1], positions[1:]):
        d = b - a
        n = np.linalg.norm(d)
        if n > 1e-9:
            dirs.append(d / n)
    if not dirs:
        return None
    return np.mean(dirs, axis=0)


def _bend_deg(track: Track, f_lo: int, f_mid: int, f_hi: int) -> float:
    """Angle between the flow directions before and after a crossing window.

    Logged as corroborating evidence; non-collinearity alone never decides,
    because a genuinely moving foreground can itself turn sharply.
    """
    pre = [uv for f, uv in zip(track.frames, track.uvs) if f_lo <= f <= f_mid]
    post = [uv for f, uv in zip(track.frames, track.uvs) if f_mid <= f <= f_hi]
    d0, d1 = _mean_direction(pre), _mean_direction(post)
    if d0 is None or d1 is None:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(d0, d1), -1.0, 1.0))))


def _segment_residual(positions: list[np.ndarray]) -> float:
    """RMS distance of observed positions from their best-fit line (pixels)."""
    if len(positions) < 3:
        return 0.0
    pts = np.asarray(positions, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    perp = centered - np.outer(centered @ direction, direction)
    return float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1))))


def _window_residual(track: Track, f0: int, f1: int) -> float:
    frag = [np.asarray(uv, dtype=np.float64)
            for f, uv in zip(track.frames, track.uvs) if f0 <= f <= f1]
    return _segment_residual(frag)


def _fit_velocity(positions: list[tuple[int, np.ndarray]]):
    """Constant-velocity fit p(f) = p0 + v*f over (frame, position) samples."""
    if len(positions) < 2:
        return None
    frames = np.asarray([f for f, _ in positions], dtype=np.float64)
    pts = np.asarray([p for _, p in positions], dtype=np.float64)
    A = np.column_stack([np.ones_like(frames), frames])
    coef, *_ = np.linalg.lstsq(A, pts, rcond=None)
    return coef[0], coef[1]


def _prediction_residual(track: Track, fit_frames: tuple, eval_frames: tuple) -> float:
    """Fit constant velocity on ``fit_frames``; RMS prediction error on
    ``eval_frames``. Corroborating (not deciding) evidence: acceleration makes
    the absolute value large for both layers, so it is only compared within a
    single track's two replay directions.
    """
    fit = [(f, np.asarray(uv, dtype=np.float64))
           for f, uv in zip(track.frames, track.uvs)
           if fit_frames[0] <= f <= fit_frames[1]]
    ev = [(f, np.asarray(uv, dtype=np.float64))
          for f, uv in zip(track.frames, track.uvs)
          if eval_frames[0] <= f <= eval_frames[1]]
    fitted = _fit_velocity(fit)
    if fitted is None or len(ev) < 2:
        return np.nan
    p0, v = fitted
    errs = [np.linalg.norm((p0 + v * float(f)) - uv) for f, uv in ev]
    return float(np.sqrt(np.mean(np.square(errs))))


# --------------------------------------------------------------------------- #
# Layer flow fields over the crossing span
# --------------------------------------------------------------------------- #
def _cohort_flow_field(by_id: dict, track_ids: list, f0: int, f1: int) -> dict:
    """Median per-frame displacement of a layer cohort over [f0, f1).

    Static-background points share one image motion field (camera-induced), and
    rigid box points share another; the per-frame median is robust to a minority
    of mislabelled cohort members, which matters precisely at a crossing where
    the input labels are already suspect.
    """
    per_frame: dict[int, list[np.ndarray]] = {}
    for tid in track_ids:
        track = by_id.get(tid)
        if track is None:
            continue
        for f, vec in _step_vectors(track, f0, f1).items():
            per_frame.setdefault(f, []).append(vec)
    return {f: np.median(vecs, axis=0) for f, vecs in per_frame.items()}


def _distance_to_field(track: Track, field: dict, f0: int, f1: int) -> float:
    """RMS distance between a track's step vectors and a layer flow field."""
    if not field:
        return np.nan
    errs = [float(np.linalg.norm(vec - field[f]))
            for f, vec in _step_vectors(track, f0, f1).items() if f in field]
    if not errs:
        return np.nan
    return float(np.sqrt(np.mean(np.square(errs))))


# --------------------------------------------------------------------------- #
# Multi-hypothesis reconnection
# --------------------------------------------------------------------------- #
def _ordered_windows(decomp: DecompositionResult) -> list:
    return sorted({(p.f0, p.f1) for p in decomp.pair_evidence},
                  key=lambda w: w[0])


def _neighbour_windows(windows: list, window: tuple):
    idx = windows.index(window)
    pre = windows[idx - 1] if idx > 0 else None
    post = windows[idx + 1] if idx + 1 < len(windows) else None
    return pre, post


def _candidate_tracks(decomp: DecompositionResult, pre_win, post_win) -> set:
    """Tracks whose hard identity is at risk across one same-speed window.

    Scope is deliberately narrow: a track enters the replay only when the
    decomposition's own per-window evidence flips background<->box across the
    window, or when the crossing window forced the track to deferred. Tracks
    whose hard label stays consistent are not touched.
    """
    out = set()
    for tid, lab in decomp.labels.items():
        pre = lab.window_labels.get(pre_win)
        post = lab.window_labels.get(post_win)
        if pre in HARD_LAYERS and post in HARD_LAYERS and pre != post:
            out.add(tid)
        elif pre in HARD_LAYERS and lab.layer == DEFERRED:
            out.add(tid)
    return out


def _build_hypotheses(track: Track, window: tuple, pre_win, post_win,
                      decomp: DecompositionResult, flow_fields: dict):
    """Replay the hold/swap hypotheses from the frame before the window.

    Both hypotheses carry the same observed fragment; they differ in the layer
    identity attached to it. Support is observation-only and combines:

      * RMS distance to each layer's median flow field over the crossing span
        (the deciding term: a track follows the field of its true layer),
      * constant-velocity prediction residual in both replay directions,
      * the decomposition's own pre/post-window hard labels (bounded
        corroboration; never enough to override the flow-field contrast).
    """
    f0, f1 = window
    lab = decomp.labels.get(track.track_id)
    pre_layer = lab.window_labels.get(pre_win)
    if pre_layer not in HARD_LAYERS:
        return None
    other_layer = BOX if pre_layer == BACKGROUND else BACKGROUND
    post_layer = lab.window_labels.get(post_win)
    trajectory = _track_fragment(track, pre_win[0], post_win[1])
    bend = _bend_deg(track, pre_win[0], f0, post_win[1])
    hold_predict = _prediction_residual(track, pre_win, post_win)
    swap_predict = _prediction_residual(track, post_win, pre_win)
    resid_pre = _window_residual(track, pre_win[0], pre_win[1])
    resid_post = _window_residual(track, post_win[0], post_win[1])
    dist = {layer: _distance_to_field(track, field, f0, f1)
            for layer, field in flow_fields.items()}
    if any(not np.isfinite(d) for d in dist.values()):
        return None

    def make(layer: str) -> LayerHypothesis:
        own = dist[layer]
        other = dist[BOX if layer == BACKGROUND else BACKGROUND]
        # Closer flow field -> higher support; the exponential kernel keeps the
        # term bounded while preserving a strong contrast across a real gap.
        field_term = float(np.exp(-0.5 * own) - 0.5 * np.exp(-0.5 * other))
        predict_term = 0.0
        if np.isfinite(hold_predict) and np.isfinite(swap_predict):
            own_predict = hold_predict if layer == pre_layer else swap_predict
            predict_term = 0.15 / (1.0 + own_predict)
        label_term = 0.1 * (1.0 if pre_layer == layer else 0.0) \
            + 0.2 * (1.0 if post_layer == layer else 0.0)
        support = field_term + predict_term + label_term
        return LayerHypothesis(
            layer=layer, support=float(support), trajectory=dict(trajectory),
            geometry_residual_px=float(own),
            evidence=dict(pre_window=list(pre_win), post_window=list(post_win),
                          pre_label=pre_layer, post_label=post_layer,
                          bend_deg=bend, flow_distance_px=dict(dist),
                          hold_prediction_residual_px=hold_predict,
                          swap_prediction_residual_px=swap_predict,
                          residual_pre_px=resid_pre,
                          residual_post_px=resid_post),
        )

    hold = make(pre_layer)
    swap = make(other_layer)
    return pre_layer, post_layer, bend, dist, hold, swap


def track_layers(tracks: list[Track], decomp: DecompositionResult,
                 strategy: str = "offline",
                 decision_margin_px: float = 4.0,
                 min_field_cohort: int = 6) -> LayerTrackerResult:
    """Reconnect layer identity chains across every detected same-speed window.

    Parameters
    ----------
    strategy:
        ``offline`` (default) keeps both hypotheses for every candidate until
        the clip ends and decides from the full replayed continuation.
        ``online`` commits the leading hypothesis at the first post-window
        anchor pair; an unresolved tie is frozen as deferred and never upgraded
        by later frames. It exists so :func:`benchmark_strategies` can measure
        the trade-off on identical observations.
    decision_margin_px:
        Minimum RMS flow-field distance difference required to commit a layer;
        below it the two explanations are observationally indistinguishable
        and the track stays deferred with both hypotheses retained.
    min_field_cohort:
        Minimum cohort size for a layer flow field to be admissible evidence.
    """
    t_start = time.perf_counter()
    if strategy not in ("offline", "online"):
        raise ValueError(f"unknown tracker strategy {strategy!r}")
    by_id = {t.track_id: t for t in tracks}
    windows = _ordered_windows(decomp)
    assignments = {tid: lab.layer for tid, lab in decomp.labels.items()}
    chains = {tid: dict(lab.window_labels) for tid, lab in decomp.labels.items()}
    corrections: list[IdentityCorrection] = []
    candidates: dict = {}
    unresolved: list[IdentityCorrection] = []
    faults: list[str] = []

    for window in list(decomp.crossing_windows):
        pre_win, post_win = _neighbour_windows(windows, window)
        if pre_win is None or post_win is None:
            # The clip ends before a post-crossing anchor pair exists: the
            # observations needed for a replay are absent, not the tracker's
            # state. Record an unresolved hold rather than an implementation
            # fault, and leave the decomposition labels untouched.
            if pre_win is not None:
                for tid, lab in decomp.labels.items():
                    if lab.window_labels.get(pre_win) in HARD_LAYERS:
                        corr = IdentityCorrection(
                            tid, window, lab.layer, None, [],
                            f"unresolved: crossing window {window} has no "
                            "post-window anchor pair in the clip; replay "
                            "evidence unavailable",
                            np.nan, {}, False, {"layer": lab.layer})
                        corrections.append(corr)
                        unresolved.append(corr)
            continue
        # Layer cohorts come from the pre-window hard labels, i.e. the state
        # immediately before the same-speed tie; the crossing labels themselves
        # are deferred and deliberately excluded from field construction.
        cohorts = {
            layer: [tid for tid, lab in decomp.labels.items()
                    if lab.window_labels.get(pre_win) == layer]
            for layer in HARD_LAYERS
        }
        thin = [layer for layer, ids in cohorts.items()
                if len(ids) < min_field_cohort]
        if thin:
            faults.append(f"window {window}: flow-field cohorts too small "
                          f"({ {k: len(v) for k, v in cohorts.items()} }); "
                          "no default label applied to candidate tracks")
            continue
        flow_fields = {layer: _cohort_flow_field(by_id, ids, *window)
                       for layer, ids in cohorts.items()}
        for tid in sorted(_candidate_tracks(decomp, pre_win, post_win)):
            track = by_id.get(tid)
            if track is None:
                faults.append(f"track {tid}: decomposition holds a label but the "
                              "track observation is missing from the input")
                continue
            built = _build_hypotheses(track, window, pre_win, post_win, decomp,
                                      flow_fields)
            if built is None:
                continue
            pre_layer, post_layer, bend, dist, hold_h, swap_h = built
            if not hold_h.trajectory or not swap_h.trajectory:
                # Implementation state loss, not an observational tie.
                faults.append(f"track {tid} window {window}: hypothesis replay "
                              "state is incomplete; no default label applied")
                continue
            candidates.setdefault(tid, []).extend([hold_h, swap_h])
            original = assignments[tid]
            d_bg, d_box = dist[BACKGROUND], dist[BOX]
            margin = d_box - d_bg  # >0: closer to background; <0: closer to box
            decided = abs(margin) >= decision_margin_px
            if not decided:
                applied, reason = DEFERRED, (
                    f"unresolved: flow-field distances bg={d_bg:.2f}px "
                    f"box={d_box:.2f}px within +/-{decision_margin_px}px; "
                    "both hypotheses retained")
                corr = IdentityCorrection(
                    tid, window, original, None, [hold_h, swap_h], reason, bend,
                    dict(dist), False, {"layer": original})
                unresolved.append(corr)
            elif margin > 0:
                applied = BACKGROUND
                reason = (f"assign background: follows the background flow field "
                          f"(d_bg={d_bg:.2f}px vs d_box={d_box:.2f}px, bend "
                          f"{bend:.1f} deg, support {hold_h.support:.3f}/"
                          f"{swap_h.support:.3f})")
                corr = IdentityCorrection(
                    tid, window, original, applied, [hold_h, swap_h], reason, bend,
                    dict(dist), True, {"layer": original})
            else:
                applied = BOX
                reason = (f"assign box: follows the box flow field "
                          f"(d_box={d_box:.2f}px vs d_bg={d_bg:.2f}px, bend "
                          f"{bend:.1f} deg, support {swap_h.support:.3f}/"
                          f"{hold_h.support:.3f})")
                corr = IdentityCorrection(
                    tid, window, original, applied, [hold_h, swap_h], reason, bend,
                    dict(dist), True, {"layer": original})
            corrections.append(corr)
            if strategy == "online":
                assignments[tid] = applied
                chains[tid][window] = applied
            else:
                chains[tid][window] = applied if decided else DEFERRED

    if strategy == "offline":
        for corr in corrections:
            assignments[corr.track_id] = (
                corr.applied_layer if corr.decided else DEFERRED)

    return LayerTrackerResult(
        assignments=assignments, corrections=corrections, chains=chains,
        candidates=candidates, unresolved=unresolved, tracker_faults=faults,
        strategy=strategy, elapsed_s=time.perf_counter() - t_start,
        n_windows=len(decomp.crossing_windows),
    )


def benchmark_strategies(tracks: list[Track], decomp: DecompositionResult,
                         repeats: int = 5) -> dict:
    """Measure offline vs. online reconnection on identical observations.

    Returns observation-only counters (no ground truth): decided corrections,
    unresolved ambiguities, tracker faults, flow-field margins and runtime.
    The strategy selection in the docs is taken from these measured numbers.
    """
    report = {}
    for strategy in ("offline", "online"):
        runs = [track_layers(tracks, decomp, strategy=strategy)
                for _ in range(max(repeats, 1))]
        ref = runs[0]
        bends = [c.bend_deg for c in ref.corrections if np.isfinite(c.bend_deg)]
        margins = [abs(c.flow_distance_px[BOX] - c.flow_distance_px[BACKGROUND])
                   for c in ref.corrections]
        report[strategy] = {
            "corrections": len(ref.corrections),
            "decided": sum(1 for c in ref.corrections if c.decided),
            "unresolved": len(ref.unresolved),
            "tracker_faults": len(ref.tracker_faults),
            "median_bend_deg": float(np.median(bends)) if bends else None,
            "median_flow_margin_px": float(np.median(margins)) if margins else None,
            "elapsed_s_mean": float(np.mean([r.elapsed_s for r in runs])),
            "elapsed_s_std": float(np.std([r.elapsed_s for r in runs])),
        }
    return report
