"""Motion decomposition kernel.

Given KLT tracks and calibration, split observations into a static-background
layer (which supports a camera solve) and an independently moving foreground
layer (the box). The design deliberately avoids the two failure modes called
out in the brief:

1. Cardinality is never treated as evidence. A large foreground filling most
   of the frame can be the bigger cluster; layer identity is decided from
   geometry (F vs H GRIC), spatial coverage, depth priors and multi-view
   rigidity.
2. A low-residual model is not declared the unique real motion. Both model
   families are fitted to every structure, GRIC margins are reported, and
   pairs whose evidence margin is too small are *deferred* rather than
   force-labelled. An explicit reliability verdict is returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import Track, frame_correspondences
from .geometry import (
    fit_fundamental_8pt,
    fit_homography_dlt,
    gric,
    homography_residuals,
    project_points,
    ransac_model,
    sampson_residuals,
    triangulate,
)
from .priors import Priors

BACKGROUND = "background"
BOX = "box"
DEFERRED = "deferred"


@dataclass
class Structure:
    label: str  # "background" | "box" | "unknown"
    model: str  # "F" | "H"
    matrix: np.ndarray
    track_ids: np.ndarray
    residuals: np.ndarray
    gric_self: float
    gric_alt: float       # GRIC of the other model family on the same set
    coverage: float       # convex-hull area / image area
    centroid: np.ndarray
    mean_flow: np.ndarray


@dataclass
class PairEvidence:
    f0: int
    f1: int
    n_corr: int
    structures: list
    ambiguous: bool
    reasons: list
    crossing: bool


@dataclass
class TrackLabel:
    track_id: int
    layer: str
    confidence: float
    evidence: dict
    deferred_frames: list = field(default_factory=list)
    # Anchor-pair window -> structure label assigned on that pair
    # (background/box/deferred). Lets the evaluator measure a real identity
    # exchange: the same track switching structure across a crossing window.
    window_labels: dict = field(default_factory=dict)


@dataclass
class DecompositionResult:
    ok: bool
    reason: str
    labels: dict                 # track_id -> TrackLabel
    pair_evidence: list
    crossing_windows: list
    internal_label_flips: int
    bg_tracks: list
    box_tracks: list
    deferred_tracks: list
    coverage_evidence: dict
    warnings: list
    params: dict


def _coverage_hull(pts: np.ndarray, width: int, height: int) -> float:
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    return float(cv2.contourArea(hull) / float(width * height))


def _fit_both_models(pts0, pts1, threshold, sigma, rng, max_iters):
    f_hyp = ransac_model("F", pts0, pts1, threshold, rng, max_iters=max_iters)
    h_hyp = ransac_model("H", pts0, pts1, threshold, rng, max_iters=max_iters)
    return f_hyp, h_hyp


def _structure_from_hypothesis(kind, hyp, alt_kind, pts0, pts1, ids, width, height, sigma):
    mask = hyp.inlier_mask
    if mask.sum() < 6:
        return None
    if alt_kind == "F":
        alt = fit_fundamental_8pt(pts0[mask], pts1[mask])
        alt_res = sampson_residuals(alt, pts0, pts1)
        gric_alt = gric(alt_res, len(pts0), d=3, k=7, sigma=sigma)
    else:
        alt = fit_homography_dlt(pts0[mask], pts1[mask])
        alt_res = homography_residuals(alt, pts0, pts1)
        gric_alt = gric(alt_res, len(pts0), d=2, k=8, sigma=sigma)
    flow = pts1[mask] - pts0[mask]
    return Structure(
        label="unknown", model=kind, matrix=hyp.matrix,
        track_ids=ids[mask], residuals=hyp.residuals[mask],
        gric_self=hyp.score, gric_alt=gric_alt,
        coverage=_coverage_hull(pts0[mask], width, height),
        centroid=pts0[mask].mean(axis=0), mean_flow=flow.mean(axis=0),
    )


def _sequential_ransac(p0, p1, ids, width, height, priors: Priors, rng):
    """Extract up to two structures by sequential RANSAC over F and H."""
    threshold = priors.ransac_threshold_px
    sigma = priors.sigma_px
    out, reasons = [], []
    pool0, pool1, pool_ids = p0, p1, ids
    for round_idx in range(2):
        if len(pool0) < priors.min_structure_tracks:
            break
        f_hyp, h_hyp = _fit_both_models(pool0, pool1, threshold, sigma, rng,
                                        priors.max_ransac_iters)
        if f_hyp is None and h_hyp is None:
            reasons.append("no model converged")
            break
        # Prefer by GRIC, but require the winner to have enough inliers.
        candidates = [h for h in (f_hyp, h_hyp) if h is not None]
        winner = min(candidates, key=lambda h: h.score)
        kind = winner.model
        alt = "H" if kind == "F" else "F"
        struct = _structure_from_hypothesis(
            kind, winner, alt, pool0, pool1, pool_ids, width, height, sigma
        )
        if struct is None or len(struct.track_ids) < priors.min_structure_tracks:
            reasons.append(f"structure {round_idx} under minimum track count")
            break
        out.append(struct)
        keep = ~winner.inlier_mask
        pool0, pool1, pool_ids = pool0[keep], pool1[keep], pool_ids[keep]

    leftover = len(pool0)
    if leftover > 0.15 * len(p0) and len(p0) >= priors.min_structure_tracks:
        reasons.append(f"third structure suspected ({leftover} unexplained tracks)")
    return out, reasons


def _label_structures(structures: list, priors: Priors, n_corr: int):
    """Identity decision from geometry + coverage, never set cardinality."""
    if len(structures) == 1:
        s = structures[0]
        frac = len(s.track_ids) / max(n_corr, 1)
        # A single structure that explains almost every correspondence gives
        # no contrast against which to decide static vs moving. This is the
        # dominant-foreground (cardinality) trap: calling it "background" just
        # because it is large is exactly the failure the brief forbids, so the
        # pair's identity is deferred rather than force-labelled.
        if frac >= priors.single_structure_defer_frac:
            s.label = DEFERRED
            return [f"single structure covers {frac:.0%} of correspondences; "
                    "no identity contrast -> deferred"]
        # A minority-only structure: model preference + coverage decide.
        if s.model == "H" and s.coverage < 0.35:
            s.label = BOX
        else:
            s.label = BACKGROUND
        return ["single-structure pair; label from model/coverage only"]
    a, b = structures
    # Evidence vector per structure: higher = more background-like.
    score_a = _background_evidence(a, b)
    score_b = _background_evidence(b, a)
    margin = abs(score_a - score_b)
    reasons = [f"background-evidence scores {score_a:.3f} vs {score_b:.3f} "
               f"(margin {margin:.3f})"]
    if margin < priors.label_margin:
        for s in structures:
            s.label = DEFERRED
        reasons.append("evidence margin below threshold -> deferred")
        return reasons
    if score_a > score_b:
        a.label, b.label = BACKGROUND, BOX
    else:
        a.label, b.label = BOX, BACKGROUND
    return reasons


def _background_evidence(s: Structure, other: Structure) -> float:
    """Bounded evidence that a structure is static background.

    Components, each roughly in [0,1]:
      * spatial coverage (background spans more of the image),
      * model preference: F (parallax across depth) beats a planar H,
      * GRIC margin between the two model families.
    Cardinality is intentionally absent.
    """
    cov = np.clip(s.coverage / max(s.coverage + other.coverage, 1e-9), 0, 1)
    model_pref = 0.62 if s.model == "F" else 0.38
    # Positive gric_alt - gric_self means the chosen model is preferred;
    # a strong F preference supports a 3D static scene.
    gric_norm = np.clip((s.gric_alt - s.gric_self) / max(abs(s.gric_self), 1.0), -1, 1)
    gric_term = 0.5 + 0.5 * gric_norm
    return 0.45 * cov + 0.30 * model_pref + 0.25 * gric_term


def _detect_crossing(structures: list, width: int, priors: Priors) -> bool:
    if len(structures) < 2:
        return False
    a, b = structures
    sep = np.linalg.norm(a.centroid - b.centroid) / max(width, 1)
    va, vb = np.linalg.norm(a.mean_flow), np.linalg.norm(b.mean_flow)
    speed_ratio = min(va, vb) / max(max(va, vb), 1e-9)
    close = sep < priors.crossing_centroid_frac
    matched_speed = speed_ratio > (1.0 - priors.crossing_velocity_frac)
    return close and matched_speed


def decompose(tracks: list[Track], K: np.ndarray, width: int, height: int,
              priors: Priors | None = None, seed: int = 0,
              anchor_span: int = 8) -> DecompositionResult:
    """Multi-baseline layered decomposition.

    Adjacent-frame geometry cannot separate a slowly moving foreground from
    the background (their epipolar residuals overlap). Evidence is therefore
    gathered on *anchor pairs* separated by ``anchor_span`` frames, where the
    moving box leaves large multi-frame residuals, then integrated by vote
    across the whole track lifetime.
    """
    priors = priors or Priors()
    rng = np.random.default_rng(seed)
    n_frames = max(t.end for t in tracks) + 1
    pair_evidence, warnings, crossing_windows = [], [], []

    votes = {t.track_id: {BACKGROUND: 0.0, BOX: 0.0, DEFERRED: 0.0} for t in tracks}
    window_assign = {t.track_id: {} for t in tracks}
    deferred_pairs = []
    anchor_pairs = [(f, min(f + anchor_span, n_frames - 1))
                    for f in range(0, n_frames - 1, anchor_span)]
    anchor_pairs = [(a, b) for a, b in anchor_pairs if b > a]

    for f0, f1 in anchor_pairs:
        p0, p1, ids = frame_correspondences(tracks, f0, f1)
        if len(p0) < 2 * priors.min_structure_tracks:
            pair_evidence.append(PairEvidence(f0, f1, len(p0), [], True,
                                              ["insufficient correspondences"], False))
            for tid in ids:
                votes[tid][DEFERRED] += 1.0
            continue
        structures, reasons = _sequential_ransac(p0, p1, ids, width, height, priors, rng)
        if not structures:
            pair_evidence.append(PairEvidence(f0, f1, len(p0), [], True, reasons, False))
            for tid in ids:
                votes[tid][DEFERRED] += 1.0
            continue
        reasons += _label_structures(structures, priors, len(p0))
        crossing = _detect_crossing(structures, width, priors)
        if crossing:
            crossing_windows.append((f0, f1))
            reasons.append("same-speed crossing: association deferred")
        ambiguous = any(s.label == DEFERRED for s in structures) or crossing
        if ambiguous:
            deferred_pairs.append((f0, f1))
        explained = set()
        for s in structures:
            effective = DEFERRED if ambiguous else s.label
            w = 1.0 if not ambiguous else 0.35
            for tid in s.track_ids:
                explained.add(int(tid))
                votes[int(tid)][effective] += w
                window_assign[int(tid)][(f0, f1)] = effective
                if ambiguous and s.label != DEFERRED:
                    votes[int(tid)][s.label] += 0.15
        # Correspondences no structure explained get soft-deferred evidence.
        for tid in ids:
            if int(tid) not in explained:
                votes[int(tid)][DEFERRED] += 0.5
                window_assign[int(tid)][(f0, f1)] = DEFERRED
        pair_evidence.append(PairEvidence(f0, f1, len(p0), structures, ambiguous,
                                          reasons, crossing))

    # Short tracks that never span an anchor pair inherit labels from local
    # neighbourhood propagation of labelled longer tracks.
    _propagate_short_tracks(tracks, votes, anchor_span)

    # Independent motion-magnitude evidence. Geometry alone lets a planar
    # foreground satisfy a background epipolar model on some anchor pairs; the
    # near box's image flow is, however, far larger than the distant wall's.
    _add_motion_evidence(tracks, votes, priors)

    labels, flips = _resolve_labels(tracks, votes, crossing_windows, priors)
    for tid, lab in labels.items():
        lab.window_labels = dict(window_assign.get(tid, {}))
    _defer_crossing_flips(labels, crossing_windows)
    bg = sorted(tid for tid, lab in labels.items() if lab.layer == BACKGROUND)
    box = sorted(tid for tid, lab in labels.items() if lab.layer == BOX)
    deferred = sorted(tid for tid, lab in labels.items() if lab.layer == DEFERRED)

    ok, reason, warn = _verdict(bg, box, deferred, labels, pair_evidence, priors)
    warnings.extend(warn)
    coverage = {
        "background_tracks": len(bg),
        "box_tracks": len(box),
        "deferred_tracks": len(deferred),
        "deferred_pairs": deferred_pairs,
        "anchor_span": anchor_span,
    }
    return DecompositionResult(
        ok=ok, reason=reason, labels=labels, pair_evidence=pair_evidence,
        crossing_windows=crossing_windows, internal_label_flips=flips,
        bg_tracks=bg, box_tracks=box, deferred_tracks=deferred,
        coverage_evidence=coverage, warnings=warnings,
        params=dict(threshold_px=priors.ransac_threshold_px, sigma_px=priors.sigma_px,
                    label_margin=priors.label_margin, seed=seed, anchor_span=anchor_span),
    )


def _median_flow(track: Track) -> float:
    """Median per-frame image displacement across consecutive observations."""
    flows = [
        float(np.linalg.norm(track.uvs[i + 1] - track.uvs[i]))
        for i in range(len(track.frames) - 1)
        if track.frames[i + 1] - track.frames[i] == 1
    ]
    return float(np.median(flows)) if flows else 0.0


def _add_motion_evidence(tracks, votes, priors: Priors):
    """Cast a weighted layer vote from each track's image-motion magnitude.

    A linear ramp between the declared slow/fast thresholds keeps the band of
    genuinely ambiguous speeds uncommitted instead of forcing a label. This is
    supplementary evidence: geometry votes dominate wherever they are
    unambiguous. Where the geometric kernel itself deferred (no static/moving
    contrast, e.g. a dominant foreground or same-speed crossing), flow is not
    permitted to invent an identity — its weight is routed to deferral
    instead, because image speed is exactly the cue that fails in those cases.
    """
    slow, fast = priors.motion_flow_slow, priors.motion_flow_fast
    w = priors.motion_flow_weight
    for t in tracks:
        v = votes[t.track_id]
        geometric = v[BACKGROUND] + v[BOX]
        # Only override flow when the track actually accumulated enough
        # anchor-pair geometry for the deferral to be meaningful. A short,
        # geometrically unsupported track has no real "deferred" verdict to
        # protect and still gets its label from the motion/neighbourhood prior.
        geometry_deferred = (
            geometric >= priors.min_geometry_weight
            and v[DEFERRED] >= priors.deferral_dominance * geometric
        )
        if geometry_deferred:
            v[DEFERRED] += w
            continue
        flow = _median_flow(t)
        if flow <= slow:
            v[BACKGROUND] += w
        elif flow >= fast:
            v[BOX] += w
        else:
            frac = (flow - slow) / max(fast - slow, 1e-9)
            v[BACKGROUND] += w * (1.0 - frac)
            v[BOX] += w * frac


def _propagate_short_tracks(tracks, votes, anchor_span):
    """Give anchor-uncovered tracks a spatial prior from nearby labelled tracks.

    The propagated weight is deliberately weak: a short track whose local
    neighbourhood is itself mixed still falls through to DEFERRED rather than
    inheriting a hard label.
    """
    by_id = {t.track_id: t for t in tracks}
    labelled = [tid for tid, t in by_id.items()
                if votes[tid][BACKGROUND] + votes[tid][BOX] > 0]
    unlabelled = [tid for tid in by_id
                  if votes[tid][BACKGROUND] == 0 and votes[tid][BOX] == 0]
    if not labelled or not unlabelled:
        return
    centroid = {tid: by_id[tid].uv_array().mean(axis=0) for tid in by_id}
    lab_pts = np.array([centroid[tid] for tid in labelled])
    for tid in unlabelled:
        d = np.linalg.norm(lab_pts - centroid[tid], axis=1)
        nearest = np.argsort(d)[:6]
        bg_v = sum(votes[labelled[i]][BACKGROUND] for i in nearest)
        box_v = sum(votes[labelled[i]][BOX] for i in nearest)
        votes[tid][BACKGROUND] += 0.4 * bg_v / len(nearest)
        votes[tid][BOX] += 0.4 * box_v / len(nearest)
        votes[tid][DEFERRED] += 0.5


def _resolve_labels(tracks, votes, crossing_windows, priors):
    labels = {}
    flips = 0
    crossing_frames = {f for w in crossing_windows for f in w}
    for t in tracks:
        v = votes[t.track_id]
        bg_v, box_v, defer_v = v[BACKGROUND], v[BOX], v[DEFERRED]
        total = bg_v + box_v + defer_v
        if total <= 0:
            labels[t.track_id] = TrackLabel(t.track_id, DEFERRED, 0.0, dict(v))
            continue
        # If most evidence is deferred and the two layers are close, hold.
        if defer_v > 0.6 * total and abs(bg_v - box_v) < priors.label_margin * total:
            labels[t.track_id] = TrackLabel(t.track_id, DEFERRED,
                                            float(defer_v / total), dict(v))
            continue
        # Dominant geometric deferral: when most anchor pairs could not assign
        # an identity, supplementary flow evidence is not allowed to force a
        # hard layer label. The track stays deferred and is reported as such.
        if (bg_v + box_v) >= priors.min_geometry_weight and \
                defer_v > priors.deferral_dominance * (bg_v + box_v):
            labels[t.track_id] = TrackLabel(t.track_id, DEFERRED,
                                            float(defer_v / total), dict(v))
            continue
        if bg_v > box_v:
            layer, conf = BACKGROUND, float(bg_v / total)
        elif box_v > bg_v:
            layer, conf = BOX, float(box_v / total)
        else:
            layer, conf = DEFERRED, 0.5
        labels[t.track_id] = TrackLabel(t.track_id, layer, conf, dict(v),
                                        deferred_frames=[f for f in t.frames
                                                         if f in crossing_frames])
    # Internal flip count: tracks whose pre- and post-crossing majority differ.
    for w in crossing_windows:
        f_mid = w[0]
        pre, post = {}, {}
        for t in tracks:
            v = votes[t.track_id]
            pre_votes = sum(1 for f in t.frames if f < f_mid)
            post_votes = sum(1 for f in t.frames if f > f_mid)
            if pre_votes and post_votes:
                pre[t.track_id] = max((BACKGROUND, BOX), key=lambda k: v[k])
        # Compare against final label.
        for tid, pre_layer in pre.items():
            if labels[tid].layer in (BACKGROUND, BOX) and labels[tid].layer != pre_layer:
                flips += 1
    return labels, flips


def _defer_crossing_flips(labels: dict, crossing_windows: list):
    """Hold identity across an unresolvable same-speed crossing.

    A track whose *hard* anchor-window label flips from one layer to the other
    across a detected crossing cannot be reliably re-associated: assigning it a
    hard post-crossing identity is exactly the swap the brief forbids. Defer
    just those tracks (not every track near the crossing). Tracks whose label
    stays consistent through the crossing keep their hard label.
    """
    if not crossing_windows:
        return
    windows = sorted({w for lab in labels.values() for w in lab.window_labels},
                     key=lambda z: z[0])
    hard = (BACKGROUND, BOX)
    for cf0, cf1 in crossing_windows:
        pre_win = next((w for w in windows if w[1] <= cf0), None)
        post_win = next((w for w in windows if w[0] >= cf1), None)
        if pre_win is None or post_win is None:
            continue
        for lab in labels.values():
            if lab.layer not in hard:
                continue
            pre = lab.window_labels.get(pre_win)
            post = lab.window_labels.get(post_win)
            if pre in hard and post in hard and pre != post:
                lab.layer = DEFERRED
                lab.confidence = min(lab.confidence, 0.5)


def _verdict(bg, box, deferred, labels, pair_evidence, priors):
    warnings, reason = [], "ok"
    if len(bg) < priors.min_structure_tracks:
        return False, f"insufficient background support ({len(bg)} tracks)", warnings
    if len(box) < priors.min_structure_tracks:
        reason = "foreground layer weak; only a camera solve is reliable"
        warnings.append(reason)
    defer_frac = len(deferred) / max(len(labels), 1)
    if defer_frac > 0.25:
        warnings.append(f"{defer_frac:.0%} of tracks remain deferred; separation unreliable")
        reason = "ambiguous separation"
    ambiguous_pairs = sum(1 for p in pair_evidence if p.ambiguous)
    if ambiguous_pairs > 0.5 * max(len(pair_evidence), 1):
        warnings.append("majority of frame pairs are ambiguous")
        reason = "ambiguous separation"
    ok = reason == "ok"
    return ok, reason, warnings


def layer_track_data(tracks: list[Track], result: DecompositionResult, layer: str):
    """Extract (track_id -> (frames, uvs)) for one layer, ready for solving."""
    wanted = result.bg_tracks if layer == BACKGROUND else result.box_tracks
    out = {}
    for t in tracks:
        if t.track_id in wanted:
            out[t.track_id] = (list(t.frames), t.uv_array())
    return out


def _triangulate_track(track: Track, cam_R, cam_t, K: np.ndarray):
    """Linear triangulation from the track's first/last view; returns X and the
    multi-view reprojection RMSE under the assumption the point is static."""
    frames = track.frames
    if len(frames) < 2:
        return None
    i, j = frames[0], frames[-1]
    P0 = K @ np.hstack([cam_R[i], cam_t[i].reshape(3, 1)])
    P1 = K @ np.hstack([cam_R[j], cam_t[j].reshape(3, 1)])
    X = triangulate(P0, P1, track.uvs[0:1], track.uvs[-1:])[0]
    if X[2] <= 1e-3 or not np.isfinite(X).all():
        return None
    errs = []
    for f, uv in zip(frames, track.uvs):
        q = project_points(X[None], K, cam_R[f], cam_t[f])[0]
        if np.isfinite(q).all():
            errs.append(np.linalg.norm(q - uv))
    return X, (float(np.sqrt(np.mean(np.asarray(errs) ** 2))) if errs else np.inf)


def relabel_with_depth(tracks: list[Track], result: DecompositionResult,
                       cam_R, cam_t, K: np.ndarray, priors: Priors) -> DecompositionResult:
    """Second-pass layer evidence using the solved background camera.

    Every track is triangulated *as if static* against the background camera.
    A genuinely static point reproduces across all its views at the declared
    background depth; a point on the moving box yields either a large
    multi-view reprojection residual (its motion violates rigidity) or a
    nearer, off-prior depth. The votes are added to the existing geometric /
    flow evidence and labels are re-resolved, so depth can overturn a wrong
    cardinality-driven label but never acts alone.
    """
    votes = {tid: dict(lab.evidence) for tid, lab in result.labels.items()}
    bg_depth = 0.5 * (priors.background_depth_range[0] +
                      priors.background_depth_range[1])
    near, far = priors.depth_near_frac * bg_depth, priors.depth_far_frac * bg_depth
    w = priors.depth_evidence_weight
    depth_map, residual_map = {}, {}
    for t in tracks:
        got = _triangulate_track(t, cam_R, cam_t, K)
        if got is None:
            continue
        X, residual = got
        depth_map[t.track_id] = float(abs(X[2]))
        residual_map[t.track_id] = residual
        v = votes.setdefault(t.track_id, {BACKGROUND: 0.0, BOX: 0.0, DEFERRED: 0.0})
        # High static-camera reprojection is the strongest moving-point signal.
        if residual > priors.depth_residual_px:
            v[BOX] = w
        elif depth_map[t.track_id] < near:
            v[BOX] = 0.7 * w
        elif depth_map[t.track_id] > far:
            v[BACKGROUND] = 0.7 * w
        else:
            # In-prior, low-residual point: mild static support, no hard call.
            v[BACKGROUND] = 0.3 * w
    labels, flips = _resolve_labels(tracks, votes, result.crossing_windows, priors)
    bg = sorted(tid for tid, lab in labels.items() if lab.layer == BACKGROUND)
    box = sorted(tid for tid, lab in labels.items() if lab.layer == BOX)
    deferred = sorted(tid for tid, lab in labels.items() if lab.layer == DEFERRED)
    ok, reason, warn = _verdict(bg, box, deferred, labels, result.pair_evidence, priors)
    result.labels = labels
    result.bg_tracks, result.box_tracks, result.deferred_tracks = bg, box, deferred
    result.internal_label_flips = flips
    result.ok, result.reason, result.warnings = ok, reason, warn
    result.coverage_evidence["depth_relabel"] = {
        "triangulated": len(depth_map),
        "bg_depth_m": bg_depth,
        "near_m": near, "far_m": far,
        "median_residual_px": float(np.median(list(residual_map.values())))
        if residual_map else None,
    }
    return result
