"""Static-background camera solve.

Incremental two-view SfM over the tracks assigned to the background layer:
essential matrix -> triangulation -> PnP registration -> pose-only bundle
adjustment. Scale is free from geometry alone; the declared background depth
range supplies a *metric* scale, reported transparently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from .geometry import essential_from_F, pose_from_essential, project_points, triangulate

BG_LAYER = "background"


@dataclass
class CameraSolve:
    ok: bool
    reason: str = ""
    R: list = field(default_factory=list)          # per frame 3x3
    t: list = field(default_factory=list)          # per frame (3,)
    points3d: dict = field(default_factory=dict)   # track_id -> X in world
    scale: float = 1.0
    mean_reproj_px: float = np.inf
    n_tracks: int = 0


def _pose_params(R, t):
    return np.hstack([cv2.Rodrigues(R)[0].ravel(), t.ravel()])


def _unpose(p):
    return cv2.Rodrigues(p[:3])[0], p[3:6]


def solve_cameras(layer_uv: dict, K: np.ndarray, n_frames: int,
                  depth_range=(1.5, 12.0), min_tracks: int = 8) -> CameraSolve:
    """``layer_uv`` maps track_id -> (frames list, Nx2 uvs)."""
    track_ids = sorted(layer_uv)
    if len(track_ids) < min_tracks:
        return CameraSolve(False, reason=f"too few background tracks ({len(track_ids)} < {min_tracks})")

    # Long-baseline seed: short baseline under mostly-rotational motion makes
    # the essential-matrix translation ill-conditioned. The RANSAC/PnP chain
    # can also collapse from a single unlucky seed (notably when the foreground
    # dominates and few static tracks survive), so try several distinct seed
    # pairs and keep the chain that registers the most frames.
    f0 = min(f for tid in track_ids for f in layer_uv[tid][0])
    f_last = max(f for tid in track_ids for f in layer_uv[tid][0])
    candidates = _candidate_seed_pairs(layer_uv, track_ids, f0, f_last)
    if not candidates:
        return CameraSolve(False, reason="no pair with enough shared tracks")

    best_chain = None
    for fa, fb in candidates:
        chain = _attempt_seed(fa, fb, layer_uv, track_ids, K, n_frames)
        if chain is None:
            continue
        if best_chain is None or len(chain[0]) > len(best_chain[0]):
            best_chain = chain
        if len(best_chain[0]) >= max(2, (2 * n_frames) // 3):
            break
    if best_chain is None:
        return CameraSolve(False, reason="essential matrix failed (degenerate motion)")
    Rw, tw, points3d = best_chain

    present = [f for f in range(n_frames) if f in Rw]
    if len(present) < max(2, n_frames // 3):
        return CameraSolve(False, reason="most frames failed PnP registration")
    R_all, t_all = _interpolate_missing(Rw, tw, n_frames)
    # Re-triangulate against the registered chain, then repeat PnP once to
    # remove drift accumulated during the single forward registration pass.
    _triangulate_new(layer_uv, track_ids, R_all, t_all, points3d, K)
    Rw2 = {f: R_all[f] for f in range(n_frames) if f in Rw}
    tw2 = {f: t_all[f] for f in range(n_frames) if f in Rw}
    for f in range(n_frames):
        if f in Rw2:
            continue
        _register_frame(f, layer_uv, track_ids, Rw2, tw2, points3d, K)
    R_all, t_all = _interpolate_missing(Rw2, tw2, n_frames)
    _triangulate_new(layer_uv, track_ids, R_all, t_all, points3d, K)

    R_all, t_all, _held, mean_rep = _bundle_adjust(
        layer_uv, track_ids, R_all, t_all, points3d, K
    )
    # Re-triangulate every background track against the refined pose chain so
    # downstream reprojection covers the full layer, not the BA subset.
    points3d = {}
    _triangulate_new(layer_uv, track_ids, R_all, t_all, points3d, K)
    # Drop outlier triangulations: near-camera points from long-baseline
    # mismatch dominate an RMSE even when the median solve is accurate.
    if points3d:
        depths = np.array([X[2] for X in points3d.values()])
        lo, hi = depth_range
        keep = {tid: X for tid, X in points3d.items()
                if lo * 0.3 <= X[2] <= hi * 2.0}
        if len(keep) >= 8:
            points3d = keep

    depths = np.array([X[2] for X in points3d.values()])
    target_depth = 0.5 * (depth_range[0] + depth_range[1])
    scale = float(target_depth / np.median(depths)) if len(depths) else 1.0
    t_all = [tt * scale for tt in t_all]
    points3d = {tid: X * scale for tid, X in points3d.items()}
    return CameraSolve(
        True, R=R_all, t=t_all, points3d=points3d, scale=scale,
        mean_reproj_px=mean_rep, n_tracks=len(track_ids),
    )


def _essential(p_a, p_b, K):
    E, mask = cv2.findEssentialMat(p_a.astype(np.float64), p_b.astype(np.float64),
                                   K, cv2.RANSAC, 0.999, 1.0)
    return E, mask


def _best_seed_pair(layer_uv, track_ids, f0, f_last, min_shared=18,
                    max_span_frac=0.45):
    """Widest pair up to a baseline cap: a moderate baseline gives a
    well-conditioned translation while keeping the PnP chain locally anchored.
    """
    span_cap = max(4, int((f_last - f0) * max_span_frac))
    best, best_span = None, -1
    for fa in range(f0, f_last):
        for fb in range(min(f_last, fa + span_cap), fa, -1):
            count = sum(1 for tid in track_ids
                        if fa in layer_uv[tid][0] and fb in layer_uv[tid][0])
            if count >= min_shared and (fb - fa) > best_span:
                best, best_span = (fa, fb), fb - fa
                break
    if best is not None:
        return best
    fallback, best_count = None, 0
    for fa in range(f0, f_last):
        for fb in range(fa + 1, f_last + 1):
            count = sum(1 for tid in track_ids
                        if fa in layer_uv[tid][0] and fb in layer_uv[tid][0])
            if count > best_count:
                fallback, best_count = (fa, fb), count
    return fallback if best_count >= 5 else None


def _candidate_seed_pairs(layer_uv, track_ids, f0, f_last, max_pairs=4,
                          min_shared=12):
    """Distinct, well-populated seed pairs ordered widest-baseline first.

    Several anchors let solve_cameras retry when one registration chain
    collapses instead of failing the whole camera solve on a single RANSAC run.
    """
    scored = []
    for fa in range(f0, f_last):
        for fb in range(fa + 1, f_last + 1):
            count = sum(1 for tid in track_ids
                        if fa in layer_uv[tid][0] and fb in layer_uv[tid][0])
            if count >= min_shared:
                scored.append((fb - fa, count, (fa, fb)))
    if not scored:
        best = _best_seed_pair(layer_uv, track_ids, f0, f_last)
        return [best] if best is not None else []
    # Prefer wide baseline, then many shared tracks; de-duplicate endpoints.
    scored.sort(key=lambda z: (z[0], z[1]), reverse=True)
    chosen, used = [], set()
    for _, _, pair in scored:
        if pair in used:
            continue
        chosen.append(pair)
        used.add(pair)
        if len(chosen) >= max_pairs:
            break
    return chosen


def _attempt_seed(fa, fb, layer_uv, track_ids, K, n_frames):
    """Build one two-view seed and register the remaining frames from it."""
    p_a, p_b, ids = _pair_points(layer_uv, track_ids, fa, fb)
    E, _ = _essential(p_a, p_b, K)
    if E is None:
        return None
    R_rel, t_rel, X_cam = pose_from_essential(E, K, p_a, p_b)

    Rw = {fa: np.eye(3), fb: R_rel}
    tw = {fa: np.zeros(3), fb: t_rel}
    points3d = {}
    depth_pos = X_cam[:, 2]
    positive = depth_pos[depth_pos > 0]
    if len(positive) < 6:
        return None
    depth_keep = (depth_pos > 0.05) & (depth_pos < np.quantile(positive, 0.9) * 3.0 + 1e-6)
    for tid, X, keep in zip(ids, X_cam, depth_keep):
        if keep:
            points3d[tid] = X

    order = [f for f in range(n_frames) if f not in (fa, fb)]
    order.sort(key=lambda f: min(abs(f - fa), abs(f - fb)))
    for f in order:
        before = len(Rw)
        _register_frame(f, layer_uv, track_ids, Rw, tw, points3d, K)
        if len(Rw) > before:
            # Grow the 3D pool as the chain extends; otherwise a sparse seed
            # (only ~7 triangulated points) stalls once the local PnP support
            # runs out and the far half of the clip never registers.
            _triangulate_new(layer_uv, track_ids, Rw, tw, points3d, K)
    if len(Rw) < max(2, n_frames // 3):
        return None
    return Rw, tw, points3d


def _pair_points(layer_uv, track_ids, fa, fb):
    pa, pb, ids = [], [], []
    for tid in track_ids:
        frames, uvs = layer_uv[tid]
        if fa in frames and fb in frames:
            pa.append(uvs[frames.index(fa)])
            pb.append(uvs[frames.index(fb)])
            ids.append(tid)
    return np.asarray(pa), np.asarray(pb), ids


def _register_frame(f, layer_uv, track_ids, Rw, tw, points3d, K):
    obj, img = [], []
    for tid in track_ids:
        if tid not in points3d:
            continue
        frames, uvs = layer_uv[tid]
        if f in frames:
            obj.append(points3d[tid])
            img.append(uvs[frames.index(f)])
    if len(obj) < 4:
        return
    obj, img = np.asarray(obj), np.asarray(img)
    if len(obj) > 120:  # PnP RANSAC cost grows quickly with point count
        sel = np.linspace(0, len(obj) - 1, 120).astype(int)
        obj_pnp, img_pnp = obj[sel], img[sel]
    else:
        obj_pnp, img_pnp = obj, img
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pnp, img_pnp, K, None, iterationsCount=100, reprojectionError=2.0,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < 4:
        return
    Rw[f] = cv2.Rodrigues(rvec)[0]
    tw[f] = tvec.ravel()


def _interpolate_missing(Rw, tw, n_frames):
    known = sorted(Rw)
    R_all, t_all = [], []
    for f in range(n_frames):
        if f in Rw:
            R_all.append(Rw[f].copy())
            t_all.append(tw[f].copy())
            continue
        nxt = min(known, key=lambda g: abs(g - f))
        R_all.append(Rw[nxt].copy())
        t_all.append(tw[nxt].copy())
    return R_all, t_all


def _triangulate_new(layer_uv, track_ids, R_all, t_all, points3d, K):
    # Accept both a per-frame list (full chain) and a frame-keyed dict (the
    # incremental seed chain still being built).
    if isinstance(R_all, dict):
        P = {f: K @ np.hstack([R_all[f], t_all[f].reshape(3, 1)]) for f in R_all}
    else:
        P = {f: K @ np.hstack([R_all[f], t_all[f].reshape(3, 1)])
             for f in range(len(R_all))}
    for tid in track_ids:
        if tid in points3d:
            continue
        frames, uvs = layer_uv[tid]
        if len(frames) < 2:
            continue
        have = [i for i, f in enumerate(frames) if f in P]
        if len(have) < 2:
            continue
        # Widest available baseline gives the best-conditioned triangulation.
        i, j = have[0], have[-1]
        X = triangulate(P[frames[i]], P[frames[j]], uvs[i:i + 1], uvs[j:j + 1])[0]
        if X[2] > 0:
            points3d[tid] = X


def _bundle_adjust(layer_uv, track_ids, R_all, t_all, points3d, K, max_points=120):
    """Pose-only refinement; points stay fixed to avoid finite-difference drift."""
    n = len(R_all)
    pids_all = sorted(points3d)
    if len(pids_all) > max_points:
        scored = sorted(pids_all, key=lambda tid: -len(layer_uv[tid][0]))
        pids = sorted(scored[:max_points])
    else:
        pids = pids_all
    pts = np.array([points3d[tid] for tid in pids])
    x0 = np.concatenate([_pose_params(R_all[f], t_all[f]) for f in range(n)])

    rows = []
    for pi, tid in enumerate(pids):
        frames, uvs = layer_uv[tid]
        for f, uv in zip(frames, uvs):
            rows.append((f, pi, uv))
    n_res = 2 * len(rows)

    def residuals(x):
        cams = x.reshape(n, 6)
        out = np.empty(n_res)
        for r, (f, pi, uv) in enumerate(rows):
            R, t = _unpose(cams[f])
            q = project_points(pts[pi:pi + 1], K, R, t)[0]
            if not np.isfinite(q).all():
                out[2 * r:2 * r + 2] = 1e3
            else:
                out[2 * r:2 * r + 2] = q - uv
        return np.nan_to_num(out, nan=1e3, posinf=1e3, neginf=1e3)

    A = lil_matrix((n_res, x0.size))
    for r, (f, _pi, _uv) in enumerate(rows):
        A[2 * r:2 * r + 2, 6 * f:6 * f + 6] = 1

    try:
        sol = least_squares(residuals, x0, jac_sparsity=A, max_nfev=40, x_scale="jac",
                            method="trf", verbose=0)
        x = sol.x
    except (ValueError, np.linalg.LinAlgError):
        x = x0
    R_new, t_new = [], []
    for f in range(n):
        R, t = _unpose(x[6 * f:6 * f + 6])
        R_new.append(R)
        t_new.append(t)
    if rows:
        r = residuals(x)
        per = np.sqrt(r[0::2] ** 2 + r[1::2] ** 2)
        inliers = per[np.abs(per - np.median(per)) < 3.0 * np.std(per)]
        mean_rep = float(np.sqrt(np.mean(inliers ** 2))) if len(inliers) else float(np.median(per))
    else:
        mean_rep = np.inf
    return R_new, t_new, dict(points3d), mean_rep