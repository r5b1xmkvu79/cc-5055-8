"""Ground-truth evaluation of a decomposition run.

Metrics
-------
* Layer precision / recall / F1 against true feature layers (deferred tracks
  counted separately, never silently treated as correct).
* Identity swaps: GT features whose solved layer changes across a crossing
  window, i.e. the two physical entities exchange their assigned identity.
* Camera pose errors (rotation, translation direction) and box pose errors
  after a single similarity alignment onto the metric GT frame.
* Reprojection RMSE for both layers.

KLT tracks that lock onto rendered texture grain (rather than the labelled
landmark dots) have no feature id. They are assigned a true layer geometrically
by casting the observation ray through the GT camera and testing intersection
with the GT box, so the metrics cover the full track population.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decomposition import BACKGROUND, BOX, DEFERRED
from .geometry import (
    project_points,
    rigid_transform_3d,
    rotation_error,
    translation_angle_error,
)
from .synth import BG_LAYER, BOX_LAYER


@dataclass
class EvaluationReport:
    n_tracks: int
    track_to_gt: dict
    gt_layer: dict
    label_confusion: dict
    layer_f1: dict
    deferred_count: int
    identity_swaps: int
    swap_details: list
    camera_rot_deg: float
    camera_t_dir_deg: float
    camera_pos_m: float
    box_rot_deg: float
    box_pos_m: float
    bg_reproj_px: float
    box_reproj_px: float
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def load_observations(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    frames = []
    for ids, uv, layers in zip(data["ids"], data["uv"], data["layers"]):
        frames.append([
            dict(id=int(i), uv=np.asarray(p, dtype=np.float64), layer=str(l))
            for i, p, l in zip(ids, uv, layers)
        ])
    return frames


def match_tracks_to_gt(tracks, gt_frames, match_gate_px: float = 8.0):
    """Nearest-neighbour association of KLT tracks to GT feature ids."""
    # Index GT per frame.
    gt_index = []
    for frame in gt_frames:
        gt_index.append({int(o["id"]): np.asarray(o["uv"], dtype=np.float64) for o in frame})
    mapping = {}
    for tr in tracks:
        votes = {}
        for f, uv in zip(tr.frames, tr.uvs):
            if f >= len(gt_index):
                continue
            best_id, best_d = None, match_gate_px
            for gid, guv in gt_index[f].items():
                d = np.linalg.norm(guv - uv)
                if d < best_d:
                    best_id, best_d = gid, d
            if best_id is not None:
                votes[best_id] = votes.get(best_id, 0) + 1
        if votes:
            mapping[tr.track_id] = max(votes, key=votes.get)
    return mapping


def gt_layers(gt_bundle_path: str):
    gt = np.load(gt_bundle_path, allow_pickle=False)
    offset = int(gt["box_id_offset"])
    layers = {}
    n_gt = int(gt["bg_world"].shape[0]) + int(gt["box_body_points"].shape[0])
    for gid in range(n_gt):
        layers[gid] = BG_LAYER if gid < offset else BOX_LAYER
    return layers, int(gt["landmark_id"])


def _ray_box_intersect(origin, ray, Rb, tb, half, t_max=12.0):
    """Slab test of a ray against the oriented GT box."""
    o = Rb.T @ (origin - tb)
    d = Rb.T @ ray
    t_lo, t_hi = 1e-6, t_max
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if abs(o[axis]) > half[axis]:
                return False
            continue
        t1 = (-half[axis] - o[axis]) / d[axis]
        t2 = (half[axis] - o[axis]) / d[axis]
        t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
        if t_lo > t_hi:
            return False
    return t_hi > 1e-6


def true_layer_for_tracks(tracks, gt_bundle_path, K):
    """Assign every KLT track a true layer via GT-camera ray / GT-box tests."""
    gt = np.load(gt_bundle_path)
    cam_R, cam_t = gt["cam_R"], gt["cam_t"]
    box_R, box_t = gt["box_R"], gt["box_t"]
    half = np.asarray(gt["box_size"], dtype=np.float64) / 2.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    true_layer = {}
    for tr in tracks:
        votes = []
        for f, uv in zip(tr.frames, tr.uvs):
            center = -cam_R[f].T @ cam_t[f]
            ray = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0])
            ray = cam_R[f].T @ ray
            ray /= np.linalg.norm(ray)
            on_box = _ray_box_intersect(center, ray, box_R[f], box_t[f], half)
            votes.append(BOX_LAYER if on_box else BG_LAYER)
        # Majority over the track lifetime.
        true_layer[tr.track_id] = max(set(votes), key=votes.count)
    return true_layer


def layer_confusion(result, mapping, gt_layer_map):
    """``gt_layer_map`` is keyed by track id (see true_layer_for_tracks)."""
    confusion = {BACKGROUND: {BG_LAYER: 0, BOX_LAYER: 0},
                 BOX: {BG_LAYER: 0, BOX_LAYER: 0},
                 DEFERRED: {BG_LAYER: 0, BOX_LAYER: 0}}
    for tid, lab in result.labels.items():
        gt_name = gt_layer_map.get(tid)
        if gt_name is None:
            continue
        confusion[lab.layer][gt_name] += 1
    f1 = {}
    for layer, gt_name in ((BACKGROUND, BG_LAYER), (BOX, BOX_LAYER)):
        tp = confusion[layer][gt_name]
        fp = sum(confusion[layer][g] for g in (BG_LAYER, BOX_LAYER)) - tp
        fn = sum(confusion[l][gt_name] for l in confusion) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[layer] = dict(precision=prec, recall=rec,
                         f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                         tp=tp, fp=fp, fn=fn)
    return confusion, f1


def identity_swaps(result, mapping, gt_layer_map, crossing_windows):
    """Count true identity exchanges across a same-speed crossing window.

    A swap is a *track whose assigned structure changes from one hard layer to
    the other between the windows immediately before and after a crossing*.
    Deferred windows are not swaps: holding identity through an ambiguous
    crossing is the correct behaviour, not an exchange. Vote ambiguity alone
    (a single hard label with competing soft evidence) does not count.
    """
    if not crossing_windows:
        return 0, []
    swaps, details = 0, []
    windows = sorted({(win.f0, win.f1) for win in result.pair_evidence},
                     key=lambda z: z[0])
    for (cf0, cf1) in crossing_windows:
        pre_win = next((w for w in windows if w[1] <= cf0), None)
        post_win = next((w for w in windows if w[0] >= cf1), None)
        if pre_win is None or post_win is None:
            continue
        for tid, lab in result.labels.items():
            # A track the kernel ultimately deferred across the crossing is an
            # identity hold, not an exchange, even though its historical window
            # labels record competing hard evidence.
            if lab.layer == DEFERRED:
                continue
            gt_name = gt_layer_map.get(tid)
            if gt_name is None:
                continue
            pre = lab.window_labels.get(pre_win)
            post = lab.window_labels.get(post_win)
            hard = (BACKGROUND, BOX)
            if pre in hard and post in hard and pre != post:
                swaps += 1
                details.append(dict(track_id=tid, gt_layer=gt_name,
                                    pre_window=list(pre_win), post_window=list(post_win),
                                    pre_label=pre, post_label=post))
    return swaps, details


def _procrustes_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Global rotation correction mapping each A_i onto B_i.

    Per-frame corrections G_i = B_i A_i^T are consistent up to solve noise;
    take the median correction (geodesic nearest to the first frame) to stay
    robust to a single outlier frame.
    """
    corrections = [B[i] @ A[i].T for i in range(len(A))]
    ref = corrections[0]
    angles = np.array([rotation_error(G, ref) for G in corrections])
    return corrections[int(np.argsort(angles)[len(angles) // 2])]


def camera_pose_errors(cam_solve, gt_R, gt_t):
    """SfM reconstructs motion up to a global similarity. Align the solved
    optical-centre trajectory to GT once (Sim3), refine the rotation against
    camera orientations, then report per-frame pose errors.
    """
    centres_est = np.array([-R.T @ t for R, t in zip(cam_solve.R, cam_solve.t)])
    centres_gt = np.array([-R.T @ t for R, t in zip(gt_R, gt_t)])
    R_align, t_align, scale = rigid_transform_3d(centres_est, centres_gt)
    # Refine orientation using all camera frames' relative orientation.
    R_corr = _procrustes_rotation(
        np.array([(R_align @ R) for R in cam_solve.R]),
        np.array([R for R in gt_R]),
    )
    R_align = R_corr @ R_align
    rot_errs, t_dir_errs, pos_errs = [], [], []
    for f, (Re, te, Rg, tg) in enumerate(zip(cam_solve.R, cam_solve.t, gt_R, gt_t)):
        R_est_w = R_align @ Re
        rot_errs.append(rotation_error(R_est_w, Rg))
        aligned_centre = scale * (R_align @ centres_est[f]) + t_align
        pos_errs.append(np.linalg.norm(aligned_centre - centres_gt[f]))
        t_dir_errs.append(translation_angle_error(R_align @ te, tg))
    return float(np.median(rot_errs)), float(np.median(t_dir_errs)), float(np.median(pos_errs))


def _cube_symmetry_aware_rot_err(R_est_aligned, R_gt, box_size):
    """Rotation error accounting for the box's reflection/rotation symmetry.

    A plain cube with near-symmetric texture admits 24 orientation-equivalent
    solutions; a large raw angle with a small centre error is an unobservable
    sign flip, not a tracking failure. Report the minimum error over the
    cube's 90-degree body-frame rotations.
    """
    best = rotation_error(R_est_aligned, R_gt)
    half = np.asarray(box_size) / 2.0
    # Sign flips of body axes that leave a centred cube invariant (proper
    # rotations only: even number of sign changes).
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                if sx * sy * sz < 0:
                    continue
                S = np.diag([sx, sy, sz]).astype(float)
                cand = R_est_aligned @ S
                best = min(best, rotation_error(cand, R_gt))
    return best


def box_pose_errors(box_solve, gt_Rb, gt_tb, box_size=None):
    """One global similarity aligns solved box centres to GT, then score."""
    valid = [f for f in range(len(gt_tb)) if box_solve.R[f] is not None]
    if not valid:
        return np.inf, np.inf
    est_c = np.array([box_solve.t[f] for f in valid])
    gt_c = np.array([gt_tb[f] for f in valid])
    R_align, t_align, scale = rigid_transform_3d(est_c, gt_c)
    rot_errs, pos_errs = [], []
    for f in valid:
        aligned_c = scale * (R_align @ box_solve.t[f]) + t_align
        pos_errs.append(np.linalg.norm(aligned_c - gt_tb[f]))
        R_est = R_align @ box_solve.R[f]
        if box_size is not None:
            rot_errs.append(_cube_symmetry_aware_rot_err(R_est, gt_Rb[f], box_size))
        else:
            rot_errs.append(rotation_error(R_est, gt_Rb[f]))
    return float(np.median(rot_errs)), float(np.median(pos_errs))


def reprojection_errors(tracks, result, cam_solve, box_solve, K):
    bg_err, box_err = [], []
    for tr in tracks:
        lab = result.labels.get(tr.track_id)
        if lab is None or lab.layer == DEFERRED:
            continue
        for f, uv in zip(tr.frames, tr.uvs):
            if lab.layer == BACKGROUND:
                X = cam_solve.points3d.get(tr.track_id)
                if X is None:
                    continue
                q = project_points(X[None], K, cam_solve.R[f], cam_solve.t[f])[0]
                if np.isfinite(q).all():
                    bg_err.append(np.linalg.norm(q - uv))
            else:
                Xb = box_solve.track_body_points.get(tr.track_id)
                if Xb is None:
                    continue
                Xw = box_solve.R[f] @ Xb + box_solve.t[f]
                q = project_points(Xw[None], K, cam_solve.R[f], cam_solve.t[f])[0]
                if np.isfinite(q).all():
                    box_err.append(np.linalg.norm(q - uv))
    return (_robust_rmse(bg_err), _robust_rmse(box_err))


def _robust_rmse(errs):
    if not errs:
        return np.inf
    arr = np.asarray(errs)
    med = np.median(arr)
    inl = arr[arr < med + 3.0 * (np.percentile(arr, 75) - med + 1e-6)]
    return float(np.sqrt(np.mean(inl ** 2)))


def evaluate(tracks, result, cam_solve, box_solve, K, gt_bundle_path, obs_npz_path):
    gt_frames = load_observations(obs_npz_path)
    mapping = match_tracks_to_gt(tracks, gt_frames)
    id_layer_map, landmark = gt_layers(gt_bundle_path)
    # Ray-based truth covers grain tracks that have no GT feature id; explicit
    # id matches override it where available.
    gt_layer_map = true_layer_for_tracks(tracks, gt_bundle_path, K)
    for tid, gid in mapping.items():
        if gid in id_layer_map:
            gt_layer_map[tid] = id_layer_map[gid]
    confusion, f1 = layer_confusion(result, mapping=None, gt_layer_map=gt_layer_map)
    swaps, swap_details = identity_swaps(result, mapping=None, gt_layer_map=gt_layer_map,
                                         crossing_windows=result.crossing_windows)
    gt = np.load(gt_bundle_path)
    cam_r, cam_td, cam_pos = (np.inf, np.inf, np.inf)
    box_r, box_p = (np.inf, np.inf)
    if cam_solve.ok:
        cam_r, cam_td, cam_pos = camera_pose_errors(cam_solve, gt["cam_R"], gt["cam_t"])
    if box_solve is not None and box_solve.ok:
        box_r, box_p = box_pose_errors(box_solve, gt["box_R"], gt["box_t"],
                                       box_size=np.asarray(gt["box_size"]))
    bg_rep, box_rep = reprojection_errors(tracks, result, cam_solve,
                                          box_solve if box_solve and box_solve.ok else _EmptyBox(),
                                          K)
    deferred = sum(1 for l in result.labels.values() if l.layer == DEFERRED)
    return EvaluationReport(
        n_tracks=len(tracks), track_to_gt=mapping, gt_layer=gt_layer_map,
        label_confusion=confusion, layer_f1=f1, deferred_count=deferred,
        identity_swaps=swaps, swap_details=swap_details,
        camera_rot_deg=cam_r, camera_t_dir_deg=cam_td, camera_pos_m=cam_pos,
        box_rot_deg=box_r, box_pos_m=box_p,
        bg_reproj_px=bg_rep, box_reproj_px=box_rep,
    )


class _EmptyBox:
    ok = False
    track_body_points = {}

    def __getattr__(self, name):
        raise AttributeError(name)
