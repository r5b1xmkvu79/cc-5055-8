"""Foreground box 6-DoF trajectory for virtual-model overlay.

The box is a rigid body of *declared* dimensions (the CAD prior). Camera poses
for every frame are already known from the background solve. The renderer
places three id-tagged ArUco fiducials on the box, so each frame supplies
*identified* 2D-3D corner correspondences: detection returns both the marker
id (which selects the four body-frame corners) and their image positions. A
direct PnP on those correspondences cannot fall into the box's 180-degree
symmetry basin, because the ids break every rotational self-symmetry of the
face grid. Marker PnP is then refined with the dense grain tracks for a
sub-pixel pose, and every supporting track is bound to a body-frame point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import Track
from .geometry import project_points
from .synth import BoxSpec, SyntheticScene
from .synth import SceneConfig

BOX_LAYER = "box"


@dataclass
class BoxSolve:
    ok: bool
    reason: str = ""
    R: list = field(default_factory=list)        # body -> world, per frame
    t: list = field(default_factory=list)
    track_body_points: dict = field(default_factory=dict)  # track_id -> X_b
    reproj_px: float = np.inf
    n_assigned: int = 0
    markers_per_frame: list = field(default_factory=list)


def _marker_template(box_size: np.ndarray):
    """Body-frame ArUco corner layout shared with the renderer."""
    scene = SyntheticScene(SceneConfig(scenario="baseline",
                                       box_size=np.asarray(box_size, dtype=np.float64)))
    pts, _, ids = scene.aruco_corners3d()
    dictionary = scene.aruco_dictionary
    valid_ids = {int(s["id"]) for s in scene.aruco_specs}
    return pts, np.asarray(ids, dtype=int), dictionary, valid_ids


def _make_detector(dictionary, valid_ids):
    params = cv2.aruco.DetectorParameters()
    # Real markers are large (~130-190 px); reject the small grain quads that
    # occasionally decode as a dictionary id by coincidence.
    params.minMarkerPerimeterRate = 0.12
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector, valid_ids


def _pnp_once(obj, img, K, R_init=None, t_init=None):
    use_extrinsic = R_init is not None and t_init is not None
    flags = cv2.SOLVEPNP_ITERATIVE
    rvec = cv2.Rodrigues(R_init)[0] if use_extrinsic else None
    tvec = t_init.reshape(3, 1).copy() if use_extrinsic else None
    # OpenCV 5.x requires the channel dimension on point arrays.
    obj_cv = np.ascontiguousarray(obj, dtype=np.float64).reshape(-1, 1, 3)
    img_cv = np.ascontiguousarray(img, dtype=np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj_cv, img_cv, K, None, rvec, tvec, useExtrinsicGuess=use_extrinsic, flags=flags
        )
    except cv2.error:
        return False, None, None
    if not ok:
        return False, None, None
    return True, cv2.Rodrigues(rvec)[0], tvec.ravel()


def _grain_template(box_size: np.ndarray, grid: int = 8):
    spec = BoxSpec(np.asarray(box_size, dtype=np.float64), grid=grid)
    pts, normals, _ = spec.face_points()
    return pts, normals


def _visible_template(template, normals, Rb, tb, cam_R, cam_t):
    """Boolean mask of template points whose face normal points at the camera.

    Back-face points project onto the same silhouette as visible ones; without
    this cull a nearest-template assignment can bind a track to the occluded
    symmetric mate on the far side of the box.
    """
    centre = -cam_R.T @ cam_t
    world_n = (Rb @ normals.T).T
    world_p = (Rb @ template.T).T + tb
    view = centre - world_p
    return np.einsum("ij,ij->i", world_n, view) > 0


def _ray_depths(uvs: np.ndarray, K: np.ndarray):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dirs = np.column_stack([(uvs[:, 0] - cx) / fx, (uvs[:, 1] - cy) / fy,
                            np.ones(len(uvs))])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def _assign(template_uv, obs_uv, gate_px: float):
    """Nearest projected template point per observation; -1 if outside gate."""
    diff = obs_uv[:, None, :] - template_uv[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    nearest = dist.argmin(axis=1)
    best = dist[np.arange(len(obs_uv)), nearest]
    assignment = np.where(best < gate_px, nearest, -1)
    return assignment, best


def _refine_with_grain(obs_uv, Rb, tb, cam_R, cam_t, K, box_size, gate_px=10.0,
                       max_rounds=12):
    """Refine a marker-derived pose using dense grain track observations.

    The marker PnP already fixes orientation (including the symmetry branch),
    so nearest-template assignment here is only a local refinement and can no
    longer jump to the 180-degree basin.
    """
    template, normals = _grain_template(box_size)
    for _ in range(max_rounds):
        world = (Rb @ template.T).T + tb
        tpl_uv = project_points(world, K, cam_R, cam_t)
        if not np.isfinite(tpl_uv).all():
            break
        visible = _visible_template(template, normals, Rb, tb, cam_R, cam_t)
        # Send hidden template points "infinitely far" so they are never the
        # nearest match while keeping template indices stable.
        tpl_uv = np.where(visible[:, None], tpl_uv, np.inf)
        assignment, _ = _assign(tpl_uv, obs_uv, gate_px)
        sel = assignment >= 0
        if sel.sum() < 6:
            break
        # solvePnP solves X_c = R_c2b X_body + t in the *camera* frame; feed
        # the assigned body points and their camera-frame coordinates, never
        # world points (a frame mix-up here jumps to the 180-degree basin).
        obj_body = template[assignment[sel]]
        img = obs_uv[sel]
        ok, R_c2b, t_c2b = _pnp_once(obj_body, img, K,
                                     cam_R @ Rb, cam_R @ tb + cam_t)
        if not ok:
            break
        Rb_new = cam_R.T @ R_c2b
        tb_new = cam_R.T @ (t_c2b - cam_t)
        if np.allclose(Rb_new, Rb, atol=1e-6) and np.allclose(tb_new, tb, atol=1e-8):
            Rb, tb = Rb_new, tb_new
            break
        Rb, tb = Rb_new, tb_new
    return Rb, tb


def solve_box(tracks: list[Track], box_track_ids, cam_R, cam_t, K: np.ndarray,
              box_size: np.ndarray, frames=None, gate_px: float = 10.0,
              seed: int = 0) -> BoxSolve:
    n_frames = len(cam_R)
    marker_pts, marker_ids, dictionary, valid_ids = _marker_template(box_size)
    id_to_rows = {mid: np.where(marker_ids == mid)[0] for mid in np.unique(marker_ids)}

    R_all, t_all = [None] * n_frames, [None] * n_frames
    markers_per_frame = [0] * n_frames
    errors, total_n = [], 0

    if frames is None:
        return BoxSolve(False, reason="ArUco box solve requires rendered frames")
    detector, _ = _make_detector(dictionary, valid_ids)

    for f in range(n_frames):
        gray = cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        obj_list, img_list, n_markers = [], [], 0
        for quad, mid in zip(corners, ids.ravel()):
            mid = int(mid)
            if mid not in id_to_rows:
                continue
            obj_list.append(marker_pts[id_to_rows[mid]])
            img_list.append(quad.reshape(4, 2))
            n_markers += 1
        markers_per_frame[f] = n_markers
        if not obj_list:
            continue
        obj = np.vstack(obj_list)
        img = np.vstack(img_list)
        # Direct PnP from identified body-frame corners. With >=2 markers the
        # 8+ non-collinear points yield a unique, symmetry-free pose.
        guess = f > 0 and R_all[f - 1] is not None
        R_init = t_init = None
        if guess:
            R_init = cam_R[f] @ R_all[f - 1]
            t_init = cam_R[f] @ t_all[f - 1] + cam_t[f]
        ok, R_c2b, t_c2b = _pnp_once(obj, img, K, R_init, t_init)
        if not ok:
            continue
        Rb = cam_R[f].T @ R_c2b
        tb = cam_R[f].T @ (t_c2b - cam_t[f])

        obs = _frame_box_uv(tracks, set(box_track_ids), f)
        if len(obs) >= 6:
            Rb, tb = _refine_with_grain(obs, Rb, tb, cam_R[f], cam_t[f], K,
                                        box_size, gate_px=gate_px)
        R_all[f], t_all[f] = Rb, tb

        world = (Rb @ obj.T).T + tb
        rep = np.linalg.norm(project_points(world, K, cam_R[f], cam_t[f]) - img, axis=1)
        errors.append(float(np.sqrt(np.mean(rep ** 2))))
        total_n += len(rep)

    solved = [f for f in range(n_frames) if R_all[f] is not None]
    if len(solved) < max(3, n_frames // 4):
        return BoxSolve(False,
                        reason=f"ArUco pose recovered on only {len(solved)} frames",
                        markers_per_frame=markers_per_frame)

    # Bind every supporting box track to a fixed body-frame point by modal
    # nearest-template assignment across the frames on which it is visible.
    # Only camera-facing template points are eligible, so a track can never be
    # bound to the occluded symmetric point on the far face.
    template, normals = _grain_template(box_size)
    box_ids = set(box_track_ids)
    track_body_votes = {}
    for f in solved:
        obs, ids_here = _frame_box_uv(tracks, box_ids, f, with_ids=True)
        if len(obs) == 0:
            continue
        world = (R_all[f] @ template.T).T + t_all[f]
        tpl_uv = project_points(world, K, cam_R[f], cam_t[f])
        visible = _visible_template(template, normals, R_all[f], t_all[f],
                                    cam_R[f], cam_t[f])
        tpl_uv = np.where(visible[:, None], tpl_uv, np.inf)
        assignment, dist = _assign(tpl_uv, obs, gate_px)
        for tid, ai, dd in zip(ids_here, assignment, dist):
            if ai >= 0:
                # Closer assignments outvote marginal ones for the same track.
                weight = 1.0 + float(np.clip(1.0 - dd / gate_px, 0.0, 1.0))
                track_body_votes.setdefault(tid, {})
                track_body_votes[tid][int(ai)] = \
                    track_body_votes[tid].get(int(ai), 0.0) + weight
    track_body = {tid: template[max(votes, key=votes.get)]
                  for tid, votes in track_body_votes.items()}

    # Fill marker-less frames with the nearest solved neighbour's pose.
    for f in range(n_frames):
        if R_all[f] is None:
            nb = min(solved, key=lambda g: abs(g - f))
            R_all[f], t_all[f] = R_all[nb].copy(), t_all[nb].copy()

    return BoxSolve(True, R=R_all, t=t_all, track_body_points=track_body,
                    reproj_px=float(np.mean(errors)) if errors else np.inf,
                    n_assigned=len(track_body),
                    markers_per_frame=markers_per_frame)


def _frame_box_uv(tracks, box_ids, f, with_ids=False):
    uvs, ids = [], []
    for tr in tracks:
        if tr.track_id not in box_ids:
            continue
        p = tr.at(f)
        if p is not None:
            uvs.append(p)
            ids.append(tr.track_id)
    if with_ids:
        return (np.asarray(uvs, dtype=np.float64), ids) if uvs \
            else (np.zeros((0, 2)), [])
    return np.asarray(uvs, dtype=np.float64) if uvs else np.zeros((0, 2))


def render_overlay(frame_bgr, box_size: np.ndarray, Rb, tb, cam_R, cam_t, K,
                   color=(0, 230, 180)):
    """Draw the virtual box driven by the solved pose onto a frame copy."""
    hx, hy, hz = np.asarray(box_size) / 2
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    world = (Rb @ corners.T).T + tb
    uv = project_points(world, K, cam_R, cam_t)
    img = frame_bgr.copy()
    if not np.isfinite(uv).all():
        return img
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), color, 2, cv2.LINE_AA)
    for p in uv:
        cv2.circle(img, tuple(p.astype(int)), 4, color, -1, cv2.LINE_AA)
    return img
