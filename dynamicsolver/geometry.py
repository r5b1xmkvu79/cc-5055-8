"""Geometric primitives for the dynamic-scene shot solver.

Coordinate conventions (documented here and in docs/coordinate_system.md):
  * World frame W: x right, y down, z forward; metric units (metres).
  * Camera frame C of frame k: same axes, origin at the camera optical centre.
  * A camera pose stores R (3x3), t (3,) such that X_c = R @ X_w + t.
    The optical centre is C = -R^T t.
  * Image frame: origin at the principal point, x right, y down, pixels.
  * Projection: p ~ K (R X_w + t).

The foreground box has its own body frame B whose axes are aligned with the
cube edges; its pose (R_b, t_b) maps B -> W, i.e. X_w = R_b X_b + t_b.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_calibration(width: int, height: int, fov_deg: float = 55.0) -> np.ndarray:
    """Pinhole K with the principal point at the image centre."""
    f = 0.5 * width / np.tan(0.5 * np.radians(fov_deg))
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def pose_from_look_at(eye: np.ndarray, target: np.ndarray, up_hint=np.array([0.0, -1.0, 0.0])):
    """Camera pose (R, t) looking from ``eye`` toward ``target``.

    The camera looks down +z in its own frame.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = target - eye
    z /= np.linalg.norm(z)
    # Right-handed camera frame with y down and z forward: x = up x z.
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    # Rows of R are the world axes expressed in camera coordinates.
    R = np.vstack([x, y, z])
    t = -R @ eye
    return R, t


def project_points(points_w: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project N world points; returns Nx2 pixel coordinates (points behind are inf)."""
    pts = np.atleast_2d(points_w)
    cam = (R @ pts.T).T + t
    depths = cam[:, 2]
    uv = (K @ cam.T).T
    out = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-12)
    out[depths <= 1e-6] = np.inf
    return out


def normalize_points(points: np.ndarray):
    """Isotropic Hartley normalization."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(pts - centroid, axis=1))
    T = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
    )
    homog = np.column_stack([pts, np.ones(len(pts))])
    normed = (T @ homog.T).T[:, :2]
    return normed, T


def _sample_8point(pts1, pts2, rng):
    idx = rng.choice(len(pts1), size=min(8, len(pts1)), replace=False)
    return pts1[idx], pts2[idx]


def _sample_homography(pts1, pts2, rng):
    idx = rng.choice(len(pts1), size=min(4, len(pts1)), replace=False)
    return pts1[idx], pts2[idx]


def fit_fundamental_8pt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Normalized 8-point algorithm; rank-2 enforced."""
    n1, T1 = normalize_points(pts1)
    n2, T2 = normalize_points(pts2)
    A = np.column_stack(
        [
            n2[:, 0] * n1[:, 0],
            n2[:, 0] * n1[:, 1],
            n2[:, 0],
            n2[:, 1] * n1[:, 0],
            n2[:, 1] * n1[:, 1],
            n2[:, 1],
            n1[:, 0],
            n1[:, 1],
            np.ones(len(n1)),
        ]
    )
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(F)
    s[-1] = 0.0
    F = u @ np.diag(s) @ vt
    F = T2.T @ F @ T1
    return F / F[2, 2]


def fit_homography_dlt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Normalized DLT homography from >= 4 correspondences."""
    n1, T1 = normalize_points(pts1)
    n2, T2 = normalize_points(pts2)
    A = np.zeros((2 * len(n1), 9))
    for i, (p, q) in enumerate(zip(n1, n2)):
        x, y = p
        u, v = q
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ H @ T1
    return H / H[2, 2]


def sampson_residuals(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Sampson distance per correspondence for a fundamental matrix."""
    p1 = np.column_stack([pts1, np.ones(len(pts1))])
    p2 = np.column_stack([pts2, np.ones(len(pts2))])
    Fp1 = (F @ p1.T).T
    Ftp2 = (F.T @ p2.T).T
    num = np.einsum("ij,ij->i", p2, Fp1) ** 2
    den = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def homography_residuals(H: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Symmetric transfer distance for a homography."""
    p1 = np.column_stack([pts1, np.ones(len(pts1))])
    warped = (H @ p1.T).T
    warped = warped[:, :2] / np.maximum(warped[:, 2:3], 1e-12)
    Hinv = np.linalg.inv(H)
    p2 = np.column_stack([pts2, np.ones(len(pts2))])
    back = (Hinv @ p2.T).T
    back = back[:, :2] / np.maximum(back[:, 2:3], 1e-12)
    return 0.5 * (np.linalg.norm(warped - pts2, axis=1) + np.linalg.norm(back - pts1, axis=1))


def gric(residuals: np.ndarray, n: int, d: int, k: int, sigma: float) -> float:
    """Geometric Robust Information Criterion (Torr 1997).

    d: manifold dimensions of the model (F -> 3, H -> 2), k: model parameters
    (F -> 7, H -> 8). Lower is better; balances fit against complexity, which
    matters when a moving plane dominates the frame and cardinality alone lies.
    """
    lam = 2.0
    rho = np.minimum(residuals ** 2 / (sigma ** 2), 2.0 * (lam - d))
    data_term = rho.sum()
    penalty = (k * np.log(n) + d * n * np.log(lam))
    return data_term + penalty


@dataclass
class TwoViewHypothesis:
    model: str  # "F" or "H"
    matrix: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray
    score: float
    ransac_iters: int


def ransac_model(
    kind: str,
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold_px: float,
    rng: np.random.Generator,
    max_iters: int = 500,
    confidence: float = 0.99,
) -> TwoViewHypothesis | None:
    """RANSAC for one model kind over a correspondence set."""
    n = len(pts1)
    min_size = 8 if kind == "F" else 4
    if n < min_size:
        return None
    best = None
    best_count = 0
    iters = max_iters
    done = 0
    while done < iters and done < max_iters:
        if kind == "F":
            s1, s2 = _sample_8point(pts1, pts2, rng)
            if len(s1) < 8:
                return None
            try:
                M = fit_fundamental_8pt(s1, s2)
                res = sampson_residuals(M, pts1, pts2)
            except np.linalg.LinAlgError:
                done += 1
                continue
        else:
            s1, s2 = _sample_homography(pts1, pts2, rng)
            try:
                M = fit_homography_dlt(s1, s2)
                res = homography_residuals(M, pts1, pts2)
            except np.linalg.LinAlgError:
                done += 1
                continue
        mask = res < threshold_px
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best = (M, mask, res)
            eps = 1.0 - count / n
            if eps < 1e-6:
                break
            iters = min(max_iters, int(np.log(1 - confidence) / np.log(max(1 - (1 - eps) ** min_size, 1e-12))))
        done += 1
    if best is None:
        return None
    M, mask, res = best
    # Refit on all inliers for a stable hypothesis.
    if mask.sum() >= min_size:
        try:
            if kind == "F":
                M = fit_fundamental_8pt(pts1[mask], pts2[mask])
                res = sampson_residuals(M, pts1, pts2)
            else:
                M = fit_homography_dlt(pts1[mask], pts2[mask])
                res = homography_residuals(M, pts1, pts2)
            mask = res < threshold_px
        except np.linalg.LinAlgError:
            pass
    score = gric(res, n, d=3 if kind == "F" else 2, k=7 if kind == "F" else 8,
                 sigma=max(threshold_px / 1.96, 0.25))
    return TwoViewHypothesis(kind, M, mask, res, score, done)


def essential_from_F(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    E = K.T @ F @ K
    u, _, vt = np.linalg.svd(E)
    E = u @ np.diag([1.0, 1.0, 0.0]) @ vt
    return E


def pose_from_essential(E: np.ndarray, K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray):
    """Cheirality disambiguation; returns R, t (unit scale) and triangulated points."""
    u, _, vt = np.linalg.svd(E)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vt) < 0:
        vt[-1, :] *= -1
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1, R2 = u @ W @ vt, u @ W.T @ vt
    t = u[:, 2]
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    best_in_front = -1
    p1 = pts1
    p2 = pts2
    for R, tt in candidates:
        P1 = K @ np.hstack([R, tt.reshape(3, 1)])
        X = triangulate(P0, P1, p1, p2)
        cam1 = X
        cam2 = (R @ X.T).T + tt
        in_front = int(((cam1[:, 2] > 0) & (cam2[:, 2] > 0)).sum())
        if in_front > best_in_front:
            best_in_front = in_front
            best = (R, tt, X)
    return best


def triangulate(P0: np.ndarray, P1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """Linear DLT triangulation, Nx2 inputs -> Nx3 Euclidean points."""
    out = np.zeros((len(pts0), 3))
    for i, (p0, p1) in enumerate(zip(pts0, pts1)):
        A = np.array(
            [
                p0[0] * P0[2] - P0[0],
                p0[1] * P0[2] - P0[1],
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
            ]
        )
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out[i] = X[:3] / X[3]
    return out


def relative_pose(R1, t1, R2, t2):
    """Relative camera pose (R_rel, t_rel) mapping frame-1 camera coords to frame 2."""
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel


def rotation_error(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    cosang = np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def translation_angle_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Direction error of translation (up-to-scale SfM has no metric t)."""
    na, nb = np.linalg.norm(t_est), np.linalg.norm(t_gt)
    if na < 1e-9 or nb < 1e-9:
        return 0.0 if na < 1e-9 and nb < 1e-9 else 180.0
    a = t_est / na
    b = t_gt / nb
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def rigid_transform_3d(A: np.ndarray, B: np.ndarray):
    """Best similarity (R, t, scale) mapping A onto B: B ~= scale*(R A) + t.

    Umeyama alignment. With covariance ``cov = Ac^T Bc`` the rotation is
    ``V S U^T`` (the SVD's right singular vectors first); using ``U S V^T``
    here returns the transpose rotation and silently misaligns every pose
    comparison downstream.
    """
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - mu_a, B - mu_b
    cov = Ac.T @ Bc / len(A)
    u, d, vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        S[2, 2] = -1.0
    R = vt.T @ S @ u.T
    var_a = (Ac ** 2).sum() / len(A)
    scale = float(np.trace(np.diag(d) @ S) / max(var_a, 1e-12))
    t = mu_b - scale * R @ mu_a
    return R, t, scale


def project_pnp_points(points_w: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    return project_points(points_w, K, R, t)
