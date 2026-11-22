"""Synthetic ground-truth renderer: moving camera filming a moving box.

The renderer is fully self-contained (numpy only) and emits, alongside the
video, every quantity needed to audit a decomposition: camera poses, box
poses and per-frame feature observations labelled with their true layer.

Scenarios
---------
baseline   : textured background, ordinary independent box motion.
sparse     : very few background features; the box fills most of the frame,
             so the largest correspondence set is the *foreground*.
similar    : camera pan + box translation tuned to equal image velocity for a
             window; the box crosses a static landmark, then re-separates.
occlusion  : a static pillar slab occludes parts of the box and background.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import make_calibration, pose_from_look_at, project_points

BG_LAYER = "background"
BOX_LAYER = "box"


@dataclass
class BoxSpec:
    size: np.ndarray  # full edge lengths (sx, sy, sz), metres
    grid: int = 3  # dots per edge on each face

    def face_points(self):
        """Stable body-frame feature points, one id per (face, grid cell)."""
        hx, hy, hz = self.size / 2.0
        pts, normals, face_ids = [], [], []
        faces = [
            (np.array([1.0, 0, 0]), hx, [0, 1, 2], [hy, hz]),
            (np.array([-1.0, 0, 0]), -hx, [0, 1, 2], [hy, hz]),
            (np.array([0, 1.0, 0]), hy, [2, 0, 1], [hz, hx]),
            (np.array([0, -1.0, 0]), -hy, [2, 0, 1], [hz, hx]),
            (np.array([0, 0, 1.0]), hz, [0, 1, 2], [hx, hy]),
            (np.array([0, 0, -1.0]), -hz, [0, 1, 2], [hx, hy]),
        ]
        fid = 0
        for normal, const, axes, extents in faces:
            for a in np.linspace(-extents[0], extents[0], self.grid):
                for b in np.linspace(-extents[1], extents[1], self.grid):
                    p = np.zeros(3)
                    p[axes[0]] = a
                    p[axes[1]] = b
                    p[axes[2]] = const
                    pts.append(p)
                    normals.append(normal)
                    face_ids.append(fid)
            fid += 1
        return np.array(pts), np.array(normals), np.array(face_ids)


@dataclass
class Pillar:
    """Axis-aligned vertical slab: x in [x0,x1], z in [z0,z1], y in [y0,y1]."""

    x0: float
    x1: float
    z0: float
    z1: float
    y0: float = -1.2
    y1: float = 1.2

    def intersect(self, origin: np.ndarray, ray: np.ndarray) -> bool:
        t_lo, t_hi = 1e-6, np.inf
        bounds = [(self.x0, self.x1), (self.y0, self.y1), (self.z0, self.z1)]
        for axis in range(3):
            if abs(ray[axis]) < 1e-12:
                if not (bounds[axis][0] <= origin[axis] <= bounds[axis][1]):
                    return False
                continue
            t1 = (bounds[axis][0] - origin[axis]) / ray[axis]
            t2 = (bounds[axis][1] - origin[axis]) / ray[axis]
            t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
            if t_lo > t_hi:
                return False
        return t_hi > 1e-6


def _segment_hits_box(origin: np.ndarray, end: np.ndarray, Rb: np.ndarray, tb: np.ndarray,
                      half: np.ndarray) -> bool:
    """Does segment origin->end intersect the oriented box? Test in body frame."""
    o = Rb.T @ (origin - tb)
    e = Rb.T @ (end - tb)
    d = e - o
    t_lo, t_hi = 0.0, 1.0
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
    return t_hi > 0.0 and t_lo < 1.0


@dataclass
class SceneConfig:
    scenario: str = "baseline"
    width: int = 960
    height: int = 720
    fov_deg: float = 55.0
    n_frames: int = 40
    noise_px: float = 0.25
    seed: int = 7
    box_size: np.ndarray = field(default_factory=lambda: np.array([0.9, 1.2, 0.7]))
    out_dir: str = "output/synth"

    @property
    def is_sparse(self) -> bool:
        return self.scenario == "sparse"


class SyntheticScene:
    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # OpenCV drawing/marker routines draw from a process-global RNG that is
        # otherwise seeded from entropy. Pin it to the configured seed so a
        # scene renders bit-identically no matter which other scenarios ran
        # first in the same process (the test suite constructs many scenes).
        cv2.setRNGSeed(int(cfg.seed))
        self.K = make_calibration(cfg.width, cfg.height, cfg.fov_deg)
        self.box = BoxSpec(cfg.box_size, grid=4 if not cfg.is_sparse else 5)
        self.box_pts, self.box_normals, self.box_faces = self.box.face_points()
        self.pillars = [Pillar(0.05, 0.22, 2.6, 3.2)] if cfg.scenario == "occlusion" else []
        # Static wall grain: fine texture rigidly attached to the world so KLT
        # has aperture-friendly structure between the sparse landmark dots.
        # Spread over back, side, floor and ceiling surfaces to give genuine
        # 3D depth variation and well-conditioned translation parallax.
        grain_pts = []
        n_back, n_side, n_floor = 900, 300, 350
        grain_pts.append(np.column_stack([
            self.rng.uniform(-3.4, 3.4, n_back),
            self.rng.uniform(-2.2, 2.2, n_back),
            np.full(n_back, 5.0 - 1e-3),
        ]))
        for x_wall in (-3.4, 3.4):
            grain_pts.append(np.column_stack([
                np.full(n_side, x_wall),
                self.rng.uniform(-2.0, 2.0, n_side),
                self.rng.uniform(1.2, 4.9, n_side),
            ]))
        for y_surf in (-2.0, 2.0):
            grain_pts.append(np.column_stack([
                self.rng.uniform(-3.4, 3.4, n_floor),
                np.full(n_floor, y_surf),
                self.rng.uniform(1.0, 5.0, n_floor),
            ]))
        grain_world = np.vstack(grain_pts)
        self.wall_grain_xy = grain_world[:, :2]
        self.wall_grain_z = grain_world[:, 2]
        self.wall_grain = self.rng.uniform(0, 255, size=(len(grain_world),)).astype(np.uint8)
        # Box grain reuses the PnP template grid (with intermediate grid
        # density) so rendered texture lies exactly on solvable body points.
        grain_spec = BoxSpec(cfg.box_size, grid=8)
        self.box_grain, self.box_grain_normals, _ = grain_spec.face_points()
        # ArUco fiducials give the box solver *identified* 2D-3D corner
        # correspondences, removing the nearest-template assignment ambiguity
        # that lets a 180-degree-symmetric pose fit an unidentified checker.
        # Three markers with distinct ids span the camera-facing faces; their
        # body-frame corner layout has no rotational self-symmetry.
        hx, hy, hz = cfg.box_size / 2.0
        self.half_extents = np.array([hx, hy, hz])
        self.aruco_dict_id = cv2.aruco.DICT_4X4_50
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        # The -z face projects partly off the left image edge for the whole
        # clip (the box starts close and left of centre). A grid probe shows
        # only body fractions x_f <= ~0.2, y_f in [-0.8, 0.3] stay on-screen in
        # every frame, so all three markers live inside that safe region and
        # form a non-collinear triangle; marker corners were validated there.
        self.aruco_specs = [
            dict(id=0, side=0.16,
                 center=np.array([0.0 * hx, -0.62 * hy, -hz - 1e-4]),
                 normal=np.array([0.0, 0.0, -1.0]),
                 u=np.array([1.0, 0.0, 0.0]), v=np.array([0.0, 1.0, 0.0])),
            dict(id=1, side=0.16,
                 center=np.array([0.05 * hx, -0.12 * hy, -hz - 1e-4]),
                 normal=np.array([0.0, 0.0, -1.0]),
                 u=np.array([1.0, 0.0, 0.0]), v=np.array([0.0, 1.0, 0.0])),
            dict(id=2, side=0.16,
                 center=np.array([-0.18 * hx, 0.30 * hy, -hz - 1e-4]),
                 normal=np.array([0.0, 0.0, -1.0]),
                 u=np.array([1.0, 0.0, 0.0]), v=np.array([0.0, 1.0, 0.0])),
        ]
        self._build_background()
        self.landmark_id = self._place_landmark()

    def aruco_body_corners(self, spec):
        """Four body-frame marker corners in ArUco order: TL, TR, BR, BL."""
        c, h = spec["center"], spec["side"] / 2.0
        u, v = spec["u"], spec["v"]
        return np.array([
            c - h * u - h * v,
            c + h * u - h * v,
            c + h * u + h * v,
            c - h * u + h * v,
        ])

    def aruco_corners3d(self):
        """All marker corners (Nx3) and parallel normals (Nx3), id-tagged."""
        pts, nrm, ids = [], [], []
        for spec in self.aruco_specs:
            bc = self.aruco_body_corners(spec)
            pts.append(bc)
            nrm.append(np.tile(spec["normal"], (4, 1)))
            ids.extend([spec["id"]] * 4)
        return np.vstack(pts), np.vstack(nrm), np.asarray(ids)

    # ------------------------------------------------------------------ scene
    def _build_background(self):
        """Static world landmarks spread over back and side walls."""
        rng = self.rng
        pts = []
        if self.cfg.is_sparse:
            xs = rng.uniform(-2.4, 2.4, 10)
            ys = rng.uniform(-1.4, 1.4, 10)
            for x, y in zip(xs, ys):
                pts.append([x, y, 5.0])
        else:
            # Back wall at z=5
            for x in np.linspace(-3.0, 3.0, 11):
                for y in np.linspace(-2.0, 2.0, 8):
                    pts.append([x + rng.normal(0, 0.05), y + rng.normal(0, 0.05), 5.0])
            # Side walls for parallax
            for z in np.linspace(1.5, 4.8, 6):
                for y in np.linspace(-1.8, 1.8, 5):
                    pts.append([-3.4, y, z])
                    pts.append([3.4, y, z])
        self.bg_pts = np.array(pts)
        self.n_bg = len(self.bg_pts)
        # Box feature ids follow background ids.
        self.box_id_offset = self.n_bg + 1  # +1 reserves the landmark id

    def _place_landmark(self) -> int:
        """A distinctive static landmark the moving box crosses in 'similar'."""
        # Chosen so that, with the similar-scenario trajectories, the box
        # overlaps it around the middle of the clip.
        self.landmark_world = np.array([0.35, -0.15, 5.0])
        self.bg_pts = np.vstack([self.bg_pts, self.landmark_world])
        self.n_bg = len(self.bg_pts)
        return self.n_bg - 1

    # ------------------------------------------------------------- trajectories
    def camera_trajectory(self, k: int):
        n = self.cfg.n_frames
        s = k / max(n - 1, 1)
        eye0 = np.array([-1.3, 0.25, 0.6])
        # Strong forward component keeps translation parallax well conditioned
        # for essential-matrix recovery alongside the lateral motion.
        eye = eye0 + np.array([0.9 * s, 0.05 * np.sin(2 * np.pi * s), 1.0 * s])
        if self.cfg.scenario == "similar":
            ang = -0.16 * s
            target = eye + np.array([np.sin(ang), 0.0, np.cos(ang)])
        else:
            target = np.array([0.0, 0.0, 3.6])
        return pose_from_look_at(eye, target)

    def box_trajectory(self, k: int):
        n = self.cfg.n_frames
        s = k / max(n - 1, 1)
        Rb = np.eye(3)
        if self.cfg.scenario == "similar":
            # Angular pan rate of the camera (~theta = -0.16 rad across clip).
            theta_dot = 0.16 / (n - 1)
            z_box = 3.0
            v_match = theta_dot * z_box  # equal image velocity condition
            t1, t2 = 0.30, 0.68
            x = np.zeros_like(s)
            if s < t1:
                x = -1.1 + (v_match * (n - 1) * t1) * (s / t1)
            elif s <= t2:
                x = -1.1 + v_match * (n - 1) * s
            else:
                x_match_end = -1.1 + v_match * (n - 1) * t2
                x = x_match_end + 2.4 * v_match * (n - 1) * (s - t2)
            tb = np.array([x, -0.1 + 0.05 * np.sin(2 * np.pi * s), z_box])
            ang = 0.25 * s
            c, sn = np.cos(ang), np.sin(ang)
            Rb = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        elif self.cfg.scenario == "sparse":
            # Large box close to camera: dominates the image throughout.
            tb = np.array([-0.9 + 0.9 * s, 0.05 * np.sin(2 * np.pi * s), 2.2])
            ang = 0.4 * s
            c, sn = np.cos(ang), np.sin(ang)
            Rb = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        else:
            tb = np.array([1.0 - 1.1 * s, 0.15 * np.cos(2 * np.pi * s), 3.0])
            ang = 0.45 * s
            c, sn = np.cos(ang), np.sin(ang)
            Rb = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        return Rb, tb

    # -------------------------------------------------------------- visibility
    def _ray(self, center, world_pt):
        ray = world_pt - center
        return ray / np.linalg.norm(ray)

    def visible_observations(self, k: int):
        """Return list of dicts: id, uv, layer, world, body_point (box only)."""
        R, t = self.camera_trajectory(k)
        Rb, tb = self.box_trajectory(k)
        center = -R.T @ t
        obs = []
        # Background.
        uv_bg = project_points(self.bg_pts, self.K, R, t)
        for i, (w, uv) in enumerate(zip(self.bg_pts, uv_bg)):
            if not np.isfinite(uv).all() or not self._in_frame(uv):
                continue
            ray = self._ray(center, w)
            if any(p.intersect(center, ray) for p in self.pillars):
                continue
            if _segment_hits_box(center, w, Rb, tb, self.cfg.box_size / 2):
                continue
            obs.append(dict(id=i, uv=uv, layer=BG_LAYER, world=w))
        # Box features.
        world_box = (Rb @ self.box_pts.T).T + tb
        uv_box = project_points(world_box, self.K, R, t)
        for i, (bpt, normal_b, uv) in enumerate(zip(self.box_pts, self.box_normals, uv_box)):
            if not np.isfinite(uv).all() or not self._in_frame(uv):
                continue
            normal_w = Rb @ normal_b
            view = center - (Rb @ bpt + tb)
            if np.dot(normal_w, view) <= 0:
                continue  # back-facing
            w = Rb @ bpt + tb
            ray = self._ray(center, w)
            if any(p.intersect(center, ray) for p in self.pillars):
                continue
            obs.append(dict(id=self.box_id_offset + i, uv=uv, layer=BOX_LAYER,
                            world=w, body=bpt))
        return obs, (R, t), (Rb, tb), center

    def _in_frame(self, uv, margin=6.0):
        return margin <= uv[0] < self.cfg.width - margin and margin <= uv[1] < self.cfg.height - margin

    # ------------------------------------------------------------------ render
    def render_frame(self, k: int):
        # Re-pin OpenCV's global RNG per frame so anti-aliased rasterization is
        # reproducible regardless of how much RNG state earlier frames or other
        # scenarios consumed in the same process.
        cv2.setRNGSeed(int(self.cfg.seed) + k)
        # Per-frame independent NumPy RNG: observation noise for frame k must be
        # identical on every render, not shifted by how many frames rendered
        # before it in this process (the test suite re-renders scenes).
        frame_rng = np.random.default_rng(
            np.random.SeedSequence([int(self.cfg.seed), int(k)]))
        obs, (R, t), (Rb, tb), center = self.visible_observations(k)
        img = np.full((self.cfg.height, self.cfg.width, 3), 26, dtype=np.uint8)
        self._draw_wall_grid(img, R, t)
        self._draw_wall_grain(img, R, t, Rb, tb, center)
        for pillar in self.pillars:
            self._draw_pillar(img, pillar, R, t)
        self._draw_box(img, Rb, tb, R, t)
        self._draw_box_grain(img, Rb, tb, R, t, center)
        for o in obs:
            uv = o["uv"] + frame_rng.normal(0, self.cfg.noise_px, 2)
            is_landmark = o["id"] == self.landmark_id
            color = (0, 220, 255) if is_landmark else (
                (90, 200, 255) if o["layer"] == BG_LAYER else (200, 120, 255)
            )
            radius = 5 if is_landmark else 3
            cv2.circle(img, tuple(np.round(uv).astype(int)), radius, color, -1, cv2.LINE_AA)
            if is_landmark:
                cv2.drawMarker(img, tuple(np.round(uv).astype(int)), (255, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 14, 2)
        return img, obs, (R, t), (Rb, tb)

    def _draw_wall_grid(self, img, R, t):
        z = 5.0
        for x in np.linspace(-3.4, 3.4, 15):
            p0 = project_points(np.array([[x, -2.2, z], [x, 2.2, z]]), self.K, R, t)
            if np.isfinite(p0).all():
                cv2.line(img, tuple(p0[0].astype(int)), tuple(p0[1].astype(int)), (48, 48, 58), 1)
        for y in np.linspace(-2.2, 2.2, 10):
            p0 = project_points(np.array([[-3.4, y, z], [3.4, y, z]]), self.K, R, t)
            if np.isfinite(p0).all():
                cv2.line(img, tuple(p0[0].astype(int)), tuple(p0[1].astype(int)), (48, 48, 58), 1)

    def _draw_wall_grain(self, img, R, t, Rb, tb, center):
        world = np.column_stack([self.wall_grain_xy, self.wall_grain_z])
        uv = project_points(world, self.K, R, t)
        finite = np.isfinite(uv).all(axis=1)
        for i in np.where(finite)[0]:
            p = uv[i]
            if not (0 <= p[0] < self.cfg.width and 0 <= p[1] < self.cfg.height):
                continue
            w = world[i]
            if _segment_hits_box(center, w, Rb, tb, self.cfg.box_size / 2):
                continue
            if any(pl.intersect(center, self._ray(center, w)) for pl in self.pillars):
                continue
            shade = int(self.wall_grain[i])
            cv2.circle(img, (int(p[0]), int(p[1])), 1, (shade, shade, shade + 6), -1)

    def _draw_box_grain(self, img, Rb, tb, R, t, center):
        world = (Rb @ self.box_grain.T).T + tb
        uv = project_points(world, self.K, R, t)
        finite = np.isfinite(uv).all(axis=1)
        # Marker quads in the current image: grain must never overwrite their
        # bits, or a foreground dot on the black core corrupts the ArUco id.
        marker_quads = []
        for spec in self.aruco_specs:
            c_w = (Rb @ spec["center"]) + tb
            if not np.isfinite(project_points(c_w[None], self.K, R, t)).all():
                continue
            if np.dot(Rb @ spec["normal"], center - c_w) <= 0:
                continue
            if any(pl.intersect(center, self._ray(center, c_w)) for pl in self.pillars):
                continue
            dst = project_points(
                (Rb @ self.aruco_body_corners(spec).T).T + tb, self.K, R, t)
            if np.isfinite(dst).all():
                marker_quads.append(dst.astype(np.float32))
        for i in np.where(finite)[0]:
            p = uv[i]
            if not (0 <= p[0] < self.cfg.width and 0 <= p[1] < self.cfg.height):
                continue
            normal_b = self.box_grain_normals[i]
            w = world[i]
            if np.dot(Rb @ normal_b, center - w) <= 0:
                continue
            if any(pl.intersect(center, self._ray(center, w)) for pl in self.pillars):
                continue
            if any(cv2.pointPolygonTest(q, (float(p[0]), float(p[1])), False) >= 0
                   for q in marker_quads):
                continue
            cv2.circle(img, (int(p[0]), int(p[1])), 1, (150, 130, 200), -1)
        # Identified ArUco fiducials. Each marker is warped onto its face with
        # a perspective homography derived from the projected body corners, so
        # ArUco detection downstream recovers both the four corners and the id
        # (an unambiguous 2D-3D correspondence for the box solve).
        for spec in self.aruco_specs:
            c_b, n_b = spec["center"], spec["normal"]
            c_w = (Rb @ c_b) + tb
            n_w = Rb @ n_b
            if not np.isfinite(project_points(c_w[None], self.K, R, t)).all():
                continue
            if np.dot(n_w, center - c_w) <= 0:
                continue
            if any(pl.intersect(center, self._ray(center, c_w)) for pl in self.pillars):
                continue
            body_corners = self.aruco_body_corners(spec)
            world_corners = (Rb @ body_corners.T).T + tb
            dst = project_points(world_corners, self.K, R, t)
            if not np.isfinite(dst).all():
                continue
            core = cv2.aruco.generateImageMarker(
                self.aruco_dictionary, spec["id"], 140)
            # White quiet-zone ring: the dark box face (gray ~65) otherwise
            # swallows the marker's black outline and no quad contour forms.
            # The ring is excluded from the body-corner mapping below, so the
            # detector's four corners still match the declared 3D corners.
            ring = 18
            tile = cv2.copyMakeBorder(core, ring, ring, ring, ring,
                                      cv2.BORDER_CONSTANT, value=255)
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            h = tile.shape[0] - 1
            src = np.array([[ring, ring], [h - ring, ring],
                            [h - ring, h - ring], [ring, h - ring]],
                           dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
            # BORDER_TRANSPARENT in this OpenCV build draws from an internal RNG
            # that setRNGSeed does not control, giving intermittent one-pixel
            # edge jitter (248 vs 255) and non-reproducible clips. Warp onto a
            # black canvas with a constant border and matching mask, then
            # composite through the mask to reproduce the same overlay exactly.
            warped = cv2.warpPerspective(
                tile, H, (self.cfg.width, self.cfg.height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(
                np.full(tile.shape[:2], 255, np.uint8), H,
                (self.cfg.width, self.cfg.height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)
            img[mask > 0] = warped[mask > 0]

    def _draw_pillar(self, img, pillar: Pillar, R, t):
        corners = np.array([
            [pillar.x0, pillar.y0, pillar.z0], [pillar.x1, pillar.y0, pillar.z0],
            [pillar.x1, pillar.y1, pillar.z0], [pillar.x0, pillar.y1, pillar.z0],
            [pillar.x0, pillar.y0, pillar.z1], [pillar.x1, pillar.y0, pillar.z1],
            [pillar.x1, pillar.y1, pillar.z1], [pillar.x0, pillar.y1, pillar.z1],
        ])
        uv = project_points(corners, self.K, R, t)
        if not np.isfinite(uv).all():
            return
        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
        cam = (R @ corners.T).T + t
        ordered = sorted(faces, key=lambda f: -cam[list(f), 2].mean())
        for f in ordered:
            poly = uv[list(f)].astype(int)
            cv2.fillPoly(img, [poly], (96, 96, 110))
            cv2.polylines(img, [poly], True, (140, 140, 160), 1)

    def _draw_box(self, img, Rb, tb, R, t):
        hx, hy, hz = self.cfg.box_size / 2
        corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ])
        world = (Rb @ corners.T).T + tb
        uv = project_points(world, self.K, R, t)
        if not np.isfinite(uv).all():
            return
        cam = (R @ world.T).T + t
        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
        ordered = sorted(faces, key=lambda f: -cam[list(f), 2].mean())
        for f in ordered:
            poly = uv[list(f)].astype(int)
            cv2.fillPoly(img, [poly], (60, 52, 92))
            cv2.polylines(img, [poly], True, (170, 140, 230), 2)


def generate(cfg: SceneConfig) -> dict:
    """Render the full clip and persist video + ground-truth bundle."""
    out = Path(cfg.out_dir) / cfg.scenario
    out.mkdir(parents=True, exist_ok=True)
    scene = SyntheticScene(cfg)
    video_path = out / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             (cfg.width, cfg.height))
    if not writer.isOpened():
        video_path = out / "clip.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 15,
                                 (cfg.width, cfg.height))
    all_obs, cam_gt, box_gt = [], [], []
    for k in range(cfg.n_frames):
        img, obs, (R, t), (Rb, tb) = scene.render_frame(k)
        writer.write(img)
        all_obs.append(obs)
        cam_gt.append((R.copy(), t.copy()))
        box_gt.append((Rb.copy(), tb.copy()))
    writer.release()

    bundle = dict(
        scenario=cfg.scenario,
        K=scene.K,
        width=cfg.width,
        height=cfg.height,
        n_frames=cfg.n_frames,
        box_size=cfg.box_size,
        box_id_offset=scene.box_id_offset,
        landmark_id=scene.landmark_id,
        box_body_points=scene.box_pts,
        bg_world=scene.bg_pts,
        noise_px=cfg.noise_px,
        video_path=str(video_path),
    )
    np.savez_compressed(out / "ground_truth.npz", **bundle,
                        cam_R=np.array([r for r, _ in cam_gt]),
                        cam_t=np.array([tt for _, tt in cam_gt]),
                        box_R=np.array([r for r, _ in box_gt]),
                        box_t=np.array([tt for _, tt in box_gt]))
    # Observations are ragged; store via npz object arrays separately.
    np.savez_compressed(
        out / "observations.npz",
        ids=np.array([[o["id"] for o in fr] for fr in all_obs], dtype=object),
        uv=np.array([[o["uv"] for o in fr] for fr in all_obs], dtype=object),
        layers=np.array([[o["layer"] for o in fr] for fr in all_obs], dtype=object),
    )
    return dict(scene=scene, out_dir=str(out), video_path=str(video_path),
                observations=all_obs, cam_gt=cam_gt, box_gt=box_gt)


SCENARIOS = ("baseline", "sparse", "similar", "occlusion")
