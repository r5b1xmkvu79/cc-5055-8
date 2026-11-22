"""Feature detection and KLT track extraction.

Tracks are the atomic observations fed to the decomposition kernel. Each
track stores the frames on which it was visible and the pixel coordinates;
occlusion naturally produces gaps and track termination rather than leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    frames: list = field(default_factory=list)
    uvs: list = field(default_factory=list)  # list of (2,) arrays

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def uv_array(self) -> np.ndarray:
        return np.asarray(self.uvs, dtype=np.float64)

    def at(self, frame: int):
        for f, uv in zip(self.frames, self.uvs):
            if f == frame:
                return uv
        return None


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def detect(gray: np.ndarray, max_corners: int = 600, quality: float = 0.02,
           min_distance: float = 9.0) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance,
        blockSize=7, useHarrisDetector=False,
    )
    if pts is None:
        return np.zeros((0, 2), dtype=np.float64)
    return pts.reshape(-1, 2)


def _lk(prev_gray, gray, pts):
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=(21, 21), maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    pb, st_b, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, p1, None,
        winSize=(21, 21), maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    if p1 is None:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    p1 = p1.reshape(-1, 2)
    pb = pb.reshape(-1, 2)
    fb = np.linalg.norm(pb - pts, axis=1)
    ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < 2.0)
    return p1, ok


def _inside(p, shape):
    return 0 <= p[0] < shape[1] and 0 <= p[1] < shape[0]


def extract_tracks(frames, min_active: int = 120, min_track_len: int = 3,
                   max_corners: int = 600) -> list[Track]:
    """KLT over the clip: forward-backward gating, redetection, full history."""
    grays = [_to_gray(f) for f in frames]
    tracks: list[Track] = []
    next_id = 0
    active: list[Track] = []
    active_pts = np.zeros((0, 2))

    for k, gray in enumerate(grays):
        if k == 0:
            active_pts = detect(gray, max_corners=max_corners)
            active = [Track(next_id + i, [0], [p]) for i, p in enumerate(active_pts)]
            next_id += len(active)
            tracks.extend(active)
            continue
        predicted, ok = (_lk(grays[k - 1], gray, active_pts) if len(active_pts)
                         else (np.zeros((0, 2)), np.zeros(0, dtype=bool)))
        survivors, survivor_pts = [], []
        for tr, p, good in zip(active, predicted, ok):
            if not good or not _inside(p, gray.shape):
                continue
            tr.frames.append(k)
            tr.uvs.append(p)
            survivors.append(tr)
            survivor_pts.append(p)
        active = survivors
        active_pts = np.asarray(survivor_pts, dtype=np.float64) if survivor_pts else np.zeros((0, 2))

        if len(active) < min_active:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for p in active_pts:
                cv2.circle(mask, tuple(np.round(p).astype(int)), 8, 0, -1)
            fresh = detect(gray, max_corners=max_corners)
            if len(fresh):
                yy = np.round(fresh[:, 1]).astype(int).clip(0, gray.shape[0] - 1)
                xx = np.round(fresh[:, 0]).astype(int).clip(0, gray.shape[1] - 1)
                fresh = fresh[mask[yy, xx] > 0]
                new_tracks = [Track(next_id + i, [k], [p]) for i, p in enumerate(fresh)]
                next_id += len(new_tracks)
                tracks.extend(new_tracks)
                active.extend(new_tracks)
                active_pts = np.vstack([active_pts, fresh]) if len(active_pts) else fresh

    return [t for t in tracks if len(t.frames) >= min_track_len]


def frame_correspondences(tracks: list[Track], f0: int, f1: int):
    """Matched (pts0, pts1, track_ids) for tracks visible on both frames."""
    pts0, pts1, ids = [], [], []
    for tr in tracks:
        a, b = tr.at(f0), tr.at(f1)
        if a is not None and b is not None:
            pts0.append(a)
            pts1.append(b)
            ids.append(tr.track_id)
    return (np.asarray(pts0, dtype=np.float64), np.asarray(pts1, dtype=np.float64),
            np.asarray(ids, dtype=int))
