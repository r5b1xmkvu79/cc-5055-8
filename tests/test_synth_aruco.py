import unittest

import cv2
import numpy as np

from dynamicsolver.geometry import project_points
from dynamicsolver.synth import SCENARIOS, SceneConfig, SyntheticScene


class TestArUcoFiducials(unittest.TestCase):
    def setUp(self):
        self.scene = SyntheticScene(SceneConfig(scenario="baseline"))
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.12
        self.detector = cv2.aruco.ArucoDetector(self.scene.aruco_dictionary, params)
        self.valid_ids = {s["id"] for s in self.scene.aruco_specs}

    def test_three_distinct_markers_break_symmetry(self):
        # Three uniquely tagged, non-collinear markers on the face give an
        # unambiguous orientation: no in-plane rotation can map the id-tagged
        # layout onto itself. The points are coplanar (one face) so rank is 2;
        # the distinguishing information is the distinct ids plus a non-trivial
        # 2D point configuration, not 3D volume.
        self.assertEqual(len(self.valid_ids), 3)
        pts, _, ids = self.scene.aruco_corners3d()
        self.assertEqual(pts.shape, (12, 3))
        self.assertEqual(len(set(ids.tolist())), 3)
        # All corners share the face plane; within that plane they span area.
        self.assertEqual(np.linalg.matrix_rank(pts - pts.mean(0)), 2)
        centres = np.array([
            self.scene.aruco_body_corners(s).mean(0) for s in self.scene.aruco_specs
        ])
        self.assertEqual(np.linalg.matrix_rank(centres - centres.mean(0)), 2)

    def test_detection_reliability_and_subpixel_alignment(self):
        detected = {i: 0 for i in self.valid_ids}
        false_positives = 0
        align = []
        for f in range(self.scene.cfg.n_frames):
            img = self.scene.render_frame(f)[0]
            corners, ids, _ = self.detector.detectMarkers(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ids is None:
                continue
            R, t = self.scene.camera_trajectory(f)
            Rb, tb = self.scene.box_trajectory(f)
            for quad, mid in zip(corners, ids.ravel()):
                mid = int(mid)
                if mid not in self.valid_ids:
                    false_positives += 1
                    continue
                detected[mid] += 1
                spec = next(s for s in self.scene.aruco_specs if s["id"] == mid)
                gt = project_points(
                    (Rb @ self.scene.aruco_body_corners(spec).T).T + tb,
                    self.scene.K, R, t)
                align.append(np.linalg.norm(quad.reshape(4, 2) - gt, axis=1).mean())
        # Every marker visible on at least 90% of frames, no spurious ids.
        for mid, count in detected.items():
            self.assertGreaterEqual(count, int(0.9 * self.scene.cfg.n_frames))
        self.assertEqual(false_positives, 0)
        self.assertLess(np.median(align), 2.0)

    def test_markers_inside_frame_across_clip(self):
        for spec in self.scene.aruco_specs:
            for f in range(self.scene.cfg.n_frames):
                R, t = self.scene.camera_trajectory(f)
                Rb, tb = self.scene.box_trajectory(f)
                uv = project_points(
                    (Rb @ self.scene.aruco_body_corners(spec).T).T + tb,
                    self.scene.K, R, t)
                self.assertTrue(
                    (uv[:, 0] >= 0).all() and (uv[:, 0] < self.scene.cfg.width).all()
                    and (uv[:, 1] >= 0).all() and (uv[:, 1] < self.scene.cfg.height).all(),
                    f"marker {spec['id']} clipped on frame {f}")


class TestScenariosExist(unittest.TestCase):
    def test_scenario_names(self):
        self.assertEqual(set(SCENARIOS),
                         {"baseline", "sparse", "similar", "occlusion"})


if __name__ == "__main__":
    unittest.main()
