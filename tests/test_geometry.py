import unittest

import numpy as np

from dynamicsolver.geometry import (
    make_calibration,
    pose_from_look_at,
    project_points,
    rotation_error,
    rigid_transform_3d,
    triangulate,
)


class TestCameraFrame(unittest.TestCase):
    def test_look_at_is_right_handed(self):
        # The original bug produced det(R) = -1 (a reflection). The camera
        # frame must always be a proper right-handed rotation.
        for eye, target in [
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([-1.3, 0.25, 0.6]), np.array([0.0, 0.0, 3.6])),
            (np.array([2.0, -1.0, 1.0]), np.array([0.0, 0.5, 4.0])),
        ]:
            R, t = pose_from_look_at(eye, target)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_look_at_optical_axis_points_at_target(self):
        eye = np.array([1.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 5.0])
        R, t = pose_from_look_at(eye, target)
        # Third row of R is the camera z axis; it should align with +world z.
        np.testing.assert_allclose(R[2], np.array([0.0, 0.0, 1.0]), atol=1e-9)
        np.testing.assert_allclose(-R.T @ t, eye, atol=1e-9)

    def test_projection_round_trip_and_behind_camera(self):
        K = make_calibration(320, 240, 55.0)
        R, t = pose_from_look_at(np.array([0.0, 0.0, 0.0]),
                                 np.array([0.0, 0.0, 1.0]))
        front = np.array([[0.0, 0.0, 3.0]])
        uv = project_points(front, K, R, t)[0]
        np.testing.assert_allclose(uv, np.array([160.0, 120.0]), atol=1e-6)
        behind = np.array([[0.0, 0.0, -3.0]])
        uvb = project_points(behind, K, R, t)[0]
        self.assertFalse(np.isfinite(uvb).all())

    def test_rotation_error_identity(self):
        self.assertAlmostEqual(rotation_error(np.eye(3), np.eye(3)), 0.0)

    def test_rigid_transform_recovers_similarity(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(30, 3))
        angle = 0.4
        R_true = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        t_true = np.array([1.0, -2.0, 0.5])
        s_true = 2.5
        B = (s_true * (R_true @ A.T)).T + t_true
        R, t, s = rigid_transform_3d(A, B)
        np.testing.assert_allclose((s * (R @ A.T)).T + t, B, atol=1e-8)
        self.assertAlmostEqual(s, s_true, places=6)
        self.assertAlmostEqual(rotation_error(R, R_true), 0.0, places=6)
        np.testing.assert_allclose(t, t_true, atol=8)


class TestTriangulation(unittest.TestCase):
    def test_triangulates_known_point(self):
        K = make_calibration(320, 240, 55.0)
        R0, t0 = pose_from_look_at(np.array([0.0, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]))
        R1, t1 = pose_from_look_at(np.array([0.2, 0.0, 0.0]),
                                   np.array([0.0, 0.0, 1.0]))
        X = np.array([[0.1, -0.2, 4.0]])
        p0 = project_points(X, K, R0, t0)
        p1 = project_points(X, K, R1, t1)
        P0 = K @ np.hstack([R0, t0.reshape(3, 1)])
        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
        Xhat = triangulate(P0, P1, p0, p1)
        np.testing.assert_allclose(Xhat, X, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
