# Coordinate frames and per-layer motion definitions

All quantities are metric (metres) unless stated otherwise. The conventions are
also asserted in `dynamicsolver/geometry.py`.

## Frames

* **World frame `W`** — x right, y down, z forward. The synthetic scene places
  the background wall near `z = 5` and the camera a few metres in front of it.
* **Camera frame `C_k`** for frame `k` — same axis orientation as `W`, origin at
  the camera optical centre. A camera pose is the pair `(R_k, t_k)` such that

  ```
  X_c = R_k @ X_w + t_k
  ```

  with optical centre `C_k = -R_k.T @ t_k`. The camera looks down its own `+z`.
* **Image frame** — origin at the principal point, x right, y down, in pixels.
  Pinhole calibration `K`, projection `p ~ K (R X_w + t)`. Points at or behind
  the camera (`z <= 0`) project to infinity and are culled, never rasterised.
* **Box body frame `B`** — axes aligned with the cube edges, origin at the box
  centre. The box pose `(R_b,k, t_b,k)` maps body to world:

  ```
  X_w = R_b,k @ X_b + t_b,k
  ```

## The two motion classes (per-layer definitions)

The decomposition assigns every KLT track to exactly one of three labels:
`background`, `box`, or `deferred`.

* **Background layer (static in `W`).** A background track has one fixed world
  point across all frames; its image motion is induced solely by camera motion.
  These tracks support the camera solve: two-view essential matrix →
  triangulation → incremental PnP registration → pose-only bundle adjustment.
* **Box layer (rigidly moving in `W`).** A box track is a 3D body point
  `X_b` constant in the box frame, moving with `(R_b,k, t_b,k)`. Its image
  motion combines camera and independent box motion. These tracks never enter
  the background SfM; they drive the foreground PnP and the virtual overlay.
* **Deferred.** No trustworthy static/moving identity could be assigned (no
  model contrast, a same-speed crossing, or a hard-label flip across a
  crossing). Deferred tracks are reported, not silently labelled.

## Foreground pose in the camera frame

The box solver runs PnP in **camera coordinates** because that is the frame
`solvePnP` expects. Given the current background camera estimate, body points
are posed relative to the camera:

```
X_c = R_k @ (R_b X_b + t_b) + t_k
    = (R_k R_b) X_b + (R_k t_b + t_k)
```

PnP recovers the camera-from-body transform `R_c2b, t_c2b`; the world-frame box
pose is then recovered as

```
R_b = R_k.T @ R_c2b
t_b = R_k.T @ (t_c2b - t_k)
```

Hidden (back-facing) template points are removed by a face-normal visibility
test before refinement so they cannot compete with visible grain.

## Why relative box-camera pose is the accuracy invariant

SfM reconstructs the background only up to an arbitrary global **similarity**
(rotation, translation and scale). The box is solved inside that same
reconstructed frame, so comparing either pose directly against ground truth in a
fixed world frame is meaningless until a single Sim3 alignment is applied.

The quantity independent of that arbitrary gauge is the **relative** box-to-
camera rotation

```
Q_k = R_k @ R_b,k
```

Ground truth gives `Q*_k = R*_k @ R*_b,k`. The test suite compares `Q_k` against
`Q*_k` directly (no alignment), which isolates genuine box-solver error from
background-frame drift.

## Sim3 (Umeyama) alignment for global-frame evaluation

For reporting camera/box trajectories in the ground-truth frame, a single
similarity `B ~= s (R A) + t` is fit once over optical-centre trajectories.
With covariance `cov = Ac.T @ Bc` the rotation is `R = V S U.T` from the SVD
`cov = U diag(d) V.T` (using `U S V.T` here would return the transpose
rotation). A reflection check flips the last singular sign so `det(R) = +1`.

## ArUco fiducials and box symmetry

A bare textured cube admits a 180° (and full 24-way) orientation ambiguity.
Three ArUco markers with **distinct ids** in a non-collinear layout on one face
break the symmetry: the identified 2D-3D corner correspondences make the pose
unique. The marker corners are coplanar (rank 2 in 3D); the distinguishing
information is the distinct ids plus their non-trivial 2D arrangement, not 3D
volume. Fine grain points then refine the pose after an initial marker PnP.

## Scale

Two-view geometry fixes translation direction only, not magnitude. The declared
`background_depth_range` midpoint supplies metric scale by normalising the
median triangulated background depth to it, so the metric foreground box size
and the camera trajectory share one consistent frame.
