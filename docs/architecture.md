# Architecture and methodology

## Pipeline stages

`dynamicsolver/pipeline.py::run_pipeline` chains the stages. Each stage returns
structured failure information instead of raising, so the GUI can surface the
failure and re-launch with adjusted priors.

1. **Synthesise / load clip** (`synth.py`, `pipeline.read_clip`).
   For synthetic scenarios, render the clip and save the ground-truth bundle.
2. **Track extraction** (`features.py`). KLT with forward-backward gating and
   redetection produces `Track` objects (frame list + pixel history). Occlusion
   naturally creates gaps and track termination.
3. **Motion decomposition** (`decomposition.py`). Assign every track to
   background, box, or deferred using multi-baseline geometric evidence plus
   declared priors.
4. **Background camera solve** (`camerasolve.py`). Incremental SfM over the
   background layer, metric-scaled from the declared depth range.
5. **Foreground box solve** (`objectpose.py`). ArUco-anchored PnP plus grain
   refinement in the solved camera frame; bind box tracks to body points.
6. **Composite** (`pipeline.build_composite`). Region-motion layer view beside
   the virtual box overlay driven by the solved poses.
7. **Evaluation** (`evaluation.py`). Layer F1, identity swaps, Sim3-aligned
   camera/box pose errors, reprojection RMSE.

## Why multi-baseline (anchor-pair) decomposition

Adjacent-frame geometry cannot separate a slowly moving foreground from the
background: over a one-frame gap their epipolar residuals overlap. The kernel
therefore gathers evidence on **anchor pairs** separated by `anchor_span`
frames, where the moving box accumulates a large multi-frame residual. Each
anchor pair is solved independently, then votes are integrated across each
track's whole lifetime.

For every anchor pair:

* correspondences are collected for tracks visible on both frames;
* **sequential RANSAC** fits a fundamental `F` and a planar homography `H`
  hypothesis to the dominant structure, removes its inliers, and repeats to find
  a second structure;
* each hypothesis keeps both an `F` and an `H` GRIC score so model *preference*
  is reported rather than assumed;
* structures are labelled from a bounded background-evidence vector combining
  spatial coverage, F-vs-H model preference and the GRIC margin. **Cardinality
  is deliberately absent.**

## The cardinality trap

Sequential RANSAC on a single pair tends to call the largest cluster the
background. In `sparse`, the foreground box supplies most of the tracks, so that
answer is wrong. Two guards prevent it:

* if a single structure explains at least
  `Priors.single_structure_defer_frac` (0.85) of a pair, there is no
  static/moving contrast, so the pair is **deferred** rather than labelled by
  size;
* the final verdict reports `ambiguous separation` instead of presenting a
  forced labelling as fact.

## Same-speed crossing and identity swaps

In `similar` the foreground crosses a static landmark while matching the
background's image speed. A crossing is flagged on an anchor pair when two
structures are both spatially close (`crossing_centroid_frac`) and have matched
flow magnitude (`crossing_velocity_frac`).

The crossing window itself is deferred (its structure votes are down-weighted
and routed to deferral). After labels are resolved, `_defer_crossing_flips`
finds any track whose **hard** anchor-window label flips background↔box between
the windows immediately before and after the crossing and defers just those
tracks. Tracks that stay consistent through the crossing keep their label.

`identity_swaps` counts only tracks that end in a hard label yet flip across a
crossing; a track the kernel ultimately deferred is an identity *hold*, not a
swap, even though its historical `window_labels` record competing evidence.

## Reliability verdicts and priors

All assumptions are explicit `Priors` fields, not hidden constants. The kernel
abstains (routes weight to `deferred`) when geometric evidence is strong enough
to make the abstention meaningful but unresolved; short, geometrically
unsupported tracks may still use the motion-magnitude and neighbourhood priors.
`_verdict` turns the resulting label distribution into `ok`,
`ambiguous separation`, or a stage-specific failure reason, with warnings.

## Background camera solve robustness

`camerasolve.py` seeds SfM from a long-baseline pair (a short baseline under
mostly-rotational motion makes the essential-matrix translation ill-conditioned).
To avoid a single unlucky RANSAC seed collapsing the whole solve (notable when
few static tracks survive):

* `_candidate_seed_pairs` returns several distinct, well-populated seed pairs;
* `_attempt_seed` builds each two-view seed (essential matrix → cheirality pose
  → depth filtering) and incrementally registers the remaining frames;
* after each successful PnP registration, newly visible tracks are
  triangulated (`_triangulate_new`), growing the 3D point pool so a sparse seed
  does not stall partway through the clip;
* the chain registering the most frames is kept, missing frames are
  interpolated, and a pose-only bundle adjustment refines camera poses while
  points stay fixed to avoid finite-difference drift.

## Determinism

* NumPy scene randomness uses a seeded generator; per-frame observation noise
  derives an independent generator from `(seed, frame_index)` so a frame's noise
  is identical on every render regardless of call order.
* OpenCV's global RNG is re-pinned per frame for anti-aliased rasterisation.
* Marker compositing warps onto a constant-border canvas plus a mask instead of
  `BORDER_TRANSPARENT`, whose internal RNG is not controlled by `setRNGSeed` and
  otherwise introduces intermittent one-pixel edge jitter.

## Module map

| Module             | Responsibility                                            |
|--------------------|-----------------------------------------------------------|
| `geometry.py`      | Calibration, projection, 8-point/DLT, RANSAC, GRIC, essential-pose, triangulation, Umeyama Sim3, rotation errors. |
| `synth.py`         | Deterministic scene/clip rendering and ground-truth bundle for the four scenarios. |
| `features.py`      | KLT track extraction and frame correspondences.           |
| `decomposition.py` | Multi-baseline layered decomposition, crossing detection, deferral, verdicts. |
| `layer_tracker.py` | Multi-hypothesis identity chains across same-speed windows; flow-field replay, reversible `IdentityCorrection` log, offline/online benchmark. |
| `layer_drift.py`   | Observation-only drift evidence chain; same-speed vs occlusion cause classification, stage-localised residuals, bounded logging. |
| `camerasolve.py`   | Background incremental SfM and pose-only bundle adjustment. |
| `objectpose.py`    | ArUco/grain box PnP, track-to-body binding, overlay rendering. |
| `evaluation.py`    | Ground-truth layer F1, identity swaps (tracker-aware), pose and reprojection metrics. |
| `pipeline.py`      | Stage orchestration, tracker commit and compositing shared by GUI and CLI. |
| `priors.py`        | Declared assumptions.                                     |
| `gui/`             | PySide6 shot-tracking window, drift panel and QThread worker. |
| `cli.py`           | Headless batch/CI helper mirroring the GUI controls.      |

See `docs/layer_identity_drift.md` for the same-speed/occlusion evidence chain,
the reversible correction semantics and the measured offline-vs-online
strategy choice.
