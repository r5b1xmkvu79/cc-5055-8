# 动态场景镜头求解器 · Dynamic-scene shot solver

A moving camera films a moving foreground prop (a textured, ArUco-tagged box).
This project reconstructs **both** motion classes from one clip:

* the **static-background camera trajectory** (incremental SfM on the background
  layer), and
* the **independent foreground box trajectory** (ArUco + grain PnP in the solved
  camera frame),

then drives a virtual-model overlay from the solved box pose. Every observation
is attributed to one motion layer, or explicitly **deferred** when the geometry
cannot separate the layers reliably.

## Graphical entry point (primary)

The GUI is the documented primary interface; the command line is only a
headless helper.

```bash
./run_gui.sh
# equivalent:
python -m dynamicsolver.gui.app
```

The shot-tracking window provides, in one place:

* **Controls** — scenario, RANSAC threshold, label margin, anchor span, feature
  budget. These map one-to-one onto kernel `Priors`, and a failed run can be
  re-launched with adjusted values so the failure feeds back into the kernel.
* **Motion-layer selector** — checkable background / box / deferred layers.
* **Observation inspector** — click a track to see its id, layer, confidence,
  frame span, and full trajectory, and exactly which observations support it.
* **Result area** — a region-motion decomposition view next to the
  virtual-model composite, with play / frame transport.
* **Reliability verdict** — the kernel's honest `ok / ambiguous separation /
  failure` verdict, never a forced label.

## Headless helper (CLI)

```bash
python -m dynamicsolver.cli --scenario {baseline,sparse,similar,occlusion} \
    --threshold 1.5 --label-margin 0.15 --max-corners 600 --out output/run
```

The CLI writes `summary.json` (layer F1, identity-swap count, camera/box pose
errors, reprojection RMSE, verdict) and a composite video. It exists for batch
runs and CI; every option mirrors a GUI control.

## Synthetic ground-truth scenarios

`dynamicsolver/synth.py` renders a moving camera filming a moving box and
saves the clip plus a ground-truth bundle (`ground_truth.npz`,
`observations.npz`). Renders are bit-reproducible: NumPy noise uses a
per-frame seeded generator and marker compositing avoids the
non-deterministic `BORDER_TRANSPARENT` warp path.

| Scenario   | Stress being tested                                                        |
|------------|----------------------------------------------------------------------------|
| `baseline` | Both layers well separated, dense background.                              |
| `sparse`   | Few static tracks; the foreground dominates (the **cardinality trap**).    |
| `similar`  | Foreground moves at the same image speed as background, then **crosses** a static landmark and re-separates. |
| `occlusion`| A pillar partially occludes the box.                                       |

## How the kernel avoids the two forbidden shortcuts

1. **It never assumes the most numerous matches are the background.** Layer
   identity comes from geometry (fundamental vs homography GRIC), spatial
   coverage, declared depth/flow priors and multi-view rigidity — never cluster
   size. When one structure explains nearly all correspondences with no
   contrast, the pair is *deferred*, not labelled background.
2. **A low-residual model is not treated as the unique real motion.** Both F
   and H models are fit everywhere; pairs whose evidence margin is too small,
   and tracks whose hard identity flips across a same-speed crossing, are
   deferred and surfaced in the verdict rather than force-assigned.

See `docs/architecture.md` for the pipeline and `docs/coordinate_system.md` for
frame conventions.

## Tests

```bash
PYTHONPATH=/workspace python -m unittest discover -s tests
```

The suite covers geometry/projection/triangulation, ArUco reliability and
symmetry breaking, the cardinality-trap verdict, same-speed crossing
detection, end-to-end completion for all four scenarios, and relative
box-camera pose accuracy.
