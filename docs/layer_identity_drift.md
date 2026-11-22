# Layer-identity drift: same-speed crossings, occlusion and the evidence chain

This document covers the two modules added to diagnose and correct layer
identity drift when the foreground and background move at the same image
speed:

* `dynamicsolver/layer_tracker.py` — multi-hypothesis identity chains with
  reversible corrections, and
* `dynamicsolver/layer_drift.py` — an observation-only diagnostic evidence
  chain that distinguishes same-speed motion from crossing occlusion.

Neither module opens a ground-truth bundle. Every quantity below is computed
from KLT pixel trajectories and the decomposition's own window labels and
fitted structures, so the same code runs on real footage.

## The failure mode

In the `similar` scenario the box is tuned to match the camera-induced
background image velocity for a span of frames while it crosses a static
landmark. Across that span the two layers' epipolar residuals overlap, so the
anchor-pair decomposition in `decomposition.py` defers the crossing window.
The failure is not the deferral — it is what happens on either side:

* tracks whose hard window label flips `background <-> box` across the
  crossing are candidate identity exchanges;
* a single best-label vote cannot represent the fact that two explanations
  remain plausible until the layers re-separate.

Measured on the rendered `similar` clip (seed 7, 40 frames, 482 tracks), the
crossing window is `(24, 32)` and 13 tracks enter the candidate set: 12 with a
hard pre/post flip and one forced to deferred across the window.

## Multi-hypothesis replay

For every candidate track the tracker builds two `LayerHypothesis` objects
from the frame immediately before the same-speed window:

* **hold** — keep the pre-window hard identity through the crossing;
* **swap** — attach the other hard layer.

Both hypotheses store the same replayable trajectory fragment (frame -> pixel
position) plus observation-only evidence. The deciding evidence is the
distance from the track's per-frame step vectors to each layer's **flow
field**: the median per-frame displacement of the pre-window hard-labelled
cohort over the crossing span. A track follows the field of its true layer; a
stitched identity does not. The decision statistic is the RMS distance
difference to the two fields, in pixels.

Corroborating (never deciding) evidence recorded on each hypothesis:

* constant-velocity prediction residual in both replay directions,
* the pre/post-window flow-direction bend angle,
* the decomposition's own pre/post-window hard labels.

A correction is committed only when the flow-field margin reaches
`decision_margin_px` (default 4.0 px, measured — see below). Below the margin
the observations genuinely cannot separate the explanations: the track stays
`deferred`, both hypotheses are retained, and the event is logged as
unresolved. This is deliberately distinct from a **tracker fault**, which is
raised only when required replay state (a neighbour anchor window, a
hypothesis fragment) is missing. No default label is ever applied to cover a
fault.

## Reversible correction log

Every decision is an `IdentityCorrection` holding the original layer, the
applied layer (`None` for unresolved), both hypotheses, the reason string,
the flow distances and an `inverse` map. Applying the inverses in reverse
order restores the exact pre-tracker assignment map; this is asserted in
`tests/test_layer_tracker.py::test_correction_log_is_fully_reversible`.

## Offline reconnection vs. online multiple hypotheses: measured choice

The brief requires the strategy to be selected from measurement rather than
asserted. `benchmark_strategies()` runs both on the identical observations:

| Strategy | corrections | decided | unresolved | faults | median flow margin (px) | mean runtime (ms) |
|----------|-------------|---------|------------|--------|--------------------------|-------------------|
| offline  | 13          | 7       | 6          | 0      | 6.86                     | 3.99 ± 0.51       |
| online   | 13          | 7       | 6          | 0      | 6.86                     | 3.84 ± 0.14       |

Numbers are from 20 repeats on the `similar` clip. Both strategies consume
the same evidence and reach the same decided/unresolved partition; runtime is
statistically identical (~4 ms against a ~9 s full pipeline). The difference
is *commit timing*: online freezes a decision at the first post-window anchor
pair and can never upgrade an early tie once more frames arrive, whereas
offline keeps every hypothesis until the clip ends.

**Selected strategy: offline reconnection** (the pipeline default), because

1. the measured runtime advantage of online is negligible (0.15 ms) and far
   below pipeline noise, while offline retains the ability to resolve ties
   from the full continuation;
2. offline decisions are bit-deterministic across repeated runs (asserted in
   the benchmark probe), and the entire correction log remains reversible;
3. the clip is processed as a batch by both the GUI worker and the CLI, so no
   consumer needs a per-frame hard commit — an online-only mode would buy
   nothing but early, unrevisable labels.

The online implementation is retained behind `strategy="online"` purely so the
trade-off stays measurable; it is not wired into the pipeline.

### Why 4.0 px decision margin

A margin sweep on the same clip shows the honest-deferral trade-off:

| margin (px) | decided | correct | wrong | unresolved |
|-------------|---------|---------|-------|------------|
| 0.5         | 13      | 8       | 5     | 0          |
| 1.5         | 11      | 8       | 3     | 2          |
| 3.0         | 10      | 8       | 2     | 3          |
| **4.0**     | **7**   | **7**   | **0** | **6**      |

At 4.0 px every committed correction matches the ground-truth layer on this
clip, and the six observationally indistinguishable tracks stay deferred with
both hypotheses open. The kernel's existing verdict surfaces the resulting
deferral fraction rather than forcing a label.

## Same-speed motion vs. crossing occlusion

`layer_drift.py` builds one `DriftEvent` per problematic anchor window with a
three-stage evidence chain (pre-crossing / crossing / post-crossing, or an
occlusion-scan stage). Each `StageEvidence` records the layer attribution
(track counts per layer), centroid separation, speed ratio, median geometry
residual per layer, active/interrupted/new track counts and the spatial
concentration of track terminations.

The cause is classified from observation thresholds only:

* **same-speed motion** — speed ratio > 0.75 and centroid separation < 0.20 of
  image width, with continuous, spatially spread tracks;
* **crossing occlusion** — no speed lock, but track terminations concentrate
  in one 4x4 image tile (`OCCLUSION_TILE_FRAC = 0.21`). Uniform KLT expiry
  spreads across all 16 tiles; measured occlusion spans reach 0.24–0.30 while
  same-speed spans stay below 0.20;
* **mixed** — both signatures present;
* **unsupported** — neither threshold reached; the diagnostic explicitly
  refuses to assert a cause.

Measured cause assignments:

| Scenario   | Same-speed events | Occlusion events | Notes                                    |
|------------|-------------------|------------------|------------------------------------------|
| `similar`  | 1 (window 24–32)  | 0                | the designed crossing, continuous tracks |
| `occlusion`| 0                 | 3                | pillar coverage loss, no speed lock      |
| `baseline` | 1 (clip-end pair) | 0                | unresolved: no post-window anchor pair   |
| `sparse`   | 1                 | 0                | dominant-layer ambiguity, deferred       |

The first anchor window is excluded from the occlusion scan because initial
feature redetection produces a co-located birth burst that mimics
concentration; a `new > 50% of active` guard rejects birth-dominated spans.

## Bounded log cost

The diagnostic log is linear in the number of *events*, not tracks:

* one aggregate line per event (`DriftEvent.as_log_line`),
* one line per stage in its (max three-stage) evidence chain,
* per-track detail only for residual alerts, capped at
  `MAX_DETAIL_PER_EVENT = 12`, with truncation reported via
  `DriftReport.bounded`; the totals still count every alert.

On the 482-track `similar` clip the full report is 18 lines.

## Where anomalies are localised

Each residual alert and attribution is attached to a named stage and window,
so a geometry-residual spike is traceable to the exact anchor pair that
produced it. The GUI surfaces this in two places:

* the **层身份漂移诊断** panel shows the bounded evidence chain;
* the region-motion panel overlays the active window's measured cause while
  the frame transport is inside that window. The overlay text comes from the
  drift report, never from a hard-coded default.

## Module boundaries

* `layer_tracker.py` decides identity (multi-hypothesis, reversible).
* `layer_drift.py` explains *why* drift occurred and *where* the evidence
  weakened; it never changes a label.
* `pipeline.py` runs the tracker, commits its decisions onto the
  decomposition label partition via `_apply_tracking`, then builds the drift
  report. Camera/box solves and the composite read the single committed
  partition exactly as before.
* `evaluation.identity_swaps` honours the correction log: a flip the tracker
  reconnected, or kept open as an unresolved tie, is not counted as an
  identity exchange because it never reaches the output partition as one.
