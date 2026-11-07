# Aperture-aware grasp correction — 2026-11-07

## Failure and cause

In failed physical run `3c0bbb7e4b904b6e8f369c150536e39e`, the fixed jaw descended onto the cube. The user confirmed the jaw was obstructed by the cube. The measured arm pose consequently stopped above the simulated pose; interpreting that difference as a request for a lower target would have been incorrect. The subsequent user-run `5b3f23abe2974368b9e093c24bded338` used essentially the same cube location and old planner.

The planner always computed the pad midpoint at a nominal occupied aperture of 0.15 rad, even when the commanded gripper opening was 35 native units. This robot has one fixed and one moving jaw. At the old open approach waypoint, the actual open-mouth midpoint was displaced **14.56 mm in world X** from the cube. Nominal geometry left **2.24 mm** between the fixed jaw and cube, but **32.44 mm** on the moving side. Small real alignment/tracking differences could therefore put the fixed jaw onto the cube. Nominal simulation passed because its jaw narrowly cleared the cube.

Recorded failed-run grip readings closed toward 0.8–1.8 native units, rather than retaining a cube. A previously successful run retained an opening around 8.5. The discrepancy is observable in the saved command/encoder trace and wrist video; command completion alone did not establish success.

## Correction

`studio_planner.py` now evaluates the jaw midpoint at the commanded opening whenever it is wider than the nominal occupied aperture. Approach and descent center the open mouth over the cube. The closing waypoint returns to the nominal occupied aperture, with bounded, smooth arm/gripper interpolation. The nominal occupied aperture is retained for closed holding, rather than solving IK at an impossible empty closure through the cube.

The grasp target height remains 24 mm. Robot/camera calibration, collision geometry, friction, dynamics, force limits, collision checks and execution gates were unchanged. Archived dataset replay and past runs were not modified.

The saved failed scene was replanned into `output/grasp-diagnostics/aperture-centered/`. The corrected nominal approach has **16.74 mm fixed-jaw clearance and 17.88 mm moving-jaw clearance**. Contact simulation passed. A regression test reproduces the original 14.56 mm midpoint error before the fix and checks centered approach plus minimum clearance on both jaws afterward.

## Fresh physical validation

Run **`a4f89f79aa54423f9e784536e55f4fe6`**:

- Return Home, fresh Top/Wrist capture, SAM From 93.7% and To 96.9%.
- New plan: **663 commands / 22.10 seconds**, maximum IK error 0.383 mm.
- Simulation: success and transfer checks passed; zero robot-self and robot-workcell contacts. Finger/table maximum penetration 0.034 mm, maximum normal force 3.66 N, within the existing validation rules.
- Hardware: all 663 commands completed, no execution or recording error.
- Real grip plateau **8.60 native units** from closing through lift/transfer, rather than empty closure.
- Fresh final Wrist image visibly shows the cube released inside the bowl and an empty gripper. Top image shows the source location empty. Recorded approach/grasp/lift frames were also inspected.
- `physical-review.json` records **success**; Run history displays **Placement verified**.
- Paired real/sim video: 664 frames / 22.133 seconds, nearest preceding camera sample on the shared command clock, no motion retiming.

Evidence is under `output/studio/a4f89f79aa54423f9e784536e55f4fe6/`, including `real-final-wrist.jpg`, `real-final-top.jpg`, `grasp-review.jpg`, command traces and `hardware-comparison.mp4`.

The subsequent Wrist-verified Home run `8d108d6e61044624a108d7e394937f91` completed all **391 commands** with no execution/recording error. `after-home-top.jpg` shows the cube remaining in the bowl and the arm at its lowered rest pose. No manual repositioning or Torque off was used.

Python suite: **119 passed**, two existing dependency deprecation warnings. This is one successful physical retest, not a measured general success rate across objects or placements.
