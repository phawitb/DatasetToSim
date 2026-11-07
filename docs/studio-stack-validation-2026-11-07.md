# Cube stacking validation — 2026-11-07

## Problem and supported workflow

The failed capture `df40666f38e54bfab07b0e8ed4ef66cf` selected two cubes, but the planner interpreted every To mask as a 118 mm bowl. The error was a destination-type restriction, not a failed SAM mask. The user confirmed both cubes are 19.5 × 19.5 × 19.5 mm.

New Run now offers **Place in bowl** and **Stack on cube** before Capture. Stack mode fits both SAM masks to the measured cube geometry. When the bowl is present, the default checked **Bowl present in workspace** requires a third **Bowl obstacle** selection. Task and obstacle presence are frozen in the capture; changing either starts a fresh session.

Both cubes are free dynamic bodies. The target is not pinned, and there are no post-initialization object state writes or artificial holding forces. Stack traces and comparison rendering include both objects. Success requires a secure grasp and lift, release without finger contact, upright stable cubes, the source center inside the target support, actual supporting contact force, and target displacement at most 4 mm. Fresh preflight checks both cube positions to 3 mm and their square-symmetric orientations to 10 degrees, as well as the bowl.

The planner predicts the held source center using contact physics, correcting placement waypoints within a 20 mm bound and at most five probes. It does not change scene geometry or physical parameters. The resulting immutable action sequence must then pass the full simulation, collision and tracking checks. The trajectory includes release, retreat and the calibrated return corridor to Home. Interrupted stacking cannot fall back to an older bowl-only Home scene; already-Home requests return without motion.

## Offline and automated checks

- Original two-cube capture reproduced the unsupported-bowl error before the change.
- A naive stack trajectory displaced the target by about 35 mm and correctly failed validation. The held cube center was offset from the nominal jaw target; the bounded planning probes address this without changing initial object poses.
- Diagnostic `output/stack-diagnostics/first/` passed full physics after four planning probes. This offline diagnostic explicitly omitted the bowl and was never dispatched to hardware.
- Full Python suite: **127 passed**, two existing dependency deprecation warnings. One initial restricted run could not invoke `ps` in a worker-cleanup test; rerunning with process inspection available passed the complete suite.
- Playback tests: **3 passed**; JavaScript syntax check passed.
- Tests cover both cube fits, explicit bowl presence, dynamic target physics and trace, stack scoring rejection cases, moved/rotated target preflight, immutable capture metadata, already-Home behavior, and rejecting bowl-only Home recovery after an interrupted stack.
- Independent read-only code review found no remaining critical or important issues after the interrupted-Home fix.

## Fresh physical run

Run **`739d5e1b983a43079151407ff3f50a79`** used a fresh live capture, two cubes and the bowl obstacle:

- SAM scores: From 89.4%, To 92.2%, bowl obstacle 99.0%.
- Plan: **1,189 commands / 39.63 seconds**, maximum IK residual approximately 0.3 mm.
- Simulation: success and transfer checks passed; no robot/workcell contacts. Target displacement 0.226 mm, final support force 0.04364 N, settled stack. Robot self-contact penetration was below 0.001 mm; finger/table peak penetration 0.030 mm and force 4.99 N remained within existing checks.
- Hardware: all **1,189 commands** completed, no execution error. Real Top/Wrist recordings show pickup, release onto the destination cube, withdrawal and automatic return Home. The upper cube has visible lateral overhang; the physical placement is not perfectly centered.
- The stack remained supported at the end of the run. No manual repositioning or Torque off was used. A subsequent Return Home click reported **Already Home** and issued no new motion.
- Paired comparison: 1,190 frames / 39.67 seconds, nearest preceding camera samples on the shared command clock, no motion retiming.
- UI checks: server restart, reconnect, Bowl/Stack task labels, all three SAM selections, plan, simulate, execute, Already Home, history labeling, Play/Pause and seeking to the final paired frame.
- `physical-review.json` records the visual assessment separately from simulation and command completion. Evidence: `stack-release-24.jpg`, `stack-release.jpg`, `stack-final.jpg` and `hardware-comparison.mp4` under the run folder.

This is one observed successful physical stack, not a general success rate. The model retains the existing estimated camera, friction and motor parameters. Stacking is limited to the confirmed cube dimensions; selecting arbitrary objects does not calibrate their geometry.
