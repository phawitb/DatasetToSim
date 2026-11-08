# New Run recorded timing and real robot validation

New Run now offers Recorded robot pace (default) and Conservative studio pace in Plan the grasp. The same profile, native commands and timestamps flow through planning, contact simulation and hardware execution. Changing the profile or speed invalidates the prior simulation. Legacy plans default to conservative timing; Home retains conservative timing.

The shared recorded envelope remains derived from 72 SIMCUBE recordings, with per-joint arm ceilings of 100/90/75/100/100 deg/s before the interpolator's margin. Planning stores a timing contract hash. Simulation and hardware independently resolve the fixed local profile and reject modified limits, changed reference bytes, mismatched speed, or a missing/mismatched simulated timing contract. Browser-supplied arbitrary limits are not accepted. Existing freshness, segmentation revision, scene, calibration, action/timestamp hash, native limits, tracking, camera, deadline, heartbeat and stop-generation gates remain active.

This extends the synthetic-only scope described in the earlier generation timing note. Hardware now accepts an explicitly selected, simulated recorded profile through this additional contract; it does not infer that profile from generation options.

## Physical trial

Run: `d584f975fc054acc858113e28907f73d`
Task: Put the red cube in the green bowl.
Settings: Natural / Recorded robot pace / 100%.

- Fresh top/wrist capture; red cube and green bowl selected with SAM (93.2% and 98.3% scores).
- Plan: 269 actions at 30 Hz, 8.9667 s, maximum IK error 0.382 mm.
- Contact simulation: success, full bowl containment, transfer checks passed. No robot-self or robot-workcell collisions; maximum finger/table penetration 0.072 mm. Maximum arm tracking error 9.359 degrees.
- Hardware: all 269 targets exactly equal the simulated action array. Last command at 8.9448 s; completed with no error. Maximum observed previous-command arm deviation 9.231 degrees (threshold remains 10).
- Recording: 270 synchronized comparison frames / 9.0 s. Final Wrist frame visibly shows the released red cube resting inside the green bowl. Physical review saved with evidence; one successful physical trial, not a general success-rate claim.
- Review artifacts under `output/studio/d584f975fc054acc858113e28907f73d/`: `physical-review.json`, `hardware-trace.json`, `hardware-comparison.mp4`, `real-evidence-269.jpg`, `new-run-recorded-pace-success.png`.

Verification: 214 Python tests and 17 JavaScript tests passed. Browser tested Capture → SAM From/To → Plan → Simulate → Execute → final Wrist comparison. Robot remains torque-enabled at the plan's released pose above the bowl.
