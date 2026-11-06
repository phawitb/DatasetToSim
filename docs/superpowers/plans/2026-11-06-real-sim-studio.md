# Real / Sim Studio implementation plan

Goal: A local robot workbench for comparing recorded real/sim data and capturing, planning, physically simulating and executing a new pick/place.

Architecture: FastAPI serves a buildless browser UI and local dataset media. Existing calibrated MuJoCo builders and native joint adapter remain the source of truth. A separate LeRobot runtime owns serial/cameras and provides cancellable execution. No remote hosting: cameras, local data and serial ports are required.

Design decisions delegated by user: light lab-workbench interface, two synchronized viewports, episode browser, native joint plots, and a capture/plan/simulate/execute panel. Current supported scene is the measured red 19.5 mm cube and green bowl with fixed top camera. New planned trajectories are explicitly distinct from strict dataset replay. Planning from recorded data is a rehearsal and cannot be executed on hardware.

Constraints: Preserve existing calibration and reports. Use actual source data, actual MuJoCo renderings and physics. Do not claim sim success proves real transfer. Run only server-owned simulated actions, matching current start pose, current scene, calibration hash and joint limits. Stop on disconnect, stale heartbeat, tracking error or stop request; hold measured pose with torque retained. Local-only listeners and same-origin write protection.

Review focus: stale captures/plans; simultaneous runs; invalid native units; unreachable or colliding plans; disconnected hardware/cameras.

- [x] Task 1: Tests for execution gate, timing/limits, media containment and recorded rehearsal restrictions; implement catalog and control contracts.
- [x] Task 2: Implement initial-image scene reconstruction, bounded IK waypoints, contact simulation and rendered trace. Verify a recorded-scene rehearsal end to end.
- [x] Task 3: Implement isolated hardware worker using existing VLA configuration and LeRobot runtime, capture freshness, execution/stop/heartbeat. Verify without physical movement.
- [x] Task 4: Build responsive workbench, timeline, episode list, joint chart and pipeline; start local server and browser QA.
- [x] Task 5: Full tests, independent review, README and runnable launch command. Document any hardware verification limit.

Review: independent reviewer identified stop/preflight race, camera loss during execution, cube-rotation freshness and per-page heartbeat ownership. All four have regression tests and fixes. Failed simulation/planning retries also invalidate earlier approval.

Evidence: recorded episode 0 rehearsal uses 583 new planned actions, succeeds in physical lift/grasp/release/containment and passes transfer checks. This is a recorded-scene rehearsal, not a hardware success claim.

Live validation: connected SO-101 using existing VLA calibration and camera configuration (one transient serial read failure, next attempt succeeded), captured both cameras and six encoders. Live capture 17911c547311473aa784bc151a5aa2b0 produced 627 actions / 20.9 s, IK max residual 0.4 mm, success in MuJoCo, transfer checks passed. No physical grasp was executed. Browser verified playback, camera toggles, narrow-screen episode selection, restored capture after reload, and desktop width without horizontal overflow.
