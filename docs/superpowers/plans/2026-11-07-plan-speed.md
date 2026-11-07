# Adjustable plan speed

**Goal:** Select 50–150% travel speed in Plan the grasp, preserving all three motion techniques and validated sim-to-real timing.

**Design:** Default 100% retains existing behavior. Scale the five arm velocity caps and travel minimum durations; leave gripper speed, contact/settle minimum durations, and Flow/Gentle acceleration/jerk bounds unchanged. This is a travel setting, not guaranteed inverse total duration. Keep original default validator caps for legacy plans and Home. Explicit per-plan speed is validated in planner, simulation and hardware, bounded to 0.5–1.5. Timestamp hashes bind new plans to their simulated timing. Never modify a trajectory at execution.

**Approaches considered:** Uniform timestamp compression would increase acceleration/jerk and shorten contact waits; rejected. Raising acceleration and jerk together with speed could undermine smoothness; rejected. Replan within unchanged acceleration/jerk bounds and adjustable arm speed; selected as implementation judgment within user's authorized scope.

**Files:** Shared contracts in src/dataset_to_sim/studio.py; interpolation in scripts/studio_motion.py and studio_planner.py; server options and hardware validation; web selector and history; regression tests.

- [x] Add failing tests for invalid speed, bounds, unchanged 100%, faster smooth travel, holds and timing mismatch.
- [x] Implement bounded speed throughout plan/sim/execute and immutable timestamp verification.
- [x] Add English speed control, reset/restore/history display and stale-plan lock.
- [x] Run regression suite and same-scene simulation at baseline and higher speeds.
- [x] Review hardware dispatch changes, restart, test UI and fresh selected scene in sim before authorized hardware run.
- [x] Save real evidence and report exact outcomes, timings and remaining limits.

**Review focus:** Nonfinite/bool/out-of-range inputs; old plans without speed fields; changed speed after simulation; timestamp mismatch; unchanged acceleration/jerk and contact hold times. Simulation success is not real-world task success.
