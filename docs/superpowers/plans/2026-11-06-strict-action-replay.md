# Strict native-action replay implementation plan

> Execution: inline, as explicitly requested. Preserve historical experiments; do not request another design approval.

**Goal:** Run recorded native SO-101 actions unchanged through a fixed, explicit unit/frame adapter and MuJoCo actuators, with measured cube/bowl dimensions.

**Architecture:** A standalone native-action adapter and replay integrator in `src/dataset_to_sim/strict_replay.py`; a script builds the measured-dimension scene and exports all-episode evidence. Reuse archived rendering and robot collision geometry, never the old trajectory-control functions.

**Spec:** `calibration/strict-action-contract.json` plus latest user measurements: wooden 19.5 mm cube, bowl outside top diameter 118 mm and height 50 mm.

## Constraints

- No IK, clearance target changes, action interpolation, retiming, or silent control clipping.
- Use measured state only for the initial state and evaluation; never feed subsequent observations to the controller.
- Native arm values are degrees; gripper is normalized 0–100. Convert using the current follower encoder calibration, saving its snapshot and hash. Historical calibration identity remains unverified.
- Use the model's documented lower gripper limit as a provisional closed reference; report that this mechanical zero is not measured. Keep inherited arm-frame offsets explicit estimates.
- Apply each recorded action with zero-order hold over its timestamp interval, including the last frame interval. Shorten the last integration substep to land exactly on an action boundary.
- Cube initialized once, then contacts only. Enable all existing physical collision categories and disable artificial gravity compensation in this new mode.
- Bowl bottom diameter/wall thickness and wooden cube density remain declared assumptions. Do not represent them as measurements.

## Tasks

- [x] Add failing tests for gripper encoder scale and inverse mapping, zero-order action timing, no state-following, no input mutation or hidden clipping, and physical dimensions.
- [x] Implement native unit adapter and strict integrator. Preserve complete command/step traces and report joint errors in native units.
- [x] Build cube and bowl geometry from the measured dimensions, including collision geometry; write a frozen configuration separate from archives.
- [x] Run all 72 episodes with the same model settings, report failures honestly, and render episode 0 plus a representative episode.
- [x] Verify action hashes, source timestamps, model geometry/collision settings, finite states, and regression tests. Document limitations and reproduction commands.

## Review focus

1. A late action must not affect any earlier simulation step.
2. Changing measured states after frame zero must not change the trajectory.
3. Out-of-range actuator targets must not be silently clipped; joint/torque limits are physical model constraints.
4. Gripper scale must come from encoder travel, not treating 0–100 as degrees.
5. The final recorded action must be applied even though no subsequent real observation exists.
