---
name: dataset-to-mujoco
description: Use when converting real robot recordings or LeRobot datasets into MuJoCo simulation, replaying native actions, calibrating robot/camera/contact models, or comparing real and simulated episodes. Applies to แปลง dataset เป็น sim, real-to-sim, SO-101 replay, and preparing a simulator for later sim-to-real policy work.
---

# Dataset to MuJoCo

Produce auditable physical replay, reusable calibration and full real/sim comparison videos. This workflow does not promise automatic 100% task success or hardware transfer.

## Start from actual inputs

Inspect the requested dataset, recording code, robot assets, existing project and latest verified results. Establish channel order, action semantics, units, timestamps, episode identities, video slices and recording-time calibration. Names such as `state_deg` do not prove every channel is degrees.

For the existing SO-101 implementation, read [local-project.md](references/local-project.md). For new datasets, new robots or unresolved calibration, read [calibration-and-diagnostics.md](references/calibration-and-diagnostics.md). Reuse working components and explicitly adapt unsupported formats.

## Replay contract

Default to unchanged native actions when physical replay or future policy transfer is intended:

- Preserve raw actions and recorded interval boundaries. Use one documented, fixed, invertible unit/frame adapter. Initialize from the observed first state, including a nonzero gripper opening.
- After initialization, recorded states are calibration/evaluation targets, not actuator commands. Apply causal action holds; model motor response/latency as shared simulator parameters.
- Objects move through physics after initialization. Initial scene poses may vary according to recordings; do not tune them from outcomes to rescue individual episodes.
- Freeze shared robot, camera, contact and motor parameters across evaluated episodes. Strict results exclude per-episode trajectory repair, IK, clipping, grasp retiming, object teleporting and artificial holding forces.
- If explicitly asked for kinematic visualization or action adaptation, label that separate mode and do not count it as unchanged-action physical replay.

## Execute

1. Separate measured geometry from estimates. Check gripper zero, tabletop/base support and collision surfaces before fitting friction.
2. Run a short rendering/physics smoke test. Resolve graphics access separately from calibration. Preserve a reproducible baseline.
3. Fit shared parameters against state, image and contact evidence. Separate camera timing from action timing. Diagnose onset, slip, release and containment; inspect recovered and regressed cases.
4. Freeze a candidate and run every requested episode. Rebuild models for final evaluation. Independently verify native actions, converted targets, timestamps, final action intervals, source/data/model hashes and physical contacts—not just optimizer scores.
5. Render complete source-length comparisons from saved integrated states. Cover all episodes if requested; otherwise select representative successes/failures and state the covered IDs. Check frame counts, FPS/timestamps, orientation and actual images. Report per-channel state errors in their own units, image metrics, successes, regressions and unresolved failures.

For a new calibration, reserve whole evaluation episodes before fitting when feasible. If all episodes informed selection, call it calibration-set performance.

## Deliver

Save a frozen config, run manifest, per-episode results, integrity audit, model/trace archives and full comparison videos, with runnable reproduction commands. State changes and uncertainty; successful replay is not validated sim-to-real transfer. Continue authorized local work without repeated confirmations. Publishing data or commanding physical hardware is outside this skill's default scope.
