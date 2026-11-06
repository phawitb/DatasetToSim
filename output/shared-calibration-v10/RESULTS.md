# Strict native-action replay: 68/72 task passes

All 72 episodes completed with 0 runtime errors. Task pass rate: **94.4%**. **This is not a physically validated digital twin or a proven sim-to-real transfer result.**

## What was replayed

- All **11,661 original six-channel actions** and **396,474 physics steps** verified. One frozen settings hash is shared across episodes.
- Arm channels are degrees. Gripper is normalized0–100, with the encoder-derived scale **1.298461538degrees/native unit**. No input clipping, trajectory repair, IK, grasp retiming, state-following after initialization, weld, object teleportation or applied holding force.
- Fixed model gripper closed reference: **-11.433150degrees**. This mechanical reference remains an identified estimate, separate from the known encoder scale.
- Motors use identified position response and causal delay history. The pre-recording history is initialized to the observed initial pose because older commands are unavailable.
- Cube poses are initialized from the first real top image using the measured-size3D silhouette. Later cube motion comes only from MuJoCo. Initial scene positions vary with the real scene; physical calibration does not vary by episode.

## Geometry and contact

- User measurements: cube19.5×19.5×19.5mm; bowl outer rim diameter118mm, height50mm; mat405×405×2mm; standard SO-101 fingers, base resting directly on tabletop.
- Model base body origin is3.4mm above world zero: table top1mm plus2.4mm from the source mesh underside to its body frame. There is no added riser or base tilt.
- Source arm/finger/workcell collisions are active. Distal fixed/moving jaw pairs explicitly override MuJoCo parent filtering. Normal same-body and connected-hinge filtering remains. Added pad proxies are disabled.
- Initial maximum overlap: **0.000000mm**.
- Bowl bottom outer diameter: 98mm (user measured). Cube mass4.449g assumes wood density600kg/m³. Wall3mm, floor4mm, friction and motor properties remain estimates.

| Contact category | Maximum normal force | Maximum penetration, including transients |
|---|---:|---:|
| robot_self | 46.488N | 0.066mm |
| finger_table | 73.884N | 0.122mm |
| cube | 8.985N | 2.382mm |

## Task criteria

- Lift at least40mm above initial cube center.
- Continuous bilateral normal force of at least0.02N per jaw, with jaw/cube penetration at most1.5mm: at least80ms of secure grasp and30ms of secure lift.
- All8 cube corners inside the physical bowl, no final finger support, final linear speed≤0.03m/s and angular speed≤0.3rad/s, and less than3mm translation over the last10samples.
- Score is measured after executing the final source action for one nominal30Hz interval. Videos show the originalN observation frames; terminalN+1 physics state is stored in the trace.

## Results

| Outcome | Count |
|---|---:|
| success | 68 |
| not_lifted | 4 |

Failed episodes: 37 (not_lifted), 44 (not_lifted), 50 (not_lifted), 51 (not_lifted).

## Video and reproduction

- [Episode0: success](episode-00/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode37: not_lifted](episode-37/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode44: not_lifted](episode-44/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode50: not_lifted](episode-50/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode51: not_lifted](episode-51/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode53: success](episode-53/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode63: success](episode-63/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode70: success](episode-70/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.

Camera observation delays: {'top': 2, 'wrist': 2} frames, using only earlier integrated states. They are rounded image-fit estimates, not measured exposure timestamps. Actuator commands retain their original timing.

```sh
.venv/bin/python scripts/strict_action_replay.py --config output/shared-calibration-v10/settings.json --output output/reproduction
.venv/bin/python scripts/audit_strict_action.py --output output/reproduction
.venv/bin/python scripts/strict_action_replay.py --output output/reproduction --render-only --episodes 0 37 44 50 51 53 63 70
```

macOS rendering requires desktop graphics access. Playback uses the archived model selected by current metadata, with XML/trace hashes checked. Shared STL assets, source snapshots, original dataset and calibration hashes are retained.

## Remaining uncertainty

- Camera intrinsics/distortion/exposure timing, historical absolute robot reference calibration, finger compliance, motor response, contact friction and object mass are not fully measured. A successful replay is not evidence that a new policy is safe or reliable on the real robot.
- Multiple global parameter fits were evaluated. Training/other-episode diagnostics are not a blind generalization experiment: earlier outcomes and all initial poses were inspected during development.
- Fits that increased task passes through large initial overlap were rejected. Later fits can improve native state error while reducing task passes; those alternatives remain separate experimental artifacts.
- Commands were not sent to the physical robot. The native action/state conversion contract is explicit and invertible, but hardware policy transfer remains unvalidated.

## Change from the previous shared calibration

Compared with measured-bowl-task-v7, this one shared profile increases task passes65→68/72, recovering episodes53,63,70 with no regressions. Episodes37,44,50,51 remain not_lifted; the requested seven-case repair is incomplete.

The only accepted changes are shared robot joint references/baseXY and contact parameters. Jaw sliding friction changes1.0→1.4, mat/table sliding friction0.6→0.3 and contact time constant6→4ms. The exact changes are in calibration-changes.json. Original actions/times, motor-response constants, cameras, encoder scales, gripper closed reference, supported base height, measured object sizes and collision categories remain unchanged.

| Metric | v7 baseline | Current v10 |
|---|---:|---:|
| Task successes |65/72|68/72|
| Frame-weighted arm RMSE |0.694971degrees|0.697486degrees|
| Gripper RMSE (native0–100) |3.058109|3.020942|
| Stable held-gripper RMSE |3.687551|3.548029|
| Top orange-mask IoU (same60frames) |0.534641|0.541318|
| Wrist orange-mask IoU (same60frames) |0.754880|0.753504|

Arm error and wrist silhouette overlap are slightly worse, while gripper error and top overlap improve. The camera itself was not changed. These small image changes arise from the integrated poses. Do not interpret IoU as physical accuracy.

A wider reference search, contact-gap fit, motor-response alternatives and a mat-edge camera fit were tested. The mat-camera candidate recovers51 but yields59/72 overall; its subsequent shared-reference search did not beat the selected profile. Image-only wrist geometry creates more than2mm initial self/table overlap when independently rebuilt and was rejected. No per-episode rescue settings, enlarged cube, extended fingers, disabled collisions or changed success thresholds were adopted.

The independent read-only review found noP1/P2 correctness issues and reran the action/timing/model audit. All8 videos have the full source frame count at30fps, covering episode0 and all7 previously failing episodes. The regression suite passed50tests. Full comparisons and four remaining failure snapshots are retained.

These are calibration-set results selected using the same72episodes. Contact coefficients and stiffness are unmeasured estimates; the reported contact-force maxima are simulated quantities, not measured hardware forces. This result does not demonstrate reliable policy transfer to hardware or resolve the remaining four failures.
