# Strict native-action replay: 59/72 task passes

All 72 episodes completed with 0 runtime errors. Task pass rate: **81.9%**. **This is not a physically validated digital twin or a proven sim-to-real transfer result.**

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
- Cube mass4.449g assumes wood density600kg/m³. Bowl bottom outer diameter82.6mm, wall3mm, floor4mm, friction and motor properties remain estimates.

| Contact category | Maximum normal force | Maximum penetration, including transients |
|---|---:|---:|
| robot_self | 18.738N | 0.198mm |
| finger_table | 35.827N | 0.220mm |
| cube | 23.177N | 4.196mm |

## Task criteria

- Lift at least40mm above initial cube center.
- Continuous bilateral normal force of at least0.02N per jaw, with jaw/cube penetration at most1.5mm: at least80ms of secure grasp and30ms of secure lift.
- All8 cube corners inside the physical bowl, no final finger support, final linear speed≤0.03m/s and angular speed≤0.3rad/s, and less than3mm translation over the last10samples.
- Score is measured after executing the final source action for one nominal30Hz interval. Videos show the originalN observation frames; terminalN+1 physics state is stored in the trace.

## Results

| Outcome | Count |
|---|---:|
| success | 59 |
| not_lifted | 12 |
| no_bilateral_grasp | 1 |

Failed episodes: 24 (not_lifted), 28 (not_lifted), 31 (not_lifted), 36 (not_lifted), 37 (not_lifted), 38 (not_lifted), 40 (not_lifted), 44 (no_bilateral_grasp), 50 (not_lifted), 51 (not_lifted), 64 (not_lifted), 66 (not_lifted), 71 (not_lifted).

## Video and reproduction

- [Episode0: success](episode-00/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode45: success](episode-45/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode50: not_lifted](episode-50/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.

Camera observation delays: {'top': 2, 'wrist': 1} frames, using only earlier integrated states. They are rounded image-fit estimates, not measured exposure timestamps. Actuator commands retain their original timing.

```sh
.venv/bin/python scripts/strict_action_replay.py --config output/touch-zero-v1/settings.json --output output/reproduction
.venv/bin/python scripts/audit_strict_action.py --output output/reproduction
.venv/bin/python scripts/strict_action_replay.py --output output/reproduction --render-only --episodes 0 4 50
```

macOS rendering requires desktop graphics access. Playback uses the archived model selected by current metadata, with XML/trace hashes checked. Shared STL assets, source snapshots, original dataset and calibration hashes are retained.

## Remaining uncertainty

- Camera intrinsics/distortion/exposure timing, historical absolute robot reference calibration, finger compliance, motor response, contact friction and object mass are not fully measured. A successful replay is not evidence that a new policy is safe or reliable on the real robot.
- Multiple global parameter fits were evaluated. Training/other-episode diagnostics are not a blind generalization experiment: earlier outcomes and all initial poses were inspected during development.
- Fits that increased task passes through large initial overlap were rejected. Later fits can improve native state error while reducing task passes; those alternatives remain separate experimental artifacts.
- Commands were not sent to the physical robot. The native action/state conversion contract is explicit and invertible, but hardware policy transfer remains unvalidated.

## Latest gripper clarification

The screenshot matches episode45/frame0: native state0.8124576807 and action0.2423263341. All72 initial states are0.677048–1.015572; none are zero. The simulator preserves each measured initial opening, then executes unchanged commands. The source-geometry zero is−11.433150029degrees: distal surfaces just touch at native0 and are1.363633mm apart at native0.812457681. These gaps are model predictions, not real measurements. The prior wrist camera fit used a different zero, so its historical0.801 mask IoU is not a measured validation score for this run. The bright top-face yaw confidence is an edge-consistency heuristic and remains sensitive to lighting/occlusion.
