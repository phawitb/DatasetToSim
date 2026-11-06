# Strict native-action replay: 64/72 task passes

All 72 episodes completed with 0 runtime errors. Task pass rate: **88.9%**. **This is not a physically validated digital twin or a proven sim-to-real transfer result.**

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
| robot_self | 77.552N | 0.141mm |
| finger_table | 66.395N | 0.240mm |
| cube | 11.124N | 3.017mm |

## Task criteria

- Lift at least40mm above initial cube center.
- Continuous bilateral normal force of at least0.02N per jaw, with jaw/cube penetration at most1.5mm: at least80ms of secure grasp and30ms of secure lift.
- All8 cube corners inside the physical bowl, no final finger support, final linear speed≤0.03m/s and angular speed≤0.3rad/s, and less than3mm translation over the last10samples.
- Score is measured after executing the final source action for one nominal30Hz interval. Videos show the originalN observation frames; terminalN+1 physics state is stored in the trace.

## Results

| Outcome | Count |
|---|---:|
| success | 64 |
| not_lifted | 7 |
| missed_bowl | 1 |

Failed episodes: 37 (not_lifted), 44 (not_lifted), 50 (not_lifted), 51 (not_lifted), 53 (missed_bowl), 63 (not_lifted), 66 (not_lifted), 70 (not_lifted).

## Video and reproduction

- [Episode0: success](episode-00/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode37: not_lifted](episode-37/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode38: success](episode-38/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode66: not_lifted](episode-66/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.
- [Episode71: success](episode-71/real-vs-sim-strict.mp4) — full top/wrist real-versus-sim comparison.

Camera observation delays: {'top': 2, 'wrist': 2} frames, using only earlier integrated states. They are rounded image-fit estimates, not measured exposure timestamps. Actuator commands retain their original timing.

```sh
.venv/bin/python scripts/strict_action_replay.py --config output/measured-bowl-state-v7/settings.json --output output/reproduction
.venv/bin/python scripts/audit_strict_action.py --output output/reproduction
.venv/bin/python scripts/strict_action_replay.py --output output/reproduction --render-only --episodes 0 37 38 66 71
```

macOS rendering requires desktop graphics access. Playback uses the archived model selected by current metadata, with XML/trace hashes checked. Shared STL assets, source snapshots, original dataset and calibration hashes are retained.

## Remaining uncertainty

- Camera intrinsics/distortion/exposure timing, historical absolute robot reference calibration, finger compliance, motor response, contact friction and object mass are not fully measured. A successful replay is not evidence that a new policy is safe or reliable on the real robot.
- Multiple global parameter fits were evaluated. Training/other-episode diagnostics are not a blind generalization experiment: earlier outcomes and all initial poses were inspected during development.
- Fits that increased task passes through large initial overlap were rejected. Later fits can improve native state error while reducing task passes; those alternatives remain separate experimental artifacts.
- Commands were not sent to the physical robot. The native action/state conversion contract is explicit and invertible, but hardware policy transfer remains unvalidated.

## Refinement versus previous strict replay

| Metric | Previous | This run |
|---|---:|---:|
| Task passes | 59/72 | 64/72 |
| Pooled arm state RMSE, degrees | 0.758863 | 0.639177 |
| Gripper state RMSE, native0–100 | 3.144348 | 2.632006 |
| Stable held gripper RMSE, native0–100 | 4.171566 | 3.689830 |
| Top orange-mask IoU,60frames |0.514646|0.535127|
| Wrist orange-mask IoU,60frames |0.670753|0.747157|

Recovered original failures: [24, 28, 31, 36, 38, 40, 64, 71]. Regressed previous passes: [53, 63, 70]. Remaining failures: [37, 44, 50, 51, 53, 63, 66, 70].

The main shared profile is measured-bowl-task-v7 (65/72). The alternative measured-bowl-state-v7 (64/72) has lower native state error. Each profile uses one unchanged calibration for all72; no episode-wise cherry-picking or different action streams. Both use the new user-measured98mm outside bowl-bottom diameter.

The original13 failures have not all been fixed. The main profile recovers9, leaves4 unresolved and introduces3 regressions. The alternative recovers8 and introduces3 regressions. All reported rates are exact for these saved traces, not a claim of robust real-world performance.

Changes: stable blocked-gripper evidence replaces a faulty opening>25 filter; shared motor gains/damping/delay/inertia/friction are refitted; shared robot reference/baseXY and contact materials are identified; wrist camera uses a balanced episode sample and independent-frame checks. Gripper native zero and all measured object dimensions remain constrained. Calibration includes all72 state/terminal-image observations, so no blind evaluation claim is made.

Independent review found an offline motor-fit cache inconsistency. The helper now calls mj_setConst and mj_resetData after mutations; a red/green regression verifies in-place inertia changes match a fresh build exactly. Earlier optimizer trial losses are historical; authoritative outcome/state/image metrics above come from freshly rebuilt strict replays. The corrected motor fit is retained in the lower-state-error alternative.

The terminal-image calibration target is an approximate red-pixel centroid; cube occlusion and unknown lens distortion remain uncertainty. Rejected detections9/53 are not used. No terminal pose enters replay control.

Detailed remaining-case contact/gap snapshots are in failure-diagnostics.json. They describe simulator evidence at real stable-hold frames and do not prove a unique physical cause.

Verification:50 regression tests passed; all72 original actions/state0, timestamps, input hashes, settings consistency, secure contact durations, terminal containment and98mm bowl-bottom collision geometry were checked. Video-verification.json confirms original full frame counts and30fps.
