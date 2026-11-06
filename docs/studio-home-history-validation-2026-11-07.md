# Home and run history validation — 2026-11-07

## Outcome

Run history is implemented and browser-verified. The physical Home request is **not complete**: three UI attempts were blocked before dispatch; the arm has not returned Home in this session. Torque was never disabled. The final controller remains connected, stationary and holding torque.

## Changes

- Current Wrist SAM verifies the calibrated bowl when the Top image is occluded. A bounded silhouette fit uses the current encoder-derived Wrist camera and rejects excessive displacement, poor overlap and poor boundary fit. Both preparation and immediate preflight repeat this evidence.
- Recovery associates with the latest completed immutable placement, with unchanged calibration, an open gripper and bounded differences from both measured and commanded release poses. It does not search older placements for a convenient match.
- Every snapshot checks encoder stability across one second, including the final read, and requires camera frames from that interval. Existing execution, timing, ownership, Stop and frame-age gates remain active.
- A Home-only solid cylinder conservatively covers the entire bowl plus 20 mm. Original bowl walls/floor remain intact. It is excluded from strict benchmark models.
- Repeated Stop writes a measured hold only once until another command is sent. Connection no longer calls the VLA speed-reset routine when position mode and uncapped speed are already correct; that routine rewrites measured goals and was accumulating gravity sag across reconnects.
- Read-only run history includes every saved capture directory, errors, simulations, Home and real recordings. Legacy `task_outcome: not_transferred` is shown as physical failure. Selecting history cannot activate a capture or authorize execution. Async image/video loads are tied to the selected run.

## Physical and simulation evidence

The completed placement used as reference is `4def04d354554195a264d4eb08ea49c3`; its last measured shoulder lift was 24.044°, while the final commanded value was 21.640°. At the start of this session the current shoulder lift was 26.066°. Earlier reconnect/hold behavior accumulated further settling to 31.868°. The latest pose is outside the 10° command association bound (10.229° difference).

At the earlier 26.066° shoulder pose, offline Home validation `8d9ed33f25f94042bde14c5af6405e48` passed: 391 commands, 13.03 seconds, maximum robot/workcell penetration 0.094 mm in the conservative model. This is simulation evidence only.

At the later measured pose, current Wrist evidence in `8764d52a357f4bdeb12df537ac350253` fitted the bowl with 1.15 px boundary RMSE, 0.938 mask IoU and an 18.0 mm reference shift. The actual camera image visibly shows the cube in the bowl. A separate offline diagnostic (`7825893e355342119f03110f1bde5b02`) found the gripper starts 27.5 mm inside the conservative bowl volume, with a 20.18° maximum simulated tracking error under its contact force. The unchanged physical bowl model has no initial contact; the 27.5 mm value is **not a measured real collision**. It means this conservative model cannot establish a safe return from the present pose. No contact threshold or clearance volume was reduced to force acceptance.

The final connection check, after fixing goal restaging, measured **zero change on all six encoders** before versus after reconnect. Evidence is in `output/studio/home-investigation/before-final-reconnect.json` and `after-final-reconnect.json`. This confirms the reconnect fix on hardware, not a successful physical Home.

The remaining work is a separately validated current-state retreat from the present near-bowl pose. It must establish clearance for the fingers and current bowl contents, retract with the gripper open, then fold Home. The existing close-first corridor is not established safe for this pose. Merely widening the pose bound or disabling the solid guard is insufficient.

## Verification

- Python suite: 107 passed, two existing dependency warnings.
- Node playback suite: 3 passed; JavaScript syntax and Git whitespace checks passed.
- Browser: history menu/dropdown, successful and failed physical recordings, completed playback from 0 to 20.03 seconds, saved-run deep links and reload were checked. History selection did not move the robot.
- Generated images, failed-attempt metadata and offline diagnostics are retained under `output/studio/`; no previous run evidence was overwritten.
