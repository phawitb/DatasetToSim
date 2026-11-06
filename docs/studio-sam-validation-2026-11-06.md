# SAM selection and real robot validation — 2026-11-06

## Implemented workflow

Live capture → click From cube → click To bowl → SAM masks → calibrated metric reconstruction → bounded IK → contact simulation → fresh SAM preflight → exact simulated commands on SO-101. The live path no longer uses HSV segmentation. Original recorded benchmark estimates are unchanged.

SAM 2.1 base runs locally through Ultralytics 8.4.80 on MPS in the existing LeRobot Python environment. Model SHA-256: `f1a9cf2dd69d84bb463b5ad98246d03e2d47a130a9295db0ec967e6cd95e2e47`. Each capture retains image hash, model hash, clicked points, masks, confidence, and selection revision. The selected connected component must contain its prompt; overlapping From/To masks are rejected. Selecting another object invalidates plan and simulation. Arbitrary object dimensions are not inferred: transfer remains limited to the calibrated 19.5 mm cube and 118 mm bowl.

The first attempted dispatch (`9ab695817d824bb299fb38ea0de84c9e`) was blocked before any motion by an initially too-strict two-second vision deadline. Measured SAM encoding of a new image took 2.620 seconds, versus 0.023 seconds for the second prompt on its cached features. The bounded deadline was set to five seconds, enforced after preflight on the server and again inside the hardware lock immediately before starting motion. Actual camera frame timestamps, not response timestamps, determine age. No guard was disabled. Slow inference, Stop races, changed masks, stale revisions, moved objects, pose mismatch, changed calibration, and consumed plans continue to block execution.

## Browser and hardware result

Final run: `4def04d354554195a264d4eb08ea49c3`.

- Actual UI actions: Connect, Live workcell, Capture, click From, click To, Generate plan, Run simulation, Execute.
- From confidence: 89.885%; To confidence: 98.662%.
- Mask-to-cube boundary fitting error: 0.703 pixels.
- Simulation: success; transfer checks passed; 600 commands / 20.000 seconds.
- Hardware: 600 commands completed; final target at 19.969 seconds; no motion or recording error.
- Visual task assessment: final recorded top and wrist images show the cube inside the bowl after release. This is one observed successful run, not a general hardware success rate.
- Comparison recording: 601 frames / 20.033 seconds, using the same command clock. Camera frames are aligned causally; no time warping.
- Browser Play advanced from 0.106 seconds to 11.516 seconds; Pause held there; Play resumed from 11.864 seconds. Real and Sim use crops of the same 1280×1024 recording. Run permalink restores the recording after reload.

Local artifacts:

- `output/studio/sam-selection-ui.jpg` — SAM overlays and sim scene on the website.
- `output/studio/4def04d354554195a264d4eb08ea49c3/real-final.jpg` — final real top + wrist images.
- `output/studio/4def04d354554195a264d4eb08ea49c3/physical-review.json` — explicit visual assessment.
- `output/studio/4def04d354554195a264d4eb08ea49c3/hardware-comparison.mp4` — synchronized replay.
- The same folder holds capture, selection, preflight SAM, plan, native commands and encoder traces.

## Home check and current limitation

Home was also clicked after transfer (`71107d16858a4015b4d957b8882a0742`). It correctly did not seek the cube at its old From pixel. The arm obscured too much of the bowl for fresh SAM verification, so Home failed closed without moving. The arm remains holding its final pose with torque enabled. Clearing that specific occlusion currently requires supporting the arm and using Torque off for manual repositioning, then Torque on and a new capture. This run does not establish that automatic Home works under full bowl occlusion.

Home checks only fresh bowl masks against a recent SAM reference (ten-minute maximum). It tries other interior prompts when the original bowl click is occluded, requires mask overlap/area and center agreement, and validates the fixed Home corridor in MuJoCo. The old cube remains simulation geometry only; cube contact is not part of the Home collision gate.

## Automated checks

96 Python tests and 3 Node playback tests passed. Added regressions cover prompt bounds, disconnected masks, overlapping selections, image/mask tampering, plan invalidation, missing selection, post-transfer Home prompts, hidden bowl rejection, expired SAM scenes at the hardware boundary, and Stop races. JavaScript syntax and `git diff --check` passed. Automated tests use fake hardware; the physical result above was separately observed on the connected robot.
