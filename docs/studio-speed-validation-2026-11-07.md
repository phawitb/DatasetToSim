# Adjustable travel speed — 2026-11-07

The Plan the grasp panel provides a 50–150% travel speed control, default 100%, independently of recorded-video playback speed. Classic, Gentle and Flow remain selectable. The five arm caps are scaled; the gripper is unchanged. At 150%, the validation envelope is 52.5°/s per arm joint, with Flow/Gentle planning below 48.3°/s and Classic below 42°/s. The base 35°/s and gripper 50 normalized units/s constants remain unchanged for legacy plans and Home.

Only travel minimum durations are shortened. Approach, grasp, placement, release and settling minimum durations remain intact. Flow/Gentle acceleration limits remain 80°/s² for the arm and 120 normalized units/s² for the gripper; jerk limits remain 600°/s³ and 900 normalized units/s³. These are command trajectory bounds, not measured physical motor guarantees. Consequently, 150% does not mean a 1.5× whole-task speedup.

Every plan saves its speed and timestamp hash. Simulation verifies generated timing and saves its own hash and speed; the hardware worker independently verifies all three plus action hashes before dispatch. Changing the UI value disables stale-plan simulation/execution. No runtime timestamp compression or collision/tracking threshold relaxation is used.

## Same-scene offline physics

| Scene / Flow | 100% | 125% | 150% |
|---|---:|---:|---:|
| Bowl placement | 17.67 s | 16.30 s | 15.63 s |
| Cube stack and Home | 30.30 s | 28.17 s | 27.00 s |

All six passed the full existing physical task verdict and transfer checks. Robot-self and robot-workcell contacts were zero. Maximum arm tracking error at 150% was 7.56° for the bowl and 7.62° for stacking, within the unchanged 10° transfer bound. Raw reports, action arrays and physical traces are under output/speed-diagnostics. Offline comparisons suppress video rendering and do not count as hardware runs. Source captures: a4f89f79aa54423f9e784536e55f4fe6 and 6257815b1c6f4fa380f67ded78a48ee9.

## Verification

The full Python suite initially returned 147 passes and one environment failure: the process-cleanup test was unable to invoke ps under restricted execution. Rerunning that test with process inspection allowed passed. Three additional hardware dispatch regressions passed, proving changed timing/speed dispatches zero targets. Three playback tests and JavaScript syntax validation passed. Independent review found no critical or important issue.

UI verified the slider's default and 150% cap, invalidation at 145% after planning at 150%, and recovery when restoring the matching 150% value before simulation. Real-run results are recorded below after completion.

## Fresh-scene physical execution

Run `2e6fbf4f58ae44eca1fa30c9b35f9af8` selected both cubes and the intervening bowl using SAM. Flow at 150% generated 812 commands / 27.07 s, passed full physical stacking and transfer checks, then completed all 812 hardware commands without error. Recorded Top/Wrist views show the source cube on the destination, empty withdrawing jaws and the stack still standing after automatic Home. Physical evidence is stored in speed-release.jpg, speed-withdraw.jpg, speed-final.jpg and physical-review.json alongside the run.

The same new capture replanned and simulated at Flow 100% passed in 30.40 s. The faster plan saves 3.33 s (11.0%) against that same-scene plan; only the 150% plan was executed physically in this comparison. Peak measured error against the prior commanded target was 8.745° across arm joints, within the unchanged 10° runtime stop threshold. Final Home error was below 3 native units. Paired replay is 813 frames / 27.1 s, with 0.010 s unpaired recording tail and no timing warp. This is one successful physical run, not a general reliability estimate.
