# Generate data: simulation demonstrations for LeRobot

The user authorizes design judgment and implementation, including all five tasks. Work continues on main; v1 remains the previous snapshot. This feature generates data locally and never commands hardware or uploads data.

Tasks: green cube to green bowl, red cube to green bowl, red cube to green cube, green cube to red cube, unstack. Unstack selects the top color and moves it to an empty table zone while preserving the bottom cube. Color is task semantics: never randomize red into green. Both cubes appear as context so the task cannot be solved from the only visible cube.

Approach: reuse calibrated SO-101 MuJoCo geometry and Natural guarded-grasp planning. Randomize bounded initial layouts, render physics-integrated observations, reject failed/colliding/unstable episodes, export only accepted episodes. Each observation/state precedes its labeled absolute native action. N actions pair with the first N of N+1 states. Store terminal verdict separately.

Techniques can combine: calibrated rendering baseline; episode-coherent visual/material/light diversity; bounded camera extrinsics; sensor brightness/contrast/gamma/noise/blur, with no spatial image warp or red/green hue swaps; optional bounded mass/friction variation requiring renewed physics. Save RNG seeds, calibration/model hashes, task labels, native units and group IDs. Whole-episode grouped train/validation splits prevent variant leakage. Default training guidance is real+sim fine-tuning with held-out new real sessions; sim success is not transfer validation.

A learned real/sim image bridge is a research option, not a prerequisite. Existing 72 episodes cover only red-to-bowl and are approximately aligned, with camera delay and state drift. Do not train pixelwise translation blindly or insert an unvalidated translator into hardware inference. Document ReBot-style compositing, constrained translation and feature alignment with gating requirements; UI must distinguish implemented generation controls from research-stage methods.

UI: fourth navigation item Generate data; independent setup/results page hides dataset episodes and live capture pipeline. Task cards, name/count/seed, motion/speed, composable technique controls, validation split, estimated count, start/cancel/progress, accepted/rejected breakdown, paired Top/Wrist video preview, export download and run history. Source data readiness and research notes explain actual limitations, including stale original metadata bounds. Original dataset is never modified.

Backend: one cancellable subprocess per batch; persisted status and diagnostics; no arbitrary path/model loading; bounded requests; block generation while live robot operates and block live dispatch while generator is active. Restart/exit terminates generator children. Rebuild scene before verification. Actual LeRobot v3 export audited with installed official reader, including temporal action windows and video decode. No silent partial batch reported complete.

Acceptance: tests for options, task semantics, physical verdicts, timing/alignment, leakage, lifecycle and export. Generate at least one accepted physics episode per task; inspect rendered colors/objects. UI start/preview/cancel/download smoke. No VLA policy training or real deployment success claimed without training and new held-out physical tests.
