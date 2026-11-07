# Natural coordinated curves — research and validation

## What caused the staged motion

Flow improved timing and limited jerk, but still passed exact Unfold, Raise, Lift, Transfer and return guides. It therefore retained the geometric up-then-turn route. Natural changes that route: travel guides shape a quintic B-spline without being mandatory stops or exact pass-through poses. Reach, carry and return are coordinated macro motions. Approach, grasp, placement/settle, release and Home task endpoints remain exact. Existing Flow, Gentle and Classic remain selectable and unchanged.

## Sources actually consulted

- [Modern Robotics chapter 9, Lynch and Park](https://modernrobotics.northwestern.edu/chapters/chapter9/) and [Cambridge chapter summary](https://www.cambridge.org/core/books/modern-robotics/trajectory-generation/9C12C16594E8507C7BD983E71138F6CB): geometric path and time scaling solve different problems. Read the public overview and summary, not the full paywalled chapter.
- [Modern Robotics 9.3 transcript](https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-3-polynomial-via-point-trajectories/): B-splines need not pass through control points; convex-hull containment can preserve joint bounds. Full public transcript read.
- [Modern Robotics 10.7 transcript](https://modernrobotics.northwestern.edu/nu-gm-book-resource/10-7-nonlinear-optimization/): trajectory optimization must enforce obstacle and dynamic constraints; local optimization can fail. Full public transcript read.
- [Flash and Hogan, 1985](https://pubmed.ncbi.nlm.nih.gov/4020415/): the abstract describes minimum integrated squared hand jerk, approximately straight unconstrained hand paths and bell-shaped velocity profiles, with reduced speed in high-curvature sections. The paper-index full-text endpoint returned not_licensed, and accessible PDF mirrors could not be retrieved. Claims here are limited to the abstract. Its human-motion model is formulated in hand space, not simply joint space.
- [MoveIt CHOMP](https://moveit.picknik.ai/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.html) and [STOMP](https://moveit.picknik.ai/main/doc/how_to_guides/stomp_planner/stomp_planner.html): optimization combines path smoothness and obstacle handling. These were evaluated as alternatives; no ROS stack, CHOMP or STOMP is installed or claimed implemented.

## Implementation and limits

Quintic soft-guide B-splines have repeated endpoint controls for zero endpoint velocity and acceleration; simple interior knots give C4 arm curves. Joined task phases are C2. Exact polynomial derivative extrema determine a whole-phase duration that respects the selected speed cap and unchanged 80°/s² acceleration / 600°/s³ jerk arm bounds. Sampling remains 30 Hz. Gripper uses an independent quintic opening/closing schedule with its original bounds, opening early on reach and delaying closure on return. Grasp/release/settle minima remain. Travel speed still ranges 50–150%.

This is a human-inspired coordinated route, not an implementation of Flash/Hogan's Cartesian optimal-control model, a guarantee of human appearance, or a globally optimal collision-avoidance planner. Softening a route may cut into obstacles. Every generated plan must pass the full unchanged contact-physics, tracking, fresh-scene and hardware execution gates. A failed new route stays blocked; there is no hidden fallback or collision-mask relaxation.

## Same-scene simulation

| Task | Natural 100% | Natural 150% | Flow 150% reference |
|---|---:|---:|---:|
| Place in bowl | 18.33 s | 13.80 s | 15.63 s |
| Stack and return Home | 32.80 s | 23.87 s | 27.07 s |

All four Natural runs passed task and transfer checks with zero robot-self and robot-workcell contacts. At 150%, maximum arm tracking errors were 7.49° in both scenes, below the unchanged 10° threshold. Natural is not invariably faster: at 100% it prioritizes a coordinated curve and can take longer than Flow.

For the same stack capture at 150%, shoulder pan reached a 1° change at 1.90 s with Natural versus 4.70 s with Flow. Reach duration fell from 7.57 to 6.27 s; commanded gripper-center reach length fell from 448.8 to 384.6 mm. This demonstrates overlap of lift and turn, independently of total runtime. Plot: output/natural-diagnostics/reach-path-comparison.png; numerical metrics: path-metrics.json. These are commanded kinematic paths, not measured real trajectories.

Source captures: a4f89f79aa54423f9e784536e55f4fe6 (bowl), 2e6fbf4f58ae44eca1fa30c9b35f9af8 (stack). Artifacts are isolated under output/natural-diagnostics. Offline comparison suppresses video rendering and does not dispatch hardware.

## Verification and fresh hardware test

Full Python regression: 155 passed (two existing dependency warnings). Three further parameterized macro-assembly cases at 50/100/150% passed after review; all six Natural tests passed. Three playback tests and JavaScript syntax checks passed. Independent review found no critical or important issue; it prompted explicit wording that the gripper schedule is C2 and a timing window alone is not proof of obstacle clearance.

Fresh UI run `087931daf6b74b619baf7015ea389f25` passed simulation but capture expiry correctly blocked execution. Recaptured, reselected all three objects with SAM, replanned and resimulated as `24e8dcc3d87b4169b9f3f634541fe243`. Natural at 150% passed simulation and transfer checks with 707 actions / 23.57 s, then passed fresh-scene checks and executed all 707 commands. Hardware trace completed without error and returned Home; final arm joint error was at most 2.20 native degrees. Paired replay is hardware-comparison.mp4.

**Physical task result: failed grasp, not successful stacking.** Recorded Top/Wrist views show the source cube remaining on the table. Saved physical-review.json marks the run failed in history, despite successful simulation and command completion. Evidence: natural-grasp-review.jpg and natural-final.jpg. This is one physical test and does not establish grasp reliability.

At the Approach endpoint (6.73 s), target jaw-center FK was [245.15, 135.57, 24.00] mm and encoder-derived FK was [238.21, 124.67, 17.38] mm, a 14.5 mm discrepancy. This is model-derived, not an independent physical pose measurement. The previous successful Flow capture used a different source location, so it is not a controlled real-world comparison and does not isolate the cause to path geometry, tracking or calibration. No guessed offsets, relaxed collision masks or safety thresholds were introduced. The motion-shape improvement is supported by the same-capture offline comparison; physical grasp reliability at this new placement remains unresolved.

## Follow-up: guarded Natural grasp — physical success

User requested fixing actual retention while retaining Natural travel. Added a Natural-only Pregrasp endpoint 45 mm above the grasp target, a centered 1.5 s minimum final descent, and 0.4 s minimum open-jaw settle before closure. Coordinated soft-guide reach now ends at Pregrasp; curved carry and return are unchanged. Shared calibration, contacts, actuator settings, speed/acceleration/jerk limits and execution gates remain unchanged. The first failed regression required these endpoint/hold guarantees before implementation. The Classic duration estimate excludes Natural-only guard points; the archived successful run predates this reporting-only correction, so its Classic comparison is not used as a performance claim here.

The hypothesis was that mixing the final turn and descent made grasp approach sensitive to real joint following errors. At the previous Approach endpoint, encoder-derived jaw midpoint differed from its target by 14.5 mm, with elbow error 2.46°. In the new run the discrepancy was about 12 mm and elbow error 0.73°. The settle period did not itself change the quantized encoder readings. The new capture's cube location also differed by about 8 mm following the failed attempt, so this is evidence of a successful corrected trajectory, not a controlled proof that one factor alone caused the earlier failure or a full calibration correction.

Offline full-contact simulations at 150% passed all existing task and transfer gates on the failed scene (25.50 s), earlier bowl scene (15.77 s), and earlier stack scene (25.77 s). Zero robot-self/workcell contacts in all three; artifacts in output/natural-grasp-diagnostics/{failed,bowl,stack}-guarded. Existing finger/table contact checks were retained.

Fresh run **dc7fe3a0b44040d6838eb254eb28e2b0**: SAM-selected current source, destination and bowl obstacle; Natural 150%; 767 actions / 25.57 s. Full simulation and fresh-scene preflight passed before hardware. Hardware completed all commands without error. During carry (10.5–13.5 s), the measured gripper remained at 8.5308 native units, consistent with loaded retention instead of empty closure. Recorded Top/Wrist frames show the cube retained during lift/carry, released onto the destination cube, empty jaws withdrawing, and the stack still standing after automatic Home. Physical review is **success**, visible in Run history as Placement verified. Evidence: natural-{grasp,carry,release,withdraw,final}.jpg, natural-success-review.jpg, hardware-comparison.mp4 and physical-review.json in the run directory. This is one successful physical trial, not a general reliability estimate. The earlier failed run remains recorded as failed.

Validation: full suite 157 passed plus one process-inspection permission failure; that single test passed with process inspection enabled. Added guarded/legacy macro derivative checks at 50/100/150%; all 14 focused planner/Natural tests passed, including the Classic comparison regression. JavaScript syntax and diff whitespace checks passed. Independent review found no blocking issue; its reporting caveat was corrected.

Pre-push verification on the final code: 161 Python tests passed (two existing dependency deprecation warnings), all three playback tests passed, JavaScript syntax and diff whitespace checks passed.
