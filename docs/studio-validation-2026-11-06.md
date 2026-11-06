# Studio playback and authorized live validation — 2026-11-06

## Playback failure and correction

The original website could play only eight pre-rendered benchmark episodes. Other episodes showed individual frames and disabled Play. A live capture was a still image paired with a moving simulation, although the display suggested synchronization. Repeated plan refreshes could reload the media source, and late image requests could replace a newly selected view.

The website now prepares complete comparison videos on demand from the saved physics traces for all 72 episodes. A single video decoder and clock drive both canvases. Media readiness controls Play; time updates, seeking and the final frame repaint both views. Replaying after the end resets time, while refreshing the same plan preserves playback. Stale image responses are ignored. Camera changes during a recorded hardware replay use the recording directly; live motion never reloads the captured still.

Before execution, the display says “Capture still + sim video.” During execution it previews fresh camera images and follows the command clock at 1×. The final real/sim recording combines both sources in one video with no motion-dependent time warping. Raw camera acquisition timestamps, recorder sample times and actual command times are saved for inspection. Live previews have transport/display latency; synchronized recording playback is the appropriate comparison afterward.

Browser checks covered episodes 00, 01 and 02 (including generating the previously missing full videos), end-to-end playback, replay after the end, pause/resume, seeking, top/wrist switching, recorded hardware replay and page reload. No browser errors were reported in the final check.

## Physical run

The user explicitly authorized physical robot execution and stated that the cube, bowl, robot and cameras were ready. Studio connected to the existing SO-101 configuration and captured both cameras and six measured joint positions.

Artifact directory: `output/studio/f191aaf0562f44dcbe7076f52379f6f6/` (local generated output).

- IK plan: 629 native commands, approximately 20.97 seconds, starting from the actual captured pose.
- MuJoCo result: successful secure lift, release and settled bowl containment; transfer checks passed.
- Hardware result: all 629 commands sent; last command recorded at 20.9397 seconds; no worker error or protective abort.
- Maximum measured-to-current-target difference: 4.58°, 7.06°, 6.34°, 4.29°, 4.10° for the five arm joints, and 8.53 native gripper units. Measurements precede the current send, so these are conservative raw differences, not latency-corrected tracking errors.
- Actual task result: the final wrist camera image shows the red cube inside the green bowl. The robot remained holding its final measured pose.
- Comparison video: 630 frames at 30 fps, 21.0 seconds, with real and simulated top/wrist views. The approximately 5 ms recorder tail after the last paired frame is reported explicitly.

Evidence: `capture.json`, `plan.json`, `plan.npz`, `trace.npz`, `hardware-trace.json`, `hardware-video.json`, `hardware-raw.mp4`, `hardware-comparison.mp4`, `execution.json`, and `real-final.jpg` in the directory above.

This is one observed live success, not a general success-rate estimate. Scene detection still assumes the calibrated camera, red 19.5 mm cube and green bowl. Another run needs a new capture and a cube placed in a supported starting position.

## Automated verification

- Python project suite: 70 passed (two pre-existing dependency deprecation warnings).
- Node playback regression suite: 3 passed.
- JavaScript syntax check and Git whitespace check passed.

Automated hardware tests use fake hardware; the physical result above is separately supported by the actual recorded run.

## Follow-up: Prepare arm and a second physical pick test

Added step 0 before Capture with Return Home, Torque off and Torque on. Home generates a smooth native trajectory from measured encoders through a calibrated elevated corridor and compact rest pose. Its trajectory is simulated with collision/tracking checks and dispatched through the same guarded executor. Stop during planning cancels dispatch; a closed control page also cancels Home. The Home profile is tied to the existing follower calibration hash.

Torque on writes each raw measured position as an acknowledged target, verifies all target readbacks, then enables and verifies torque. A partial failure marks torque state unknown and blocks capture/execution. Torque off/on invalidates previous captures; Stop never reenables disabled torque. The UI asks the operator to support the arm before releasing torque.

A live Home test (`20a2d6a4376b430181c4df3836bccc27`) completed 392 commands in about 13 seconds. The camera then saw both objects clearly. Torque off/on were tested at the supported rest pose; Capture and Home were disabled while torque was off.

The user had moved the bowl to the left and cube to the right. Live detection now covers both sides of the mat and uses a pale-green HSV mask, avoiding the original benchmark detector's fixed right-side crop. The strict benchmark detector remains unchanged. An initial preflight rejected a lighting-sensitive detection without sending commands; the detector was corrected and a fresh scene was captured and simulated.

The second physical pick (`0f32d487ce944fab9f6dd8f0879b4f36`) completed all 623 commands with no control error (last sample 20.7382 s), but **the real task failed**: the cube slipped during lift and remained on the mat. The simulation had passed. Maximum raw measured-to-current-target differences were [4.29, 6.98, 6.47, 4.28, 4.05] degrees and 5.10 native gripper units. Camera evidence is in `grasp-inspection.jpg`, `hardware-comparison.mp4`, and `physical-review.json`. This exposes remaining grasp/contact/alignment uncertainty at the new location; the successful prior pick must not be generalized to all positions.

Home also handles partial bowl occlusion after execution. It may reuse a verified capture for up to 600 seconds only when the prior actual execution completed, current encoders remain within the existing 10° arm tracking envelope (5 gripper units) of its last measured pose, and the current image contains both visible bowl remnants and the orange arm over the expected bowl region. The image evidence and age are checked again before dispatch. If these checks fail, Home stops with an explanation. This is a calibrated workcell heuristic, not general obstacle detection.

Final follow-up status: 87 Python tests and 3 playback tests passed. A reconnect-specific issue was also removed: Studio now opens the already-calibrated motor bus without LeRobot's configure-time torque toggle, reads the existing torque state, and preserves it. The final physical reconnect succeeded. The second post-pick Home attempt was not dispatched because its hidden-bowl reference expired during diagnosis. The arm remains stationary with torque enabled; the operator was asked to move the bowl clear of the arm so a fresh Home check can proceed. It was not automatically released while elevated.
