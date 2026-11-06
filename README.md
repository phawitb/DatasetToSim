# DatasetToSim

Replay the real `SIMCUBE_1O1B_VF` LeRobot v3 data in a calibrated MuJoCo SO-101 workcell.  The active strict mode preserves recorded native actions at 30 Hz and reports model uncertainty explicitly. Older state-following modes below are archived diagnostics.

## Active workflow: unchanged real actions

Latest confirmed geometry: **cube19.5mm**, bowl **118mm outer rim /98mm outside bottom /50mm height**, mat **405×405×2mm**; standard SO-101 fingers, base on the tabletop. Native gripper0 is modeled at first fingertip contact. Each episode starts from its actual measured state, including its nonzero gripper opening.

The main shared calibration now passes **68/72 episodes (94.4%)**, versus65/72 in the previous measured-bowl run. It recovers **53,63,70** with **no previously successful episodes lost**. **Four remain unresolved:37,44,50,51.** This is an incomplete calibration-set replay, not a validated digital twin or a proven policy-transfer result.

| Full72 shared profile | Task passes | Arm state RMSE | Gripper state RMSE |
|---|---:|---:|---:|
| Previous touch-zero-v1 |59/72|0.759degrees|3.144native units|
| Previous measured-bowl-task-v7 |65/72|0.695degrees|3.058native units|
| Main: shared-calibration-v10 |68/72|0.697degrees|3.021native units|
| Alternative: measured-bowl-state-v7 |64/72|0.639degrees|2.632native units|

Each row uses one fixed calibration across all72 episodes. Results are never mixed episode-by-episode. The latest profile improves task count and gripper-state error; arm-state error is slightly higher than v7 (0.0025degrees). The lower-state-error alternative remains separate. All scenes were available during development, so these are not blind generalization results.

```sh
.venv/bin/python scripts/strict_action_replay.py --config calibration/native-action-current.json --output output/reproduction
.venv/bin/python scripts/audit_strict_action.py --output output/reproduction
.venv/bin/python scripts/strict_action_replay.py --output output/reproduction --render-only --episodes 0 37 44 50 51 53 63 70
# Alternative complete shared calibration with lower state error:
.venv/bin/python scripts/strict_action_replay.py --config calibration/native-action-state-fit.json --output output/state-reproduction
.venv/bin/python scripts/audit_gripper_data.py
```

Use these explicit config/output arguments. The no-argument replay workflow remains the historical baseline.

The latest main run independently verifies **11,661 original six-channel commands /396,474 physics steps**, original action boundaries, fixed adapter equality, zero initial overlap, archived sources/model hashes and the measured98mm bowl-bottom collision envelope. Five arm channels are degrees; gripper is normalized0–100, with fixed scale **1.298461538degrees/native unit**. Native0 maps to source distal-jaw contact at **−11.433150degrees**, a model interpretation of the user constraint rather than a measured hardware angle. Only state0 initializes the robot. No action repair, IK, command retiming/clipping, later measured-state control, object weld/teleport or external holding force is used.

The screenshot previously supplied by the user matches episode45/frame0: measured gripper0.812457681, command0.242326334. All72 initial states are0.677048–1.015572. Replaying commands does not force subsequent simulated state to equal recorded state.

The cameras and causal2-frame observation delays are shared across episodes; actuator command times remain unchanged. Actual integrated-image comparison on the same60frames/10episodes is recorded in the linked image report. These orange-mask overlaps are silhouette proxies, not percentages of physical accuracy.

[Main results](output/shared-calibration-v10/RESULTS.md) · [Before/after comparison](output/shared-calibration-v10/comparison.json) · [Command/geometry audit](output/shared-calibration-v10/verification.json) · [Image comparison](output/shared-calibration-v10/image-comparison.json) · [Alternative results](output/measured-bowl-state-v7/RESULTS.md).

Full main-profile videos: [episode0](output/shared-calibration-v10/episode-00/real-vs-sim-strict.mp4), recovered [53](output/shared-calibration-v10/episode-53/real-vs-sim-strict.mp4), [63](output/shared-calibration-v10/episode-63/real-vs-sim-strict.mp4), [70](output/shared-calibration-v10/episode-70/real-vs-sim-strict.mp4), and remaining failures [37](output/shared-calibration-v10/episode-37/real-vs-sim-strict.mp4), [44](output/shared-calibration-v10/episode-44/real-vs-sim-strict.mp4), [50](output/shared-calibration-v10/episode-50/real-vs-sim-strict.mp4), [51](output/shared-calibration-v10/episode-51/real-vs-sim-strict.mp4). Left real/right sim, top and wrist views, original full frame count at30fps.

The latest changes are shared arm-reference offsets/baseXY, jaw sliding friction1.0→1.4, table/mat sliding friction0.6→0.3 and contact time constant6→4ms. Encoder scales, gripper0 contact, object dimensions, base support, collision categories, native actions and their timestamps remain unchanged. Material values remain estimates. Wider reference searches, motor-response alternatives and a mat-edge camera fit did not solve the remaining four without regressions; no episode-specific rescue settings are used.

Contact/motor properties and camera distortion/timing remain estimated. Wooden cube mass4.45g assumes600kg/m³; bowl wall3mm and floor4mm remain unmeasured. All source collision categories and explicit distal-jaw contacts remain active. The motor-fitting helper now refreshes MuJoCo constants after in-place inertia changes; a regression verifies equality with a fresh rebuilt model. Earlier optimizer trial losses are historical evidence; reported final task/state/image metrics come from freshly rebuilt strict replays. No physical robot commands were sent.

The workflows below are historical diagnostics, some with trajectory corrections, and do not implement the active strict contract.

## Reusable Codex skill

The reusable workflow is in [skills/dataset-to-mujoco/SKILL.md](skills/dataset-to-mujoco/SKILL.md), with calibration diagnostics and a reference for this project. Invoke `$dataset-to-mujoco` with the dataset path and robot/scene assets. It preserves native actions by default, verifies units and timing, uses shared calibration, and delivers audited real/sim comparisons. Another dataset may require loader, camera, geometry and timing adaptation; the current SO-101 calibration is not a universal preset.

## Setup

The repository includes code, calibration files and selected evaluation reports/videos. Local refinements may be newer than the public baseline commit. Full generated model/trace archives and historical experiments remain local and are excluded from Git. Recreate them with the explicit replay commands above. The source dataset and the reference SO-101 scene-editor assets are external dependencies; current scripts retain their original `/Users/phawit/Projects/` paths. Adjust the path constants in `scripts/episode0.py` and `scripts/benchmark72.py`, and the follower-calibration source path in the chosen calibration JSON, when using another machine. The recorded source hashes intentionally verify the exact inputs used for the reported run.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
```

The default robot XML points to the supplied scene editor. Override `--robot-xml` for another compatible SO-101 XML model.

## Run

```bash
.venv/bin/python -m dataset_to_sim inspect \
  --dataset /Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF

.venv/bin/python -m dataset_to_sim replay \
  --dataset /Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF \
  --episode 0 --frame 0 --mode kinematic

.venv/bin/python -m dataset_to_sim report \
  --dataset /Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF \
  --episode 0 --frame 0 --mode kinematic
```

Artifacts are written under `output/episode-.../`: real/rendered images, labelled comparisons and `report.json` with normalized residuals.  `calibration/simcube-draft.json` is deliberately **draft**.  Edit it from measured camera/world correspondences; only an `accepted` calibration carrying residual evidence can receive an accepted report verdict.

## Replay modes

* `kinematic` — directly applies the measured joint state after degrees→radians conversion. Use this to align rendered views to real demonstrations.
* `pd` — applies recorded action targets to MuJoCo position actuators. Use it to diagnose dynamics/model error; it is never labelled as visual ground truth.

The source data contains no object pose track or camera calibration.  Object dimensions, poses, camera pose, intrinsics and joint offsets therefore remain explicit editable hypotheses until calibrated; see the design document in `docs/superpowers/specs/`.

On macOS, rendering uses `MUJOCO_GL=cgl`. Run `replay`/`report` from a logged-in desktop session. In a headless shell, run `replay ... --no-render` to validate the exact MuJoCo state without attempting OpenGL rendering.

## Episode 0 full comparison (2026-11-06)

`output/episode0/real-vs-sim.mp4` is a 235-frame, 30 fps, 7.833-second native MuJoCo comparison. Left: real footage; right: simulation; top row: top camera; bottom row: wrist camera.

The reproducible recipe is `scripts/episode0.py`. Install SciPy in addition to the package dependencies. Run `.venv/bin/python scripts/episode0.py` to refit the top camera and arm silhouette; run `.venv/bin/python scripts/episode0.py --video` to render the complete episode using `calibration/episode0.json`. Encode its `comparison-raw.mp4` to H.264 with FFmpeg for broad playback support.

The macOS rendering failure was resolved by running with desktop graphics access outside the restricted process. It was not a missing graphics backend. The CGL setting alone does not grant that access.

Calibration uses manually selected mat corners with estimated 40 cm dimensions, arm masks at frames 0/130/190, and approximate wrist landmarks at frames 0/190. The fitted offsets are episode-specific estimates, not measured servo calibration. `output/episode0/report.json` includes configuration, per-frame mask overlap and limitations; `replay.npz` contains timestamps, joint radians and reconstructed cube positions.

Cube poses are observed at the start and end. During frames 168–197 the pose is inferred from TCP displacement, and frames 198–213 interpolate the release. This is a kinematic visual reconstruction, not a verified contact-driven grasp. The cameras, appearance and gripper still differ from real footage; calibration remains draft. Orange-mask overlap is not a general physical-accuracy percentage.

## Contact-driven episode 0

The newer `output/episode0-physics/real-vs-sim-physics.mp4` uses an actuated MuJoCo simulation. Reproduce with `.venv/bin/python scripts/episode0_physics.py --video`, then encode `comparison-raw.mp4` with FFmpeg. `scripts/fit_wrist_physics.py` fits wrist camera silhouettes from frames 0, 130 and 190, saved in `calibration/wrist-physics.json`.

Cube free-joint pose is initialized once at `[0.175, -0.045, 0.0162]` m. This XY is about 11 mm from the previous image-derived estimate, selected during contact calibration. Every subsequent cube pose comes from `mj_step`, without welds, applied object forces or pose corrections. The renderer reads saved integrated states; its pose assignments are playback only. `physics_trace.npz` records joints, free-cube pose, contact peaks, timestamps and controller targets; `physics_report.json` records outcomes. The scene XML references the original project's meshes.

The simulation uses recorded **state** as position-servo arm targets, recorded **action** for the gripper with an estimated −8° offset, gravity compensation, a 1 ms integration step, an estimated 12 g cube and friction coefficient 1.4. Thin collision-pad proxies supplement the source jaw meshes; they are hidden with other collision geometry in the comparison video. Cube/bowl/table and finger/cube contacts are enabled. Arm self-collision and finger/table collision are excluded in this scoped replay, so this is not yet a validated general-purpose robot/environment model.

Regression coverage checks that the cube lifts over 10 cm, has bilateral finger contact, releases and finishes inside the bowl. Disabling finger contacts must prevent lifting. Run `.venv/bin/python -m pytest -q`. The successful trial reaches z≈0.126 m with 44 sampled intervals containing bilateral contact. Visual alignment and real-world parameter identification remain approximate.

## Full collision revision

The latest script now writes to `output/episode0-full-contact/`, preserving the earlier comparison. All source physical robot colliders, jaw pads, cube, table, mat, floor and bowl participate in collision detection. There are no explicit body-pair exclusions; MuJoCo's standard same-rigid-body and immediate parent/child filters remain. Render-only meshes remain non-colliding because separate physical geometry represents them.

The original initial finger pose penetrated the table by approximately 10 mm. `clear_table_targets` adjusts wrist flex in 0.25-degree increments until tabletop overlap disappears: 20 frames change, at most 10 degrees. These corrections are saved in `physics_trace.npz`; this is an explicitly corrected replay, not exact reproduction of every recorded angle. A convex mesh disk replaces the bowl-bottom cylinder to avoid unstable cylinder/box contact results. Contact parameters use solref `[0.002, 1]` and solimp `[0.99, 0.999, 0.0001]`.

The regression suite deliberately causes nonadjacent shoulder/lower-arm self-contact and finger/table contact, checks clearance correction, and verifies a complete physical pick/release plus the no-finger-contact ablation. The report includes per-category contact count, deepest penetration and peak normal force. This verifies the present episode with estimated materials and geometry; it does not identify the real robot's dynamics.

Run `.venv/bin/python scripts/episode0_physics.py` first to generate the integrated trace, then `.venv/bin/python scripts/fit_wrist_physics.py` to fit six real wrist frames against the actual simulated joints and cube poses, saving `calibration/wrist-full-contact.json`. Finally run `.venv/bin/python scripts/episode0_physics.py --video` and encode `output/episode0-full-contact/comparison-raw.mp4` as H.264. macOS rendering requires desktop graphics access.

## Shared-calibration evaluation of all 72 episodes

**Scope correction:** The user requires unchanged real-robot actions and calibration of the simulator for sim-to-real transfer. Neither historical score below is a strict action-replay result: the 39/72 mode uses measured arm states plus clearance correction, and 72/72 additionally repairs trajectories. Keep these as archived diagnostics. The active requirements are recorded in `calibration/strict-action-contract.json`, and newly measured object dimensions in `calibration/real-measurements.json`. The current recording stack uses degrees for five arm joints and normalized 0–100 for the gripper; the historical `action_deg`/`state_deg` array names incorrectly imply uniform units. The new strict-action baseline achieves 24/72 task-criterion passes; one (episode 0) fails the initial collision screen. This is not a physically validated success rate. See the active workflow above.

The full results are in [output/benchmark72/RESULTS.md](output/benchmark72/RESULTS.md), with per-episode CSV, exact model/trace files and real-versus-sim videos. All 11,661 frames were integrated. A shared calibration with image-derived initial object poses and vertical tabletop-clearance correction succeeds in **39/72** episodes. Adding the shared image-conditioned trajectory-repair controller succeeds in **72/72** in this deterministic calibration-set run.

The repaired mode modifies arm waypoints, wrist heading and gripper timing, so this is task success after trajectory adaptation, not exact demonstration replay or validated real-world accuracy. Cube motion always comes from MuJoCo contacts after initialization. All arm/self, finger/table and workcell collision masks remain enabled; no weld or applied object force is used. The same final scoring rule checks lift, bilateral contact, all cube corners inside the physical bowl walls/floor/rim, released fingers and final translational settling.

```sh
.venv/bin/python scripts/benchmark72.py --prepare
.venv/bin/python scripts/benchmark72.py --fit
.venv/bin/python scripts/benchmark72.py oriented-gripper
.venv/bin/python scripts/benchmark72.py repaired
.venv/bin/python scripts/evaluate_benchmark.py baseline oriented-gripper repaired
.venv/bin/python scripts/summarize_benchmark.py
.venv/bin/python scripts/render_benchmark.py repaired 2 30
```

`calibration/shared72.json` is fitted on these same demonstrations, not a held-out set. Each run freezes its settings and detected scenes. `repaired/runtime_sources/` archives the exact simulation sources matching the recorded hashes; subsequent scoring and video-label changes are documented in the results. The loader resolves compacted trajectory shards using `episode_index` while preserving each video's original slice timestamps.

## Real / Sim Studio (local website)

```sh
.venv/bin/python -m pip install -e '.[studio]'
.venv/bin/python scripts/studio_server.py
```

Open **http://127.0.0.1:8765**. The server binds only to this computer. No hardware is connected when the website starts. Connecting preserves the existing motor torque state and does not run a torque-disabling reconfiguration; use Torque on explicitly if the arm is released.

- **Dataset comparison:** browse all 72 episodes, select top/wrist views, inspect recorded and simulated joint positions on the same timeline. All 72 episodes support full playback at 0.5×/1×/2×. Existing videos open directly; the remaining episodes prepare a cached full video from their archived integrated states when first selected. Both viewports are painted from one decoded video frame, including seeking and replay after the end. The original camera delays are retained.
- **Rehearsal:** select an episode and capture its first recorded frame. Generate a bounded IK plan, then run actual MuJoCo contact physics. This is a newly planned trajectory, separate from the unchanged-action benchmark. Rehearsals cannot be sent to hardware.
- **Live:** connect SO-101, select Live robot cameras, then Prepare arm → Capture → Generate plan → Run simulation → Execute on SO-101. Return Home validates a smooth path from the measured joints through the calibrated rest corridor before moving; it uses the latest workcell model and re-detects the current bowl. If the arm partially hides the bowl, a recent verified scene is allowed only with matching end-pose encoders and visible bowl/arm evidence, checked again before dispatch. Torque off releases the motors for supported manual positioning, and Torque on verifies each measured hold target before enabling. Capture and motion stay locked while torque is off or unknown; changing torque clears the previous capture. Home and all motion remain cancellable with Stop/Escape. The current scene detector supports the calibrated red 19.5 mm cube and green bowl, with the original 640×480 fixed top camera; Live bowl detection covers both sides of the mat. It is not a general object detector or an automatic camera calibration system.

Hardware runs in the existing `/opt/miniconda3/envs/lerobot/bin/python` environment, using only the robot/camera section of `VLA-Benchmark/config.yaml`, the saved follower calibration and its torque-preserving handoff helper. Set `STUDIO_HARDWARE_PYTHON` or `STUDIO_VLA_ROOT` before launch if those locations differ. Disconnect an active VLA-Benchmark robot session before connecting Studio; do not start another controller while Studio owns the serial port.

Execution requires a live capture less than 180 seconds old, unchanged calibration, matching starting joints, a successful physics verdict and transfer checks, and a fresh scene check including cube orientation. Commands retain their simulated native units and timing. Initial encoder readings up to 2 degrees outside the calibrated arm interval may only move back into range; the gripper always stays within 0–100. Runs stop if camera data becomes stale, the initiating page stops sending its heartbeat, command timing slips, encoder tracking exceeds limits, or Stop/Escape is pressed. Stop holds the measured position with torque retained; this is a **software stop, not a hardware emergency stop**. A completed/attempted hardware plan is single use.

Before execution, the real image is explicitly a captured still while the simulated plan plays. During execution, fresh camera previews accompany the simulation at the command clock and 1× speed. After execution, timestamped top/wrist recordings are paired with the simulation into one replayable comparison video. Pairing uses the nearest preceding recorded sample on the shared monotonic command clock; it does not retime motion to hide differences. Original camera acquisition timestamps and raw video are retained.

Generated captures, immutable command arrays, scene models, simulation traces/videos, reports, hardware target/encoder traces and recorded real/sim comparisons are under `output/studio/<capture-id>/`. Existing benchmark evidence is never overwritten. The automated tests use fake hardware. One explicitly authorized live run on 2026-11-06 completed 629 commands over 20.94 seconds; the final wrist image showed the cube inside the bowl. See [playback and hardware validation](docs/studio-validation-2026-11-06.md). A follow-up with cube and bowl on opposite sides completed its commands but the cube slipped during lift; see the same validation log. These runs do not establish a general hardware success rate. The frozen camera, contact and dynamics estimates remain subject to the limitations documented above.

The website is served directly from `web/` without a JavaScript build step. Optional Google fonts have local font fallbacks. Run `.venv/bin/python -m pytest -q` and `node --test tests/test_studio_playback.mjs` to validate the project and playback state machine.
