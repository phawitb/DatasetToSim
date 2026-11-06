# DatasetToSim

Replay the real `SIMCUBE_1O1B_VF` LeRobot v3 data in a calibrated MuJoCo SO-101 workcell.  The active strict mode preserves recorded native actions at 30 Hz and reports model uncertainty explicitly. Older state-following modes below are archived diagnostics.

## Active workflow: unchanged real actions

The current candidate uses the user-confirmed geometry and **native gripper zero at modeled fingertip contact**. The latest run is [touch-zero-v1](output/touch-zero-v1/RESULTS.md): **59/72 task passes (81.9%)**, 12 not lifted and 1 insufficient bilateral grasp. All72 complete without runtime errors. This is still an incomplete physical calibration, not a validated digital twin or sim-to-real policy result. The previous61/72 experimental calibration used a different mechanical zero and cube-yaw estimator; its higher score does not override the user's physical measurements.

```sh
.venv/bin/python scripts/strict_action_replay.py --config calibration/native-action-current.json --output output/reproduction
.venv/bin/python scripts/audit_strict_action.py --output output/reproduction
.venv/bin/python scripts/strict_action_replay.py --output output/reproduction --render-only --episodes 0 45 50
.venv/bin/python scripts/audit_gripper_data.py
```

Use the explicit config/output arguments above. The script's no-argument workflow retains the historical baseline.

All **11,661 original six-channel commands / 396,474 physics steps** are audited. Five arm channels are degrees; gripper is normalized0–100, with fixed encoder scale **1.298461538degrees/native unit**. Native0 maps to the unchanged source distal-jaw surfaces' first contact, **−11.433150°**, under the user's closure constraint. This is a model-derived reference, not a measured hardware angle.

The initial state is never forced closed. The user's screenshot matches **episode45, frame0**: measured gripper **0.812457681**, commanded gripper **0.242326334**. All72 initial measured values lie in **0.677048–1.015572**. The model starts at each actual measured state; subsequent motion uses only the original actions. No action clipping, IK, trajectory repair, future-state following, object weld/teleport or holding force is used. No physical robot commands were sent.

Measured geometry: cube **19.5mm** each side; bowl **118mm outer rim diameter ×50mm high**; mat **405×405×2mm**; standard SO-101 fingers and robot directly on the table. Initial cube placement uses the first RGB frame only, measured-size silhouettes and top-face orientation when segmentation is consistent. Its confidence is a lighting/edge heuristic, not proof of an unobstructed face. Initial overlap is zero in this run. Source robot/self, finger/table and object collision categories remain enabled, including explicit distal jaw pairs.

Wood mass4.45g, bowl bottom diameter82.6mm, wall3mm, floor4mm, motor response, friction, historical joint references and camera distortion/timing remain estimates. Wrist camera calibration predates the latest0.568° zero-reference correction; its earlier mask-overlap figure must not be read as validation of this candidate. See the [full report](output/touch-zero-v1/RESULTS.md), [independent trace audit](output/touch-zero-v1/verification.json) and [raw gripper audit](output/system-identification/gripper-native-data-audit.json).

Full comparison videos: [episode0: success](output/touch-zero-v1/episode-00/real-vs-sim-strict.mp4), [episode45: screenshot episode, success](output/touch-zero-v1/episode-45/real-vs-sim-strict.mp4), [episode50: failure](output/touch-zero-v1/episode-50/real-vs-sim-strict.mp4). Left real/right sim; top and wrist views. Exact traces, one frozen configuration, source snapshots, dataset/model hashes and self-contained XML/shared mesh assets accompany the run. macOS rendering requires desktop graphics access.

The commands below are historical diagnostic workflows; some modify trajectories and must not be used as strict native-action replay.

## Setup

The GitHub repository includes code, calibration files, the latest evaluation reports and three comparison videos. Full generated model/trace archives and historical experiments remain local and are excluded from Git. Recreate them with the explicit replay commands above. The source dataset and the reference SO-101 scene-editor assets are external dependencies; current scripts retain their original `/Users/phawit/Projects/` paths. Adjust the path constants in `scripts/episode0.py` and `scripts/benchmark72.py`, and the follower-calibration source path in the chosen calibration JSON, when using another machine. The recorded source hashes intentionally verify the exact inputs used for the reported run.

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
