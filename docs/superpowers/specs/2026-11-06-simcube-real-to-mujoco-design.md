# SIMCUBE real-data replay in MuJoCo

## Intent

Turn the real-robot LeRobot dataset at `/Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF` into an executable MuJoCo replay.  The replay is for inspecting real demonstrations, calibrating a digital twin, and later generating additional, compatible data.  It must preserve the recorded 30 Hz joint trajectory and align its rendered top and wrist views to the recorded videos as closely as the dataset permits.

## Findings from the supplied data

* The dataset is LeRobot Dataset v3, with 72 episodes / 11,661 frames, two synchronized AV1 videos (top and wrist, 640×480 at 30 Hz), and six action/state values.
* The task is `pick up red cube to green bowl`.  A sampled top frame contains the orange SO-101, a black work mat, a red cube and a pale-green bowl; the wrist view is fixed to the end effector.
* Joint values are degrees, despite the feature metadata only naming `.pos`; the supplied SO-101 MuJoCo model uses radians and has the same six joint names in the same order.
* Object pose, camera intrinsic/extrinsic calibration, table/mat dimensions, robot mounting transform, and servo zero offsets are not stored.  Therefore a literal “100% matching” claim would be unsupported.  The system must report pixel/object/joint residuals and mark a calibration as accepted only against configured thresholds.

## Goals

1. Load any complete episode from the provided LeRobot v3 directory, including episode/video time alignment.
2. Build a MuJoCo scene from the existing SO-101 mesh model without modifying that reference project.
3. Replay observed joint states in a deterministic kinematic mode and replay recorded commands in an optional PD/dynamics mode.
4. Model the visible table, mat, red cube and green bowl; save all estimated transforms and camera parameters in versioned JSON.
5. Calibrate top and wrist renderings against selected real frames, using a reproducible configuration plus a residual report and side-by-side image artifacts.
6. Fail clearly when a required model dependency, video mapping, joint range, or calibration field is missing.

## Non-goals for this increment

* Claiming physically correct cube motion before object tracking / calibrated contacts are validated.
* Training or publishing a policy/dataset.
* Overwriting any source parquet/video file or altering the reference scene editor.
* Hiding uncertainty through undocumented hard-coded camera/object poses.

## Approaches considered

1. **Command-only physics replay.** Drive MuJoCo controllers with `action` and let contacts determine the scene.  This is valuable for dynamics validation but cannot visually reproduce the real trial until offsets, friction, latency and object poses are known.
2. **Frame-by-frame state replay.** Set recorded joint state directly and retain static guessed objects.  It gives exact arm timing but does not represent the cube/bowl state after pickup.
3. **Chosen: calibrated hybrid replay.** Treat recorded state as authoritative for arm pose, use a calibration file for static scene geometry and an optional per-frame object-pose track for visual replay, then offer controller replay as a separately labelled diagnostic.  This provides an immediately runnable baseline and a traceable path to high visual fidelity.

## Architecture

`DatasetIndex` reads `info.json`, episode parquet shards, frame parquet shards, and the video slice for an episode. `JointConvention` maps raw degrees to model radians using explicit signs/offsets and checks limits. `SceneBuilder` emits MuJoCo XML based on the source SO-101 model and a self-contained calibration manifest. `ReplaySession` runs kinematic or PD replay and returns frames from the top and wrist cameras. `CalibrationReport` compares rendered/reference images and writes metrics plus contact sheets; it never changes calibration silently.

The project deliberately keeps camera calibration configurable.  The top camera begins as a perspective camera and the wrist begins from the reference model's fixed camera.  A user can supply calibrated intrinsics/extrinsics, while the first implementation supplies colour-mask-based cube/bowl landmarks and image residual measurement for repeatable fitting.  Robot joint pose is the primary constraint for the wrist camera; table/mat corners and object masks are the primary top-camera constraints.

## Data contracts

* Canonical joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
* Input `observation.state` and `action`: degrees; canonical simulator values: radians.
* Kinematic replay applies state at each recorded timestamp; PD replay applies converted action target and advances at configurable 1 ms steps to each timestamp.
* Calibration JSON records schema version, dataset path fingerprint, reference model path, joint convention, world geometry, camera fields, calibration frame selections, optimization result and residual thresholds.
* Report JSON records exact episode/frame identifiers, per-view MAE/RMSE/mask IoU, object pixel centroids, model version and pass/fail.  It does not call a calibration “matched” without evidence.

## Success criteria

* `inspect` reports all 72 episodes and maps episode 0's 235 frames to the correct video interval (0.0–7.833… s).
* A kinematic episode replay produces one 640×480 render per requested frame for both views and its six model qpos values equal the converted recorded state.
* A default calibration compiles the SO-101 model and has all required visible objects/cameras.
* The calibration validator rejects omitted transforms and invalid camera fields, while reports expose rather than suppress residuals.
* A documented calibration can be accepted only when its configured thresholds pass.  Default generated configuration is explicitly `draft`, not accepted.

## Constraints and implementation calls

* Python 3.11+, NumPy, PyArrow, OpenCV, MuJoCo.  Rendering can run headless with `MUJOCO_GL=egl` or `osmesa` where supported.
* The first runnable scene references the supplied editor's `assets/so101/so101.xml` and mesh directory as immutable input.  A CLI option chooses another compatible model path.
* Source files remain read-only; all output goes under this project’s `output/`.
* Initial object dimensions are explicit hypotheses: cube 40 mm; bowl outer radius 75 mm, height 55 mm; mat 600×450 mm.  They are calibration parameters, not measured facts.

## Remaining uncertainty

The first calibration must be fitted/approved from measured real-world dimensions or image correspondences.  The initial draft uses explicit range-compatible joint offsets (`elbow_flex=-2.5°`, `wrist_flex=-5°`) derived from the observed state extrema and the source model limits; they are not yet visually calibrated. The implementation will make the dataset executable immediately, but only a calibration report based on selected real frames can substantiate visual agreement.
