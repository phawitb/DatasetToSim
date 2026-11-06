# SIMCUBE Real-to-MuJoCo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the supplied SIMCUBE LeRobot dataset replayable and calibratable in a MuJoCo SO-101 scene, with objective alignment reporting.

**Architecture:** A small Python package isolates dataset indexing, joint conversion, scene/calibration construction, replay, and visual reporting.  The package reads source data only and stores calibration/output under this repository.

**Tech Stack:** Python 3.11+, PyArrow, NumPy, OpenCV, MuJoCo, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-simcube-real-to-mujoco-design.md`

## Global Constraints

* Source dataset and reference project are read-only.
* Dataset values are degrees; every MuJoCo qpos/control value is radians and range-checked.
* Calibration is versioned JSON and defaults to `draft`; accepted status requires measured evidence.
* Kinematic and dynamics replays have visibly distinct modes and never share an ambiguous result label.
* All outputs are contained under `output/`.

## Review Focus

* An episode crossing parquet/video shard boundaries must still resolve its exact frame/video locations.
* A malformed dataset must fail with a field-specific error rather than silently using a guessed convention.
* A joint angle outside model limits after conversion must be reported with the joint name and frame.
* Render comparison must not resize or otherwise mutate the reference image silently.
* A calibration with missing residual metrics must not be treated as accepted.

---

### Task 1: Dataset contract and trajectory loader

**Files:**
- Create: `src/dataset_to_sim/dataset.py`
- Create: `src/dataset_to_sim/joints.py`
- Create: `tests/test_dataset.py`
- Create: `tests/test_joints.py`

**Interfaces:**
- Produces: `DatasetIndex(root: Path)`, `Episode(index, frame_indices, timestamps, state_deg, action_deg, video_slices)`, `JointConvention.to_radians(values_deg)`.

- [ ] **Step 1: Write failing tests** for episode 0 frame indexing, 30 Hz timestamps, joint-name validation, degree-to-radian conversion, and named range violations using a synthetic LeRobot shard.
- [ ] **Step 2: Run** `pytest tests/test_dataset.py tests/test_joints.py -v` and confirm the tests fail because package interfaces do not exist.
- [ ] **Step 3: Implement** strict metadata/parquet episode loading and explicit six-joint conversion.
- [ ] **Step 4: Re-run** `pytest tests/test_dataset.py tests/test_joints.py -v` and confirm all pass.

### Task 2: Calibration schema and scene XML builder

**Files:**
- Create: `src/dataset_to_sim/calibration.py`
- Create: `src/dataset_to_sim/scene.py`
- Create: `calibration/simcube-draft.json`
- Create: `tests/test_calibration.py`
- Create: `tests/test_scene.py`

**Interfaces:**
- Consumes: `JointConvention` from Task 1.
- Produces: `Calibration.load(path)`, `Calibration.validate()`, `build_scene_xml(calibration, robot_xml: Path) -> str`.

- [ ] **Step 1: Write failing tests** for draft status, required calibration fields, object geometry, camera entries, and a compiled XML string that includes the six-joint source model.
- [ ] **Step 2: Run** `pytest tests/test_calibration.py tests/test_scene.py -v` and confirm expected missing-interface failures.
- [ ] **Step 3: Implement** versioned calibration validation and XML construction for mat, table, cube, bowl and calibrated top camera while retaining the source wrist camera.
- [ ] **Step 4: Re-run** the two test files and confirm all pass; additionally compile with MuJoCo when installed.

### Task 3: Deterministic replay and rendering

**Files:**
- Create: `src/dataset_to_sim/replay.py`
- Create: `tests/test_replay.py`

**Interfaces:**
- Consumes: `Episode`, `JointConvention`, `Calibration`, `build_scene_xml`.
- Produces: `ReplaySession(...).kinematic_frame(frame: int)`, `ReplaySession(...).pd_frame(frame: int)`, each returning simulator state and named view images.

- [ ] **Step 1: Write failing tests** asserting kinematic qpos equals converted state, frame timestamp selection is exact, and PD/kinematic result metadata differ.
- [ ] **Step 2: Run** `pytest tests/test_replay.py -v` and confirm failures are due to absent replay implementation.
- [ ] **Step 3: Implement** headless MuJoCo model/data creation, state setting, controller stepping, and native top/wrist rendering.
- [ ] **Step 4: Re-run** `pytest tests/test_replay.py -v`; render a requested real episode frame when MuJoCo is available.

### Task 4: Visual evidence and command-line workflow

**Files:**
- Create: `src/dataset_to_sim/report.py`
- Create: `src/dataset_to_sim/cli.py`
- Create: `src/dataset_to_sim/__main__.py`
- Create: `tests/test_report.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Create: `pyproject.toml`

**Interfaces:**
- Consumes: Task 1–3 interfaces.
- Produces: `dataset-to-sim inspect`, `replay`, and `report` commands; `ComparisonReport` JSON and labelled side-by-side PNGs.

- [ ] **Step 1: Write failing tests** for report metrics, shape mismatch rejection, draft calibration rejection for accepted result, and CLI argument validation.
- [ ] **Step 2: Run** `pytest tests/test_report.py tests/test_cli.py -v` and confirm expected failures.
- [ ] **Step 3: Implement** report metric generation, source-video extraction, contact sheets, command-line commands and concise setup/calibration documentation.
- [ ] **Step 4: Run** `pytest -v` and a real-dataset inspect/replay/report smoke test.  Preserve outputs under `output/`.

## Self-review

* Coverage: Task 1 covers dataset indexing and units; Task 2 scene/calibration; Task 3 MuJoCo replay; Task 4 evidence, CLI and all listed review-focus cases.
* Interfaces: all later tasks consume only named Task 1–3 types/functions.
* Proportion: implementation details remain in code/tests; this plan fixes only contracts and verification boundaries.
