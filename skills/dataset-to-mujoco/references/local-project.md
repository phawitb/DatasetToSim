# Existing DatasetToSim implementation

These paths and measurements describe one SO-101 project, not universal defaults.

## Locate and verify

- Workspace: `/Users/phawit/Projects/DatasetToSim`
- Dataset: `/Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF`
- Robot XML: `/Users/phawit/Projects/SO-101 MuJoCo 3D Manipulation Scene Editor/assets/so101/so101.xml`
- Repository: `https://github.com/phawitb/DatasetToSim.git`
- Interpreter: workspace `.venv/bin/python`; FFmpeg/ffprobe also used.

Inspect current files first. Local refinements may be newer than GitHub. If paths are unavailable, locate the user's checkout or obtain the specified source. This skill does not bundle datasets, meshes or a simulator environment.

Read `README.md`, `calibration/real-measurements.json`, `calibration/native-action-current.json` and the selected output's `RESULTS.md`, `results.json`, `verification.json`. The original `calibration/strict-action-contract.json` contains historical values: current audited settings, source and confirmed measurements supersede obsolete status text.

Read recording/normalization code narrowly. Do not dump unrelated VLA application configuration, which can contain credentials.

## Entry points

| File | Purpose / caveat |
|---|---|
| `src/dataset_to_sim/dataset.py` | LeRobot v3 shard/video indexing; currently 30 Hz and a specific six-channel schema |
| `src/dataset_to_sim/strict_replay.py` | Native adapter, causal holds, terminal step, contact/action trace |
| `scripts/strict_action_scene.py` | Geometry, initial RGB estimation, model construction, full-cube containment |
| `scripts/strict_action_replay.py` | Strict simulation/archive and separate render-only mode |
| `scripts/audit_strict_action.py` | Independent audit; some assertions are workcell-specific |
| `scripts/summarize_strict_run.py` | Audited report and available video links |
| `scripts/compare_strict_runs.py` | Recovered/regressed episodes and frame-weighted native errors |
| `scripts/evaluate_replay_images.py` | Integrated top/wrist comparison; fixed evaluation IDs |
| `scripts/identify_dynamics.py` | Fast fitting helper; refreshes model constants before trials |
| `src/dataset_to_sim/identification.py` | Stable blocked-gripper evidence without an opening-25 threshold |

Historical `episode0.py`, `episode0_physics.py`, `benchmark72.py` and `trajectory_repair.py` include state-following/trajectory-repair modes. Some model/render functions are reused, but standalone historical successes are not strict replay results. The strict runner without explicit `--config` uses a historical baseline.

A reference to the old 72/72 score alone does not authorize action changes. Preserve any unchanged-action requirement and explain the difference. If the user explicitly requests the historical repaired mode, reproduce it under that label: inspect its frozen settings, scene observations, evaluation and archived runtime sources; preserve them in an isolated copy or implement a fresh output destination. The legacy runner has no config/output flags and writes fixed destinations. Do not refit or overwrite archived evidence while claiming an exact historical reproduction.

## Reproduce the compatible existing dataset

From the workspace, after verifying paths/profile, use a fresh destination:

```sh
.venv/bin/python scripts/strict_action_replay.py --config calibration/native-action-current.json --output output/reproduction-new
.venv/bin/python scripts/audit_strict_action.py --output output/reproduction-new
.venv/bin/python scripts/strict_action_replay.py --output output/reproduction-new --render-only --episodes 0 37 53 63 70
.venv/bin/python scripts/summarize_strict_run.py --output output/reproduction-new
```

Select video IDs that exist. On this macOS workspace, rendering needs desktop graphics access; request execution escalation through the tool when required. `MUJOCO_GL=cgl` alone does not grant graphics access. Render with MuJoCo, not synthetic substitute images.

The example renders five complete clips, not all72. For all episodes of the compatible existing dataset, omit `--episodes` on the render-only command; its current default is all72. For a new dataset, derive actual IDs first. Record video coverage in the deliverable.

## Adapt another dataset before running

The current strict script **does not support `--dataset`**. `DATA`/`ROBOT` originate in `scripts/episode0.py`; `benchmark72.py` imports them transitively. A config label or output-folder change does not switch datasets.

For a different dataset, implement or use a supported explicit dataset/robot-path interface through loader, runner, renderer and auditor. Record resolved paths/hashes. Do not invent flags or hide bindings in runtime monkeypatches. Preserve the previous benchmark and meaningful tests.

Check inherited assumptions together:

- Six ordered `.pos` channels, absolute position targets, five verified degree channels and normalized gripper. Another encoding/robot requires a verified adapter; do not copy SO-101 scales blindly.
- Loader `fps == 30`, final `1/30` in integrator/auditor/fitting helpers, video slicing, labels and output FPS. Use source timing instead of resampling actions. Preserve the intended camera-delay duration across FPS changes.
- Default `range(72)`, fixed fitting/evaluation IDs and source directories. The loader currently requires contiguous IDs starting at zero. Validate/generalize that restriction and preserve original episode identity; derive actual IDs from metadata, not just the count.
- 640×480 projection, principal point, red-cube detector/ROI, measured cube size, bowl shape, coordinate frames and scorer. The auditor has workcell-specific size assertions too.
- Follower calibration path/hash and robot assets. A current snapshot is not proof of recording-time calibration identity.

The package's `python -m dataset_to_sim inspect --dataset ...` accepts a dataset argument, but shares the restrictive loader; it is not a generic strict conversion command. Only advertise new strict-runner flags after implementing and checking them with `--help` and a real smoke test.

## Verified example, not a promised result

At skill creation on 2026-11-06, the latest audited local run was `output/shared-calibration-v10`: **68/72**, unresolved **37,44,50,51**, **11,661 actions /396,474 steps**, zero initial overlap. Five arm channels were degrees; gripper was 0–100, with **1.298461538 degrees/native unit** and model closed reference **−11.433150029°**. These values belong to this recording/model.

Measured scene: cube19.5mm per side; bowl outside rim118mm/outside bottom98mm/height50mm; mat405×405×2mm; standard fingers/no base riser. Cube mass4.45g, bowl wall3mm/floor4mm and contact properties were estimates, not measured defaults for future scenes.

All72 episodes informed calibration; this is not blind generalization or proof of immediate hardware policy transfer. Re-read outputs for later results. Historical72/72 used trajectory repair and must not be presented as unchanged-action success.
