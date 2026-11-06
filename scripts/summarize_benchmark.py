"""Validate saved evidence and assemble the 72-episode result bundle."""
import copy
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import mujoco
from benchmark72 import BATCH, ROOT

def summarize():
    tags=['baseline','oriented-gripper','repaired']
    reports={tag:json.loads((BATCH/tag/'evaluation.json').read_text()) for tag in tags}
    rows=reports['repaired']['results'];common=[];frame_total=0
    dynamics=[]
    for row in rows:
        folder=BATCH/'repaired'/f"episode-{row['episode']:02d}"
        cal=json.loads((folder/'calibration.json').read_text())
        cal['world']['cube'].pop('position');cal['world']['cube'].pop('quaternion',None)
        cal['world']['bowl'].pop('position')
        common.append(hashlib.sha256(json.dumps(cal,sort_keys=True).encode()).hexdigest())
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        assert model.neq==0 and model.nexclude==0
        dynamics.append(hashlib.sha256(b''.join(v.tobytes() for v in [model.actuator_gainprm,model.actuator_biasprm,model.actuator_forcerange,model.dof_damping,model.dof_armature,model.geom_contype,model.geom_conaffinity,model.geom_friction])).hexdigest())
        trace=np.load(folder/'physics_trace.npz')
        assert len(trace['qpos'])==row['frames']==len(trace['cube_pose'])
        assert np.isfinite(trace['qpos']).all() and np.isfinite(trace['cube_pose']).all()
        assert np.max(np.abs(np.linalg.norm(trace['cube_pose'][:,3:],axis=1)-1))<1e-6
        assert row['cube_pose_writes_after_initialization']==0
        assert not row['external_object_forces'] and not row['weld_constraints']
        assert np.max(np.abs(trace['timestamps']-np.arange(row['frames'])/30))<.00101
        frame_total+=row['frames']
    assert len(set(common))==1 and len(set(dynamics))==1
    summary={'episodes':72,'frames':frame_total,'counts':{tag:reports[tag]['outcome_counts']for tag in tags},
             'shared_calibration_verified':True,'shared_dynamics_and_collision_masks_verified':True,
             'calibration_excluding_initial_object_poses_sha256':common[0],
             'dynamics_and_collision_masks_sha256':dynamics[0],
             'max_repaired_hand_translation_m':max(r['trajectory_repair']['max_hand_translation_m']for r in rows),
             'max_repaired_joint_correction_deg':max(r['trajectory_repair']['max_joint_correction_deg']for r in rows),
             'max_ik_position_residual_m':max(r['trajectory_repair']['max_ik_position_residual_m']for r in rows),
             'max_contact_penetration_m':max(max(a['max_penetration_m']for a in r['contact_audit'].values())for r in rows),
             'max_cube_contact_force_N':max(r['max_contact_normal_N']for r in rows),
             'scope':'Task success in one deterministic run on the calibration demonstrations, not held-out accuracy or exact real-motion reproduction.'}
    (BATCH/'summary.json').write_text(json.dumps(summary,indent=2))
    with (BATCH/'episodes.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['episode','frames','baseline','calibrated_replay','trajectory_repaired','repaired_max_cube_z_m','repaired_bilateral_intervals','repaired_grasp_quarter_turn','max_hand_translation_m'])
        for i,r in enumerate(rows):writer.writerow([i,r['frames'],*[reports[t]['results'][i]['outcome']for t in tags],r['max_cube_z'],r['bilateral_frames'],r['trajectory_repair']['grasp_quarter_turn'],r['trajectory_repair']['max_hand_translation_m']])
    text=f'''# SIMCUBE: 72-episode physical replay audit

All 72 episodes ({frame_total:,} frames, 30 Hz) were processed with one shared calibration per run. Initial cube position/orientation and bowl position come from each episode's first top-camera image. No per-episode simulation-outcome optimization or hand-picked object locations are used in the final run.

| Run | Success | Runtime/input errors | Interpretation |
|---|---:|---:|---|
| Archived episode-0 configuration | {reports['baseline']['successes']}/72 | {reports['baseline']['runtime_errors']} | Original detector and wrist clearance; 62 clips integrated |
| Shared calibrated replay | {reports['oriented-gripper']['successes']}/72 | {reports['oriented-gripper']['runtime_errors']} | Measured arm-state targets, recorded gripper action; vertical tabletop-clearance correction |
| Shared trajectory-repair controller | {reports['repaired']['successes']}/72 | {reports['repaired']['runtime_errors']} | Image-conditioned approach, grasp/transport/release waypoints and gripper timing |

The same geometric scoring rule is applied to all three archived runs. Early provisional figures used a 28 mm center-distance rule; that incorrectly rejected cubes resting against the inner bowl wall. Consequently the archived baseline changes from 2 to 3 successes, and calibrated replay from 34 to 39. No simulation state was changed by rescoring.

## Success rule

Cube center must rise above 55 mm and have bilateral finger contact during at least two sampled intervals. At the final state, **all eight cube corners** must be inside all 32 inner wall planes, above the bowl floor and below the rim, with 0.5 mm contact tolerance. There must be no finger contact, and cube translation over the last 10 frames must stay within 3 mm. This tests final placement; it is not a force-identification or visual-fidelity metric.

## Repairs and limits

- Fixed the loader to identify parquet rows by episode identity when metadata retains stale global indices/shard pointers. Video slices keep their recorded timestamps.
- Fixed red-cube detection near pink gripper parts and the dark red robot screw; fitted bowl outer masks instead of an upper-half color centroid.
- Fitted one robust image-to-grasp-plane homography from the same 72 demonstrations: median planar residual 5.87 mm, maximum 30.45 mm. These are calibration-set results, not unseen-data generalization.
- Replaced wrist-only table clearance with a vertical hand lift that retains XY and orientation. The gripper's excessive inherited damping/inertia was reduced using one common configuration: joint damping 0.1, armature 0.003 and servo kv 0.1. Torque cap remains ±0.35 Nm. These are estimates, not measured motor parameters.
- The optional repair controller replaces the approach and transport path with task-space waypoints anchored to the recorded close/release events. It chooses quarter-turn grasp directions using static collision checks and keeps jaw heading fixed during transport. Episodes 30 and 63 select opposite 90-degree turns to avoid the bowl.
- The repair is a substantial trajectory adaptation: maximum requested hand displacement {summary['max_repaired_hand_translation_m']*100:.1f} cm, maximum joint-target change {summary['max_repaired_joint_correction_deg']:.1f} degrees. Unreachable elevated waypoints are projected by IK, with a maximum residual of {summary['max_ik_position_residual_m']*1000:.1f} mm. **72/72 is not unmodified dataset replay, and does not establish 100% real-world fidelity.**
- Cube pose is initialized once and subsequently integrated by MuJoCo contacts. No object teleportation, weld, or externally applied object force is used. Playback assigns stored poses only for rendering. All physical arm/self, finger/table and workcell collisions remain active, with MuJoCo's standard same-body/direct-parent filtering.
- Collision geometry, 12 g cube mass, friction and cameras remain approximate. Maximum penetration in the final batch is {summary['max_contact_penetration_m']*1000:.3f} mm; peak cube contact force is {summary['max_cube_contact_force_N']:.1f} N. Successful placement alone does not validate these transient forces. Wrist/top appearance and motion still differ from real footage.

## Evidence

- [Per-episode CSV](episodes.csv), [verified summary](summary.json).
- [Baseline evaluation](baseline/evaluation.json), [calibrated replay evaluation](oriented-gripper/evaluation.json), [repaired evaluation](repaired/evaluation.json).
- [Episode 1: successful calibrated replay](oriented-gripper/episode-01/real-vs-sim.mp4).
- [Episode 2: real / failed replay / repaired simulation](episode-02-real-before-after.mp4), all 140 frames, 4.667 seconds.
- [Episode 2: failed replay alone](oriented-gripper/episode-02/real-vs-sim.mp4), [repaired alone](repaired/episode-02/real-vs-sim.mp4).
- [Episode 30: repaired bowl-avoiding grasp](repaired/episode-30/real-vs-sim.mp4).

Each final episode directory contains `calibration.json`, `scene.xml`, `physics_report.json` and `physics_trace.npz` (integrated joints/cube poses, actuator targets, contact peaks, bilateral flags, clearance corrections). `repaired/settings.json` records shared parameters and source-code hashes at execution; `repaired/runtime_sources/` preserves the exact four source files and matches every recorded hash. Later changes add video labels/frozen camera selection and derive the scoring rim height from the collision geometry. The canonical `evaluation.json` files apply this stricter final scoring rule without changing the success counts or rerunning dynamics. Rendering applies the frozen wrist camera from that run's settings; use the rendering script for consistent cameras. Original outputs and the intermediate 68/72 repair batch (`repaired-v1`) are preserved.

The archived baseline has no original saved XML/settings. Its geometric re-evaluation reconstructs the original observed object positions with the current model geometry; baseline model provenance is therefore less complete than the two final batches. Independent read-only review rechecked all 144 final replay/repaired model-and-trace pairs and found no disagreements with the reported classifications. The calibrated replay includes clearance corrections up to 10.49 degrees.

## Reproduce

From `/Users/phawit/Projects/DatasetToSim`:

```sh
.venv/bin/python scripts/benchmark72.py --prepare
.venv/bin/python scripts/benchmark72.py --fit
.venv/bin/python scripts/benchmark72.py oriented-gripper
.venv/bin/python scripts/benchmark72.py repaired
.venv/bin/python scripts/evaluate_benchmark.py baseline oriented-gripper repaired
.venv/bin/python scripts/summarize_benchmark.py
.venv/bin/python scripts/render_benchmark.py repaired 2 30
.venv/bin/python -m pytest -q
```

Rendering requires macOS desktop graphics access. The real dataset and reference robot project are read-only inputs. The final repaired batch passes the saved-state checks for 72 unique episodes, all frame counts, finite states, unit quaternions, shared physics/collision settings, no equality constraints, no explicit collision exclusions, and no object-state writes after initialization.
'''
    (BATCH/'RESULTS.md').write_text(text)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':summarize()
