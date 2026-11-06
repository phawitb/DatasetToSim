"""Run/export native recorded actions without any trajectory repair."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import platform
import importlib
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from benchmark72 import DATA, DatasetIndex, detect_scene, read_real
from episode0 import ROOT, ROBOT, Calibration
from episode0_physics import render_video
from strict_action_scene import make_configuration, episode_scene, build_model, containment, estimate_scene
from dataset_to_sim.strict_replay import NativeJointAdapter, integrate_actions, array_hash
from dataset_to_sim.archive import load_saved_model

DEST=ROOT/'output/strict-action'


def score(trace,inside):
    if float(trace['cube_pose'][:,2].max())<float(trace['cube_pose'][0,2]+.04):return 'not_lifted'
    if trace['bilateral'].sum()<2:return 'no_bilateral_grasp'
    support=trace['audit'].get('grasp_support')
    if support and (support['longest_secure_grasp_s']<.08 or support['longest_secure_lift_s']<.03):return 'no_bilateral_grasp'
    if not inside['inside']:return 'missed_bowl'
    if trace['audit']['final_finger_contact']:return 'not_released'
    if np.linalg.norm(trace['cube_pose'][-10:,:3]-trace['cube_pose'][-1,:3],axis=1).max()>.003:return 'not_settled'
    if support and (trace['audit']['final_cube_linear_speed_m_s']>.03 or trace['audit']['final_cube_angular_speed_rad_s']>.3):return 'not_settled'
    return 'success'


def run(episodes,configuration=None):
    DEST.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(configuration).read_text()) if configuration else make_configuration();adapter=NativeJointAdapter(**config['adapter']);ds=DatasetIndex(DATA)
    sources=['scripts/strict_action_replay.py','scripts/strict_action_scene.py','scripts/benchmark72.py','src/dataset_to_sim/strict_replay.py','scripts/episode0_physics.py','scripts/episode0.py','src/dataset_to_sim/scene.py','src/dataset_to_sim/dataset.py','src/dataset_to_sim/scene_estimation.py']
    sources.append('src/dataset_to_sim/archive.py')
    config['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in sources}
    config['dataset_input_sha256']={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [DATA/'meta/info.json',*sorted((DATA/'data').glob('chunk-*/*.parquet')),*sorted((DATA/'meta/episodes').glob('chunk-*/*.parquet')),*sorted((DATA/'videos').rglob('*.mp4'))]}
    config['robot_asset_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROBOT,*sorted((ROBOT.parent/'assets').glob('*.stl'))]}
    config['runtime_versions']={'python':platform.python_version(),**{p:importlib.import_module(p).__version__ for p in ['mujoco','numpy','scipy','cv2','pyarrow']}}
    asset_dir=DEST/'assets';asset_dir.mkdir(exist_ok=True)
    for p in sorted((ROBOT.parent/'assets').glob('*.stl')):shutil.copy2(p,asset_dir/p.name)
    config['model_archive']='self-contained per-episode XML with one shared copy of source mesh assets'
    config['requested_episodes']=episodes
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    (DEST/'settings.json').write_text(json.dumps(config,indent=2))
    if configuration is None:(ROOT/'calibration/strict-native-action.json').write_text(json.dumps(config,indent=2))
    follower_bytes=Path(config['follower_calibration_source']).read_bytes()
    assert hashlib.sha256(follower_bytes).hexdigest()==config['follower_calibration_sha256']
    (DEST/'follower-calibration-snapshot.json').write_bytes(follower_bytes)
    for p in sources:
        destination=DEST/'runtime_sources'/p;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/p).read_bytes())
    rows=[]
    for i in episodes:
        ep=ds.episode(i);folder=DEST/f'episode-{i:02d}';folder.mkdir(exist_ok=True)
        row={'episode':i,'frames':ep.frame_count,'settings_sha256':config_hash,'native_action_sha256':array_hash(ep.action_deg)}
        try:
            scene=estimate_scene(config,read_real(ep,'top')[0]);cal=episode_scene(config,scene);model=build_model(cal,config)
            (folder/'calibration.json').write_text(json.dumps(cal.data,indent=2));(folder/'scene-observation.json').write_text(json.dumps(scene,indent=2))
            trace=integrate_actions(model,ep,adapter)
            np.testing.assert_array_equal(trace['native_action_applied'],ep.action_deg)
            np.testing.assert_array_equal(trace['step_controls'],trace['model_targets'][trace['step_action_index']])
            np.testing.assert_allclose(trace['sample_times'][:-1],ep.timestamps,atol=1e-10,rtol=0)
            native_state=adapter.to_native(trace['qpos'])
            inside=containment(model,trace['qpos'][-1],trace['cube_pose'][-1],cal,config)
            row.update(outcome=score(trace,inside),joint_state_rmse_native=np.sqrt(np.mean((native_state[:-1]-ep.state_deg)**2,axis=0)).tolist(),max_cube_z_m=float(trace['cube_pose'][:,2].max()),bilateral_intervals=int(trace['bilateral'].sum()),final_cube=trace['cube_pose'][-1,:3].tolist(),bowl_containment=inside,terminal_time_s=float(trace['sample_times'][-1]),cube_mass_kg=float(model.body('cube').mass[0]),audit=trace['audit'],model_ctrlrange_exceedances_per_joint=np.count_nonzero((trace['model_targets']<model.actuator_ctrlrange[:,0])|(trace['model_targets']>model.actuator_ctrlrange[:,1]),axis=0).tolist())
            row['action_equality_verified']=True
            row['initial_collision_screen_passed']=trace['audit']['initial_max_penetration_m']<=.001
            row['physical_model_validated']=False
            row['physical_validity_note']='Initial penetration screen only (1 mm threshold); passing does not validate real-world dynamics.'
            trace['simulated_native_state']=native_state;trace['recorded_native_state']=ep.state_deg.copy();trace['source_timestamps']=ep.timestamps.copy()
            np.savez_compressed(folder/'strict_trace.npz',**{k:v for k,v in trace.items()if k!='audit'})
            row['trace_archive_sha256']=hashlib.sha256((folder/'strict_trace.npz').read_bytes()).hexdigest()
            mujoco.mj_saveLastXML(str(folder/'scene.xml'),model)
            xml=ET.parse(folder/'scene.xml');xml.getroot().find('compiler').set('meshdir','../assets');xml.write(folder/'scene.xml',encoding='unicode')
            row['model_archive_sha256']=hashlib.sha256((folder/'scene.xml').read_bytes()).hexdigest()
        except Exception as error:
            row.update(outcome='error',error=f'{type(error).__name__}: {error}')
        (folder/'report.json').write_text(json.dumps(row,indent=2));rows.append(row)
        counts=dict(Counter(r['outcome']for r in rows))
        report={'mode':'strict_native_action','dataset':str(DATA),'total_requested':len(episodes),'completed':len(rows),'settings_sha256':config_hash,'counts':counts,'successes':counts.get('success',0),'success_count_meaning':'task criteria only; not validated physical fidelity','physically_validated_successes':None,'task_successes_passing_initial_collision_screen':sum(r['outcome']=='success' and r.get('initial_collision_screen_passed',False) for r in rows),'errors':counts.get('error',0),'frames':sum(r['frames']for r in rows),'native_action_changed':False,'trajectory_repair':False,'state_following':False,'gravity_compensation':False,'joint_rmse_units':['degree']*5+['normalized gripper 0-100'],'score_time':'after executing final recorded action for one nominal frame interval; videos show N observation-time states','results':rows}
        (DEST/'results.json').write_text(json.dumps(report,indent=2))
        print(i,row['outcome'],'max z',round(row.get('max_cube_z_m',0),3),row.get('error',''),flush=True)
    print('TOTAL',report['successes'],'/',len(episodes),'errors',report['errors'],flush=True)
    return report


def render(episodes):
    settings=json.loads((DEST/'settings.json').read_text());ds=DatasetIndex(DATA)
    sources=['scripts/strict_action_replay.py','scripts/episode0_physics.py','src/dataset_to_sim/observations.py','src/dataset_to_sim/archive.py']
    manifest={'note':'Rendering-only sources; simulation source snapshot is preserved separately. Binary model playback skips XML re-export.','sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (DEST/'render-manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in sources:
        target=DEST/'render_sources'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    for i in episodes:
        folder=DEST/f'episode-{i:02d}';cal=Calibration.load(folder/'calibration.json');report=json.loads((folder/'report.json').read_text());model=load_saved_model(folder,settings,report.get('model_archive_sha256'))
        trace=np.load(folder/'strict_trace.npz');ep=ds.episode(i)
        render_video(cal,ep,model,trace['qpos'][:-1],trace['cube_pose'][:-1],folder,label='MuJoCo | ORIGINAL actions',camera_config=settings['wrist_camera'],save_scene_xml=False,camera_delay_frames=settings.get('camera_delay_frames'))
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(folder/'comparison-raw.mp4'),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'real-vs-sim-strict.mp4')],check=True)
        print(folder/'real-vs-sim-strict.mp4',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,nargs='+',default=list(range(72)))
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,default=DEST)
    args=parser.parse_args()
    DEST=args.output.resolve()
    if args.render_only:render(args.episodes)
    else:run(args.episodes,args.config)
