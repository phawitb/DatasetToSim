"""Independently verify exported strict-action traces against source data."""
import hashlib
import json

import mujoco
import numpy as np

from strict_action_replay import DEST, DATA, DatasetIndex, score
from strict_action_scene import containment
from episode0 import Calibration
from dataset_to_sim.strict_replay import NativeJointAdapter, array_hash


def main():
    settings=json.loads((DEST/'settings.json').read_text())
    results=json.loads((DEST/'results.json').read_text())
    config_hash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    assert config_hash==results['settings_sha256']
    assert results['errors']==0
    assert hashlib.sha256((DEST/'follower-calibration-snapshot.json').read_bytes()).hexdigest()==settings['follower_calibration_sha256']
    for path,digest in settings['source_sha256'].items():
        assert hashlib.sha256((DEST/'runtime_sources'/path).read_bytes()).hexdigest()==digest,path
    for path,digest in settings['dataset_input_sha256'].items():
        assert hashlib.sha256((DATA/path).read_bytes()).hexdigest()==digest,path
    for path,digest in settings['robot_asset_sha256'].items():
        from pathlib import Path
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        if settings.get('model_archive','').startswith('self-contained per-episode XML') and Path(path).suffix=='.stl':
            assert hashlib.sha256((DEST/'assets'/Path(path).name).read_bytes()).hexdigest()==digest,path
    ds=DatasetIndex(DATA);adapter=NativeJointAdapter(**settings['adapter'])
    squared_error=np.zeros(6);frames=steps=0;initial_penetrations=[];score_counts={}
    assert [row['episode'] for row in results['results']]==settings['requested_episodes']
    for row in results['results']:
        ep=ds.episode(row['episode']);folder=DEST/f'episode-{ep.index:02d}'
        from dataset_to_sim.archive import load_saved_model
        if 'trace_archive_sha256' in row:
            assert hashlib.sha256((folder/'strict_trace.npz').read_bytes()).hexdigest()==row['trace_archive_sha256']
        trace=np.load(folder/'strict_trace.npz');m=load_saved_model(folder,settings,row.get('model_archive_sha256'))
        assert row['settings_sha256']==config_hash
        assert row['native_action_sha256']==array_hash(ep.action_deg)==row['audit']['action_sha256']
        np.testing.assert_array_equal(trace['native_action_applied'],ep.action_deg)
        np.testing.assert_array_equal(trace['model_targets'],adapter.to_model(ep.action_deg))
        indices=trace['step_action_index'];np.testing.assert_array_equal(np.unique(indices),np.arange(ep.frame_count))
        np.testing.assert_array_equal(trace['step_controls'],trace['model_targets'][indices])
        np.testing.assert_allclose(trace['sample_times'][:-1],ep.timestamps,atol=1e-10,rtol=0)
        ends=np.r_[ep.timestamps[1:],ep.timestamps[-1]+1/30]
        assert np.all(trace['step_start_times']>=ep.timestamps[indices]-1e-10)
        assert np.all(trace['step_end_times']<=ends[indices]+1e-10)
        np.testing.assert_allclose(trace['step_end_times'][:-1],trace['step_start_times'][1:],atol=1e-12,rtol=0)
        np.testing.assert_allclose(trace['sample_times'][-1],ends[-1],atol=1e-10,rtol=0)
        assert np.all(trace['step_end_times']>trace['step_start_times'])
        if 'step_grasp_support' in trace:
            support=trace['step_grasp_support'];assert support.shape==(len(indices),4) and np.isfinite(support).all()
            secure=(support[:,:2].min(axis=1)>=.02)&(support[:,2]<=.0015)
            lifted=secure&(support[:,3]>=trace['cube_pose'][0,2]+.04)
            dt=trace['step_end_times']-trace['step_start_times']
            for mask,name in [(secure,'grasp'),(lifted,'lift')]:
                longest=current=0.
                for yes,interval in zip(mask,dt):
                    current=current+interval if yes else 0.;longest=max(longest,current)
                np.testing.assert_allclose(longest,row['audit']['grasp_support'][f'longest_secure_{name}_s'],atol=1e-9)
                np.testing.assert_allclose(dt[mask].sum(),row['audit']['grasp_support'][f'total_secure_{name}_s'],atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(trace['cube_velocity'][-1,:3]),row['audit']['final_cube_linear_speed_m_s'],atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(trace['cube_velocity'][-1,3:]),row['audit']['final_cube_angular_speed_rad_s'],atol=1e-12)
        for key in ['qpos','qvel','cube_pose']:assert np.isfinite(trace[key]).all()
        np.testing.assert_allclose(m.geom_size[m.geom('cube_geom').id],[.00975]*3)
        assert not m.actuator_ctrllimited.any() and not m.body_gravcomp.any()
        assert m.neq==m.nexclude==0
        assert set(m.geom_contype)>={0,1,2,4,8}
        assert np.all(m.geom_conaffinity[m.geom_contype>0]==15)
        cal=Calibration.load(folder/'calibration.json')
        inside=containment(m,trace['qpos'][-1],trace['cube_pose'][-1],cal,settings)
        outcome=score(dict(trace,audit=row['audit']),inside)
        assert outcome==row['outcome']
        score_counts[outcome]=score_counts.get(outcome,0)+1
        error=adapter.to_native(trace['qpos'][:-1])-ep.state_deg
        squared_error+=(error**2).sum(axis=0);frames+=ep.frame_count;steps+=len(indices)
        initial_penetrations.append(row['audit']['initial_max_penetration_m'])
    assert score_counts==results['counts']
    summary={'verified_episodes':len(results['results']),'verified_frames':frames,'verified_physics_steps':steps,
             'all_native_actions_exactly_equal':True,'all_step_targets_exactly_equal_to_fixed_adapter':True,
             'all_action_boundaries_verified':True,'all_saved_sources_and_dataset_hashes_verified':True,
             'same_settings_sha256':config_hash,'outcomes':score_counts,
             'task_successes_passing_initial_collision_screen':results['task_successes_passing_initial_collision_screen'],
             'physically_validated_successes':None,
             'physical_validity_note':'Task scoring is not physical validation; initial overlap >1 mm fails a diagnostic screen.',
             'frame_weighted_joint_rmse_native':np.sqrt(squared_error/frames).tolist(),
             'joint_rmse_units':['degree']*5+['normalized gripper 0-100'],
             'initial_max_penetration_m':max(initial_penetrations),
             'initial_penetration_episodes_over_1mm':sum(x>.001 for x in initial_penetrations)}
    (DEST/'verification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEST);args=parser.parse_args();DEST=args.output;main()
