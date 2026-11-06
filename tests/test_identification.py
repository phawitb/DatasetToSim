import numpy as np
from dataset_to_sim.identification import held_gripper_frames


def test_small_opening_grasp_plateau_is_used_but_closing_transient_is_not():
    state=np.array([.8]*5+[10,20,24,24,24,20,14,9]+[8.4]*12+[12,20,30,20,10,5,1]+[.8]*5)
    action=np.array([.2]*5+[20]*5+[0]*15+[30]*3+[0]*9)
    mask=held_gripper_frames(state,action)
    assert mask[15:23].all()
    assert not mask[:13].any()
    assert not mask[25:].any()


def test_unblocked_free_gripper_is_not_a_held_object_observation():
    state=np.full(30,8.4);action=np.full(30,8.)
    assert not held_gripper_frames(state,action).any()


def test_in_place_motor_inertia_fit_matches_fresh_rebuilt_model():
    import copy,json,sys
    from pathlib import Path
    root=Path(__file__).parents[1];sys.path.insert(0,str(root/'scripts'))
    from identify_dynamics import fast_trace,DatasetIndex,DATA
    from strict_action_scene import episode_scene,build_model
    from dataset_to_sim.strict_replay import NativeJointAdapter
    config=json.loads((root/'calibration/native-action-current.json').read_text())
    scene={'cube_pixel':[400,240],'bowl_pixel':[480,176],'cube_pixel_corners':[[390,230],[410,230],[410,250],[390,250]]}
    changed=copy.deepcopy(config);changed['identified_dynamics']['armature'][1]*=2
    mutated=build_model(episode_scene(config,scene),config)
    mutated.dof_armature[1]=changed['identified_dynamics']['armature'][1]
    rebuilt=build_model(episode_scene(changed,scene),changed)
    e=DatasetIndex(DATA).episode(0);adapter=NativeJointAdapter(**config['adapter'])
    q_mutated,cube_mutated=fast_trace(mutated,e,adapter)
    q_rebuilt,cube_rebuilt=fast_trace(rebuilt,e,adapter)
    np.testing.assert_allclose(mutated.dof_invweight0,rebuilt.dof_invweight0,atol=1e-12)
    np.testing.assert_allclose(q_mutated,q_rebuilt,atol=1e-10,rtol=0)
    np.testing.assert_allclose(cube_mutated,cube_rebuilt,atol=1e-10,rtol=0)
