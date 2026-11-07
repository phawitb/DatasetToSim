import sys
from pathlib import Path
import numpy as np
import json
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_planner import interpolate_native
from dataset_to_sim.studio import validate_trajectory


def test_interpolation_preserves_endpoints_and_obeys_speed_limits():
    start=np.array([0,0,0,0,0,5.]); end=np.array([80,-40,20,10,-15,60.])
    q,t=interpolate_native([(start,.5),(end,1.)])
    np.testing.assert_array_equal(q[0],start)
    np.testing.assert_allclose(q[-1],end)
    validate_trajectory(q,t,np.array([[-180,180]]*5+[[0,100]]))
    assert t[-1]>3


def test_open_approach_centers_cube_between_both_jaws(tmp_path,monkeypatch):
    """Regression: fixed jaw had only 2.2 mm clearance while moving jaw had 32 mm."""
    import studio_planner as planner
    from strict_action_scene import episode_scene,build_model
    from dataset_to_sim.strict_replay import NativeJointAdapter
    cfg=json.loads((planner.ROOT/'calibration/native-action-current.json').read_text())
    scene={'cube_pixel':[484.5,254.5],'cube_pixel_corners':[[497,268],[472,268],[472,241],[497,241]],'bowl_pixel':[204.115,229.118],
           'cube_pose_from_initial_rgb':{'position':[.166864,-.136652,.01295],
                                       'quaternion':[.72033934,0,0,.69362182]}}
    cal=episode_scene(cfg,scene);model=build_model(cal,cfg)
    capture={'source':'live','initial_native':[-.747253,-98.153846,96.395604,40.879121,-8.923077,.947867]}
    monkeypatch.setattr(planner,'load_scene',lambda folder:(capture,cfg,scene,cal,model))
    monkeypatch.setattr(planner,'render_preview',lambda *args:None)
    planner.plan(tmp_path)
    actions=np.load(tmp_path/'plan.npz')['actions']
    closing=np.flatnonzero((actions[1:,5]<34.99)&(actions[:-1,5]>=34.99))[0]
    data=mujoco.MjData(model);data.qpos[:6]=NativeJointAdapter(**cfg['adapter']).to_model(actions[closing]);mujoco.mj_forward(model,data)
    center=(data.geom('fixed_pad').xpos+data.geom('moving_pad').xpos)/2
    np.testing.assert_allclose(center[:2],cal.world['cube']['position'][:2],atol=.001)
    for side in ['fixed','moving']:
        jaws=[g for g in range(model.ngeom) if model.geom(g).name.startswith(side+'_jaw')]
        distance=min(mujoco.mj_geomDistance(model,data,g,model.geom('cube_geom').id,.1,None) for g in jaws)
        assert distance>.006,(side,distance)
