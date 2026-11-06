import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from episode0_physics import simulate,physical_model,clear_table_targets,Calibration,DatasetIndex,ROOT,DATA
import mujoco

def test_cube_lifts_and_releases_in_bowl_only_when_finger_contacts_enabled():
    cal=Calibration.load(ROOT/'calibration/episode0.json');ep=DatasetIndex(DATA).episode(0)
    actual,m,joints,poses=simulate(cal,ep,[.175,-.045])
    assert actual['max_cube_z']>.10
    assert actual['bilateral_frames']>20
    assert actual['final_inside_bowl']
    assert not actual['final_bilateral_contact']
    assert np.isfinite(poses).all() and np.isfinite(joints).all()
    assert m.neq==0
    assert actual['contact_audit']['cube']['max_penetration_m']<.001
    assert actual['collision_configuration']['arm_self']
    assert actual['collision_configuration']['finger_table']
    assert m.nexclude==0
    disabled,*_=simulate(cal,ep,[.175,-.045],disable_fingers=True)
    assert disabled['max_cube_z']<.018
    assert not disabled['final_inside_bowl']

def test_self_collision_is_detected_for_nonadjacent_links():
    cal=Calibration.load(ROOT/'calibration/episode0.json');m=physical_model(cal);d=mujoco.MjData(m)
    d.qpos[:6]=[1.7012113897,.03954061965,1.6097037253,-1.3900005594,.648267576,.548269362]
    mujoco.mj_forward(m,d)
    contacts=[c for c in d.contact if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}<={2,8} and c.dist<0]
    assert contacts
    assert any({m.body(m.geom_bodyid[c.geom1]).name,m.body(m.geom_bodyid[c.geom2]).name}=={'shoulder','lower_arm'} for c in contacts)

def test_finger_table_collision_exists_and_target_correction_clears_it():
    cal=Calibration.load(ROOT/'calibration/episode0.json');ep=DatasetIndex(DATA).episode(0)
    m=physical_model(cal);d=mujoco.MjData(m)
    q=cal.joint_convention.to_radians(ep.state_deg[0]);q[5]=np.deg2rad(-8)
    d.qpos[:6]=q;mujoco.mj_forward(m,d)
    def table_contacts():
        return [c for c in d.contact if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}=={2,4} and {m.geom(c.geom1).name,m.geom(c.geom2).name}&{'table','work_mat'} and c.dist<0]
    assert table_contacts()
    corrected,delta=clear_table_targets(m,q[None]);assert delta[0]>0
    d.qpos[:6]=corrected[0];mujoco.mj_forward(m,d)
    assert not table_contacts()
