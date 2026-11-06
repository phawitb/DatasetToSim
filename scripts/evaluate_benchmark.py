"""Apply one physical containment/release/settling criterion to saved runs."""
import copy
import hashlib
from collections import Counter
from benchmark72 import *

def evaluate(tag):
    folder=BATCH/tag;source=folder/'results.json';report=json.loads(source.read_text())
    scenes=json.loads((folder/'scenes.json' if (folder/'scenes.json').exists() else BATCH/'scenes.json').read_text())
    base=Calibration.load(ROOT/'calibration/episode0.json')
    for row in report['results']:
        if row['outcome']=='error':continue
        i=row['episode'];epdir=folder/f'episode-{i:02d}'
        if (epdir/'scene.xml').exists():
            m=mujoco.MjModel.from_xml_path(str(epdir/'scene.xml'));trace=np.load(epdir/'physics_trace.npz')
        else:
            # Archived baseline keeps its exact observed initial positions.
            cal=copy.deepcopy(base);s=scenes[i]
            cal.data['world']['cube']['position']=[*s['cube_xy'],.0162]
            cal.data['world']['bowl']['position']=[*s['bowl_xy'],.005]
            m=physical_model(cal);trace=np.load(folder/f'episode-{i:02d}.npz')
        d=mujoco.MjData(m);addr=m.jnt_qposadr[m.joint('cube_free').id]
        d.qpos[:6]=trace['qpos'][-1];d.qpos[addr:addr+7]=trace['cube_pose'][-1];mujoco.mj_forward(m,d)
        row['previous_outcome']=row['outcome']
        row['bowl_containment']=bowl_containment(m,d);row['final_inside_bowl']=row['bowl_containment']['inside']
        row['final_finger_contact']=bool(any(m.body('cube').id in [m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]] and 2 in [m.geom_contype[c.geom1],m.geom_contype[c.geom2]] for c in d.contact))
        poses=trace['cube_pose'];row['final_cube_motion_last_10_frames_m']=float(np.linalg.norm(poses[-10:,:3]-poses[-1,:3],axis=1).max())
        row['outcome']=verdict(row)
    counts=Counter(r['outcome'] for r in report['results'])
    report.update(successes=counts['success'],runtime_errors=counts['error'],outcome_counts=dict(counts),evaluated_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scoring_version='wall-halfspaces-physical-rim-v2',success_criterion={'lift_center_z_above_m':.055,'bilateral_intervals_at_least':2,'containment':'all 8 cube corners inside the 32 inner wall planes, above floor and below rim derived from physical wall geometry','containment_tolerance_m':.0005,'release':'no finger contact at final state','settled':'cube translation within 3 mm over last 10 frames'})
    if tag=='baseline':report['baseline_provenance_note']='Archived baseline lacks saved XML/settings; scoring reconstructs the original scene positions with the current physical model geometry. Original dynamics were not rerun.'
    (folder/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(tag,dict(counts),flush=True)
    return report

if __name__=='__main__':
    import sys
    for tag in sys.argv[1:]:evaluate(tag)
