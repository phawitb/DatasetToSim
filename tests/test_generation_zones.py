"""Object identities, allowed regions and accepted quotas stay consistent."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation import validate_request
from studio_generation_scene import ROOT, initialize_generation_scene
from studio_generation_layout import workspace_grid_cells
from studio_generation_balance import BalancedLayouts

ALL=list(range(9))

def zone(world, xy):
    cells=workspace_grid_cells(world)
    return next(c['cell_id'] for c in cells if np.all(np.asarray(xy)>=c['bounds_xy_m'][0]) and np.all(np.asarray(xy)<c['bounds_xy_m'][1]))


def test_defaults_and_partial_zone_request_are_normalized():
    assert validate_request({})['object_zones']=={k:ALL for k in ['red_cube','green_cube','tape','bowl','unstack_destination']}
    assert validate_request({'object_zones':{'red_cube':[8,0,4]}})['object_zones']['red_cube']==[0,4,8]

@pytest.mark.parametrize('zones',[[],{'unknown':[1]},{'red_cube':[]},{'red_cube':[True]}, {'red_cube':[9]}, {'bowl':['2']}, {'red_cube':[1,1]}])
def test_invalid_zone_constraints_are_rejected(zones):
    with pytest.raises(ValueError,match='zone|Zone'):
        validate_request({'object_zones':zones})


def test_fixed_layout_and_impossible_unstack_are_explicit_errors():
    with pytest.raises(ValueError,match='Randomize'):
        validate_request({'randomize_layout':False,'object_zones':{'red_cube':[1]}})
    with pytest.raises(ValueError,match='Unstack.*shared'):
        validate_request({'tasks':['unstack'],'object_zones':{'red_cube':[0], 'green_cube':[8]}})
    validate_request({'tasks':['red_cube_to_green_cube'],'object_zones':{'red_cube':[0], 'green_cube':[8]}})

@pytest.mark.parametrize('task,top',[(t,'red') for t in ['red_cube_to_green_cube','green_cube_to_red_cube','red_cube_to_green_bowl','green_cube_to_green_bowl','unstack']]+[('unstack','green')])
def test_saved_scene_respects_object_colors_and_unstack_intersection(tmp_path,task,top):
    zones={'red_cube':[0,1,3,4], 'green_cube':[4,5,7,8], 'bowl':[2,5], 'unstack_destination':[0,2]}
    for seed in range(6):
        folder=tmp_path/str(seed)
        options=validate_request({'tasks':[task],'unstack_top':top,'object_zones':zones})
        initialize_generation_scene(folder,task,seed,options)
        scene=json.loads((folder/'generation-scene.json').read_text());world=scene['world']
        # By default bowl tasks hold only the task cube; the other one is parked out of the scene.
        assert world['other_cube_present']==(not task.endswith('bowl'))
        for key in ['cube','target_cube'] if world['other_cube_present'] else ['cube']:
            body=world[key]
            assert zone(world,body['position'][:2]) in zones[body['color']+'_cube']
        if task.endswith('bowl'):assert zone(world,world['bowl']['position'][:2]) in zones['bowl']
        if task=='unstack':
            assert zone(world,world['cube']['position'][:2])==4
            assert zone(world,world['placement']['position'][:2]) in zones['unstack_destination']
        assert scene['layout_sampling']['object_zones']==options['object_zones']


def test_balancer_restricts_choices_and_accounts_for_inherited_outside_zones():
    world=json.loads((ROOT/'calibration/native-action-current.json').read_text())['base_calibration']['world']
    b=BalancedLayouts(world,seed=42,target=20,source_cells=[0,1],target_cells=[7,8])
    b.accept_cells(4,4);b.accept_cells(0,8);b.accept_cells(0,8)
    b.prepare_additions(20)
    for _ in range(20):
        choice=b.choose()
        assert choice['source_cell'] in [0,1] and choice['target_cell'] in [7,8]
        b.accept_cells(choice['source_cell'],choice['target_cell'])
    assert b.source[4]==b.target[4]==1
    assert abs(b.source[0]-b.source[1])<=1 and abs(b.target[7]-b.target[8])<=1
    with pytest.raises(ValueError,match='full'):b.choose()


def test_zone_spread_keeps_objects_near_zone_centers_and_rotation_covers_full_range(tmp_path):
    from studio_generation_layout import workspace_grid_cells
    yaws, bowl_yaws = [], []
    for seed in range(12):
        folder = tmp_path / str(seed)
        options = validate_request({'tasks': ['red_cube_to_green_bowl'], 'zone_spread_mm': {'red_cube': 20}, 'object_yaw_range_deg': {'red_cube': 360, 'green_cube': 0}})
        initialize_generation_scene(folder, 'red_cube_to_green_bowl', seed, options)
        scene = json.loads((folder / 'generation-scene.json').read_text()); world = scene['world']
        cells = workspace_grid_cells(world)
        xy = np.asarray(world['cube']['position'][:2])
        nearest = min(np.linalg.norm(xy - np.mean(np.asarray(c['bounds_xy_m']), axis=0)) for c in cells)
        assert nearest <= .020 + 1e-9
        q = world['cube']['quaternion']; yaws.append(np.degrees(2 * np.arctan2(q[3], q[0])))
        assert world['align_grasp_yaw'] is True
        assert scene['layout_sampling']['zone_spread_mm']['red_cube'] == 20 and scene['layout_sampling']['zone_spread_mm']['bowl'] is None
        assert scene['layout_sampling']['object_yaw_range_deg'] == {'source_cube': 360, 'other_cube': 0, 'bowl': 360}
        b = world['bowl']['quaternion']; bowl_yaws.append(np.degrees(2 * np.arctan2(b[3], b[0])))
    assert max(yaws) - min(yaws) > 90  # far beyond the legacy ±5.7°
    assert max(bowl_yaws) - min(bowl_yaws) > 90  # the bowl rotates too
    with pytest.raises(ValueError, match='zone_spread_mm'):
        validate_request({'zone_spread_mm': 150})
    with pytest.raises(ValueError, match='object_yaw_range_deg'):
        validate_request({'object_yaw_range_deg': 400})
    defaults = validate_request({})
    assert all(v is None for v in defaults['zone_spread_mm'].values())
    assert defaults['object_yaw_range_deg'] == {'red_cube': 360, 'green_cube': 360, 'bowl': 360}
    # One value still applies to every object.
    assert validate_request({'zone_spread_mm': 15})['zone_spread_mm']['bowl'] == 15
