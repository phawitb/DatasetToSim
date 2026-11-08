"""Seeded synthetic demonstrations, driven only by simulated actuator contacts.

These are planned demonstrations, not unchanged recorded-action benchmarks and
not evidence of hardware transfer. No camera, segmentation or hardware is used.
"""
import copy
import hashlib
from functools import lru_cache
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dataset_to_sim.calibration import Calibration
from dataset_to_sim.dataset import Episode
from dataset_to_sim.strict_replay import NativeJointAdapter, integrate_actions, array_hash
from dataset_to_sim.studio import native_limits, validate_trajectory
from strict_action_scene import build_model, containment
from strict_action_replay import score
from studio_stack import stack_verdict
from studio_generation_layout import LEGACY_YAW_RANGE_DEG, PARKED_XY, sample_workspace_layout

TASKS = {
    'green_cube_to_green_bowl': ('bowl', 'green', 'Put the green cube in the green bowl.'),
    'red_cube_to_green_bowl': ('bowl', 'red', 'Put the red cube in the green bowl.'),
    'red_cube_to_green_cube': ('stack', 'red', 'Stack the red cube on the green cube.'),
    'green_cube_to_red_cube': ('stack', 'green', 'Stack the green cube on the red cube.'),
    'unstack': ('unstack', 'red', 'Lift the red cube off the green cube and place it on the empty mat.'),
}
COLORS = {'red': [.85, .045, .035, 1.], 'green': [.045, .65, .09, 1.]}
# Empirical proposal preference, not a hard reachability/safety limit: screening
# 56 probe layouts retained all 17 successes, but excludes 3 of 45 older accepted
# scenes. Existing accepted data are retained; every new scene still needs the
# unchanged physical audit. The filter is only enabled for balanced generation.
PLACEMENT_PITCH_RISK_LIMIT_DEG = 20.


def _save(path, value):
    path.write_text(json.dumps(value, indent=2))


@lru_cache(maxsize=4)
def _robot_base_bounds(config_json):
    """Conservative world XY bounds of the calibrated robot base meshes."""
    config = json.loads(config_json)
    cal = Calibration(ROOT / 'calibration/native-action-current.json', config['base_calibration'])
    model = build_model(cal, config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_id = model.body('base').id
    bounds = []
    for index in np.flatnonzero(model.geom_bodyid == base_id):
        rotation = data.geom_xmat[index].reshape(3, 3)
        center = data.geom_xpos[index] + rotation @ model.geom_aabb[index, :3]
        half = np.abs(rotation) @ model.geom_aabb[index, 3:]
        bounds.append((center - half, center + half))
    if not bounds:
        raise ValueError('Robot base geometry is required for workspace sampling')
    return (np.min([b[0][:2] for b in bounds], axis=0).tolist(),
            np.max([b[1][:2] for b in bounds], axis=0).tolist())


@lru_cache(maxsize=4)
def _layout_ik_context(config_json):
    """Cache calibrated geometry and the same recorded IK branch seed as planner."""
    from dataset_to_sim.dataset import DatasetIndex
    from episode0 import DATA
    config = json.loads(config_json)
    cal = Calibration(ROOT / 'calibration/native-action-current.json', config['base_calibration'])
    model = build_model(cal, config)
    adapter = NativeJointAdapter(**config['adapter'])
    limits = native_limits(config)
    bounds = adapter.to_model(np.array([limits[:, 0], limits[:, 1]]))[:, :5]
    lower, upper = np.min(bounds, axis=0) + 1e-6, np.max(bounds, axis=0) - 1e-6
    episode = DatasetIndex(DATA).episode(0)
    close = np.flatnonzero((episode.action_deg[1:, 5] < 5) & (episode.action_deg[:-1, 5] >= 5))[0] + 1
    reference = adapter.to_model(episode.action_deg[min(close + 3, len(episode.action_deg) - 1)])
    home = adapter.to_model(np.asarray(json.loads((ROOT / 'calibration/studio-home.json').read_text())['home']))
    return model, adapter, lower, upper, reference, home


def _stack_layout_reachability(config_json, world, source_xy, target_xy, support, technique='natural',
                               placement_pitch_limit_deg=None):
    """Screen initial layouts using bounded IK only; never simulate or alter actions.

    Heights, openings, orientation weights and tolerance mirror studio_planner.
    This does not certify collision safety or later contact-based correction.
    """
    model, adapter, lower, upper, reference, home = _layout_ik_context(config_json)
    data = mujoco.MjData(model)
    current = home.copy()
    roll_offset = reference[4] - reference[0]
    edge = world['cube']['size']
    grasp_z = max(support + edge / 2 + .0002 + .010, .024)
    hover = max(.12, grasp_z + .09, support + world['bowl']['height'] + .065)
    place_z = grasp_z + edge + .005
    stages = [('source_hover', source_xy, hover, 35.)]
    if technique == 'natural':
        stages.append(('source_pregrasp', source_xy, grasp_z + .045, 35.))
    stages += [('source_approach', source_xy, grasp_z, 35.), ('source_grasp', source_xy, grasp_z, 0.),
               ('source_lift', source_xy, hover, 0.), ('target_transfer', target_xy, hover, 0.),
               ('target_lower', target_xy, place_z, 0.), ('target_release', target_xy, place_z, 40.),
               ('target_retreat', target_xy, hover, 40.)]
    errors = []

    def center(q):
        data.qpos[:6] = q
        data.qpos[5] = max(.15, q[5])
        mujoco.mj_forward(model, data)
        return (data.geom('fixed_pad').xpos + data.geom('moving_pad').xpos) / 2

    for name, xy, z, grip in stages:
        xyz = np.r_[xy, z]
        prior = current.copy()
        opening = adapter.to_model(np.array([0, 0, 0, 0, 0, grip]))[5]

        def residual(x):
            return np.r_[center(np.r_[x, opening]) - xyz, .012 * (x[1:4].sum() - np.pi / 2),
                         .008 * (x[4] - x[0] - roll_offset), .00003 * (x - prior[:5])]

        seed = reference if name == 'source_hover' else current
        fit = least_squares(residual, np.clip(seed[:5], lower, upper), bounds=(lower, upper),
                            max_nfev=180, gtol=1e-10)
        current = np.r_[fit.x, opening]
        error = float(np.linalg.norm(center(current) - xyz))
        pitch_error = float(np.rad2deg(current[1:4].sum() - np.pi / 2))
        errors.append({'stage': name, 'error_m': error, 'target_m': xyz.tolist(),
                       'pitch_residual_deg': pitch_error})
        if not np.isfinite(error) or error > .004:
            return {'passed': False, 'failure': errors[-1], 'waypoints': errors}
        if (name == 'target_lower' and placement_pitch_limit_deg is not None
                and abs(pitch_error) > placement_pitch_limit_deg):
            return {'passed': False, 'failure': {**errors[-1], 'reason': 'placement_pitch_risk',
                    'threshold_deg': placement_pitch_limit_deg}, 'waypoints': errors}
    return {'passed': True, 'waypoints': errors}


def initialize_generation_scene(folder, task_id, seed, options=None):
    """Freeze scene and modest parameter draws before planning or simulation."""
    options = dict(options or {})
    if task_id not in TASKS:
        raise ValueError(f'Unknown synthetic task: {task_id}')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    for key in ('randomize_layout', 'physics_randomization'):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f'{key} must be boolean')
    if options.get('_layout_cell') is not None and not options.get('randomize_layout', True):
        raise ValueError('_layout_cell requires randomize_layout=True')
    task, color, instruction = TASKS[task_id]
    if task == 'unstack':
        color = options.get('unstack_top', 'red')
        if color not in COLORS:
            raise ValueError('unstack_top must be red or green')
        other = 'green' if color == 'red' else 'red'
        instruction = f'Lift the {color} cube off the {other} cube and place it on the empty mat.'
    other = 'green' if color == 'red' else 'red'
    from studio_generation_zones import validate_object_zones, layout_zone_roles, ALL_ZONES
    object_zones = validate_object_zones(options.get('object_zones', {}))
    zone_roles = layout_zone_roles(task_id, options)
    from studio_generation_zones import scene_presence
    present = scene_presence(task_id, options)
    if not options.get('randomize_layout', True) and any(z != ALL_ZONES for z in object_zones.values()):
        raise ValueError('Object zones require Randomize object layout')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'calibration/native-action-current.json'
    config = json.loads(config_path.read_text())
    rng = np.random.default_rng(int(seed))
    data = copy.deepcopy(config['base_calibration'])
    world = data['world']
    edge = world['cube']['size']
    mat = world['mat']
    support = mat['position'][2] + mat['size'][2] / 2
    z = support + edge / 2 + .0002
    if options.get('randomize_layout', True):
        config_json = json.dumps(config, sort_keys=True)
        descriptor = options.get('_layout_cell')
        pitch_limit = PLACEMENT_PITCH_RISK_LIMIT_DEG if options.get('layout_strategy') == 'balanced' else None
        maximum_proposals = 128 if pitch_limit is not None else 32
        screen = {'method': 'bounded_calibrated_robot_IK_before_physical_planning',
                  'tolerance_m': .004, 'maximum_proposals': maximum_proposals, 'proposal_count': 0,
                  'filtered_proposal_count': 0, 'rejection_counts': {}, 'filtered_proposals': [],
                  'note': 'Kinematic screen only; full planner and unchanged physical contact audit remain mandatory.'}
        if pitch_limit is not None:
            screen.update(placement_pitch_risk_limit_deg=pitch_limit,
                          placement_pitch_risk_note='Empirical conservative proposal preference for target lowering only; '
                          'not proof of unreachability or failure. Some physically valid layouts can be filtered; '
                          'previously accepted episodes remain valid and are retained.')
        for _ in range(maximum_proposals if descriptor is not None else 1):
            spread = options.get('zone_spread_mm')
            source, target, bowl_xy, draws, layout_sampling = sample_workspace_layout(
                world, rng, _robot_base_bounds(config_json), task, descriptor, zone_roles, present,
                zone_spread_m=None if spread is None else spread / 1000,
                yaw_range_deg=options.get('object_yaw_range_deg', LEGACY_YAW_RANGE_DEG))
            if descriptor is None:
                break
            screen['proposal_count'] += 1
            check = _stack_layout_reachability(config_json, world, source[0], target[0], support,
                                               options.get('motion_technique', 'natural'), pitch_limit)
            if check['passed']:
                screen['selected_proposal'] = check
                break
            screen['filtered_proposal_count'] += 1
            stage = check['failure'].get('reason', check['failure']['stage'])
            screen['rejection_counts'][stage] = screen['rejection_counts'].get(stage, 0) + 1
            screen['filtered_proposals'].append({'layout_draws': draws, **check})
        if descriptor is not None:
            _save(folder / 'generation-layout-screen.json', screen)
            if 'selected_proposal' not in screen:
                raise ValueError(f'No layout passed candidate screening in requested cells after {maximum_proposals} proposals; '
                                 f'rejections={screen["rejection_counts"]}; see generation-layout-screen.json')
            layout_sampling.update(reachability_filter='bounded_calibrated_robot_IK; planner_and_contact_audit_must_pass',
                                   kinematic_screen=screen)
    else:
        source, target, bowl_xy = ([.23, .12], 0.), ([.24, -.01], 0.), [.24, -.14]
        draws = {name: {'position_xy_m': list(xy), 'yaw_rad': yaw, 'xy_offset_m': [0., 0.]}
                 for name, (xy, yaw) in [('source_cube', source), ('other_cube', target)]}
        draws['bowl'] = {'position_xy_m': list(bowl_xy), 'xy_offset_m': [0., 0.]}
        layout_sampling = {'version': 2, 'distribution': 'fixed_reference_layout'}
    layout_sampling['object_zones'] = object_zones
    layout_sampling['zone_spread_mm'] = options.get('zone_spread_mm')
    layout_sampling['object_yaw_range_deg'] = float(options.get('object_yaw_range_deg', LEGACY_YAW_RANGE_DEG))

    def pose(xy, yaw):
        return {'position': [*map(float, xy), z], 'size': edge,
                'quaternion': [float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))]}

    world['cube'] = pose(*source)
    world['target_cube'] = pose(*target)
    world['bowl']['position'] = [*map(float, bowl_xy), support]
    world['bowl_present'] = present['bowl']
    # Requests with the rotation control may rotate cubes freely: align the jaws to the cube faces.
    world['align_grasp_yaw'] = 'object_yaw_range_deg' in options
    world['other_cube_present'] = present['other_cube'] or task == 'unstack'
    if not world['other_cube_present']:
        world['target_cube']['position'] = [*PARKED_XY, -1.]
    if task == 'unstack':
        # Reserve the independently sampled destination as empty mat space.
        # Only initial freejoint poses are assigned.
        destination = copy.deepcopy(world['target_cube'])
        world['target_cube'] = copy.deepcopy(world['cube'])
        world['cube']['position'][2] += edge + .0002
        world['placement'] = destination
    world['cube']['color'] = color
    world['target_cube']['color'] = other
    physics_draws = {}
    if options.get('physics_randomization', False):
        mass_scale = float(rng.uniform(.9, 1.1))
        config['estimates']['cube_mass_kg'] *= mass_scale
        physics_draws['cube_mass_scale'] = mass_scale
        physics_draws['friction_scales'] = {}
        for material, values in config.get('contact_materials', {}).items():
            scale = float(rng.uniform(.9, 1.1))
            config['contact_materials'][material] = (np.asarray(values) * scale).tolist()
            physics_draws['friction_scales'][material] = scale
    scene = {'schema_version': 1, 'source': 'synthetic', 'task': task, 'task_id': task_id,
             'world': world, 'layout_draws': draws, 'layout_sampling': layout_sampling, 'physics_draws': physics_draws,
             'randomization_policy': 'Seeded uniform full-mat XY proposals with footprint margins, base exclusion and object separation; cube yaw uniform within the requested rotation range (legacy ±0.10 rad); optional disk spread around zone centers. Only reachable, physically valid proposals are accepted. Optional cube mass and contact friction scale within 10%; fixed geometry, colors, gains and collision masks.',
             'calibration': data, 'base_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    if options.get('_layout_cell') is not None:
        scene['randomization_policy'] = ('Seeded uniform proposals within requested cells of the full calibrated mat; '
            'footprint margins, base exclusion and 80 mm initial cube separation. Accepted-episode cell counts are '
            'balanced by the generation worker; planner reachability and unchanged contact audits decide acceptance. '
            'Yaw within 0.10 rad; optional cube mass/contact friction within 10%; fixed geometry, colors, gains and collision masks.')
    home = json.loads((ROOT / 'calibration/studio-home.json').read_text())
    capture = {'source': 'synthetic', 'task': task, 'task_id': task_id, 'instruction': instruction,
               'seed': int(seed), 'initial_native': home['home'], 'bowl_present': world['bowl_present'],
               'other_cube_present': world['other_cube_present'],
               'calibration_hash': config['follower_calibration_sha256'], 'generation_options': options,
               'simulation_only': True, 'executed': False}
    _save(folder / 'generation-config.json', config)
    _save(folder / 'generation-scene.json', scene)
    _save(folder / 'capture.json', capture)
    return capture


def load_generation_scene(folder):
    folder = Path(folder)
    capture = json.loads((folder / 'capture.json').read_text())
    if capture.get('source') != 'synthetic':
        raise ValueError('Generation scenes must be synthetic')
    config = json.loads((folder / 'generation-config.json').read_text())
    scene = json.loads((folder / 'generation-scene.json').read_text())
    cal = Calibration(folder / 'generation-scene.json', copy.deepcopy(scene['calibration']))
    root = ET.fromstring(build_model(cal, config, return_xml=True))
    world = root.find('worldbody')
    bowl = root.find(".//body[@name='bowl']")
    if not cal.world['bowl_present']:
        world.remove(bowl)
    # Preserve the calibrated bowl material from strict_action_scene. Its sage
    # RGBA is shared with the frozen real-dataset Simulation; cube green is a
    # separate material and must not recolor the bowl or its collision proxies.
    source = root.find(".//geom[@name='cube_geom']")
    source.set('rgba', ' '.join(map(str, COLORS[cal.world['cube']['color']])))
    target = cal.world['target_cube']
    body = ET.SubElement(world, 'body', name='target_cube',
                         pos=' '.join(map(str, target['position'])), quat=' '.join(map(str, target['quaternion'])))
    ET.SubElement(body, 'freejoint', name='target_cube_free')
    geom = copy.deepcopy(source)
    geom.set('name', 'target_cube_geom')
    geom.set('rgba', ' '.join(map(str, COLORS[target['color']])))
    if not cal.world.get('other_cube_present', True):
        # Not in the scene: keep the body so traces/renderers stay uniform, but
        # park it out of view, hidden, collision-free and weightless.
        body.set('gravcomp', '1')
        geom.set('contype', '0'); geom.set('conaffinity', '0'); geom.set('group', '3')
    body.append(geom)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    return capture, config, scene, cal, model


def unstack_verdict(trace, edge, destination, support_z, support_force):
    """Require released physical placement while preserving the lower cube."""
    source = np.asarray(trace['cube_pose'])
    target = np.asarray(trace['target_pose'])
    audit = trace['audit']
    grasp = audit.get('grasp_support', {})
    rotation = Rotation.from_quat(source[-1, [4, 5, 6, 3]]).as_matrix()
    corners = np.asarray(list(itertools.product([-1, 1], repeat=3))) * edge / 2
    vertices = corners @ rotation.T + source[-1, :3]
    gap = float(vertices[:, 2].min() - support_z)
    bottom_rotation = Rotation.from_quat(target[-1, [4, 5, 6, 3]]).as_matrix()
    bottom_shift = float(np.max(np.linalg.norm(target[:, :3] - target[0, :3], axis=1)))
    destination_error = float(np.linalg.norm(source[-1, :2] - np.asarray(destination)[:2]))
    stable = all(np.max(np.linalg.norm(p[-30:, :3] - p[-1, :3], axis=1)) < .0015 for p in (source, target))
    settled = bool(stable and audit['final_cube_linear_speed_m_s'] < .01 and audit['final_cube_angular_speed_rad_s'] < .2
                   and np.linalg.norm(trace['target_velocity'][-1, :3]) < .01 and np.linalg.norm(trace['target_velocity'][-1, 3:]) < .2)
    outcome = 'success'
    if source[:, 2].max() < source[0, 2] + .04: outcome = 'not_lifted'
    elif grasp.get('longest_secure_grasp_s', 0) < .08 or grasp.get('longest_secure_lift_s', 0) < .03: outcome = 'no_bilateral_grasp'
    elif audit['final_finger_contact']: outcome = 'not_released'
    elif bottom_shift > .004 or bottom_rotation[2, 2] < np.cos(np.deg2rad(8)): outcome = 'target_moved_or_tipped'
    elif destination_error > .012 or not -.001 <= gap <= .001 or support_force < .02: outcome = 'not_placed'
    elif not settled: outcome = 'not_settled'
    return {'outcome': outcome, 'bottom_max_shift_m': bottom_shift, 'destination_error_m': destination_error,
            'support_gap_m': gap, 'support_force_N': float(support_force), 'settled': settled}


def _table_support(model, trace):
    data = mujoco.MjData(model)
    data.qpos[:6] = trace['qpos'][-1]
    for name, key in [('cube', 'cube_pose'), ('target_cube', 'target_pose')]:
        address = model.joint(name + '_free').qposadr[0]
        data.qpos[address:address+7] = trace[key][-1]
    mujoco.mj_forward(model, data)
    total = 0.
    for index, contact in enumerate(data.contact):
        names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
        if 'cube_geom' in names and names & {'work_mat', 'table'}:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            total += abs(force[0])
    return float(total)


def generate_episode(folder: Path, task_id: str, seed: int, options: dict) -> dict:
    """Plan, integrate and audit one episode; rendering is the caller's job."""
    from studio_planner import prepare_plan
    folder = Path(folder)
    options = dict(options or {})
    initialize_generation_scene(folder, task_id, seed, options)
    report = prepare_plan(folder, technique=options.get('motion_technique', 'natural'),
                          speed_scale=options.get('speed_scale', 1.), preview=False)
    capture, config, scene, cal, model = load_generation_scene(folder)
    saved = np.load(folder / 'plan.npz')
    actions, times = saved['actions'], saved['timestamps']
    adapter = NativeJointAdapter(**config['adapter'])
    validate_trajectory(actions, times, native_limits(config), speed_scale=report['speed_scale'],
                        velocity_limits=report['motion'].get('timing_reference', {}).get('limits', [None])[0])
    states = np.tile(capture['initial_native'], (len(actions), 1))
    episode = Episode(-1, np.arange(len(actions)), np.arange(len(actions)), times, states, actions, {})
    trace = integrate_actions(model, episode, adapter)
    # The integrator is shared with recorded replay. Correct its explanatory
    # provenance for generated commands; its no-state-write audit is unchanged.
    trace['audit']['control_source'] = 'synthetic planned native actions; zero-order hold'
    trace['audit']['prehistory'] = 'synthetic initial Home joint pose held in actuator history'
    trace['audit']['timing_note'] = 'Generated native commands at 30 Hz; each interval and the terminal 1/30 s hold integrated through contact physics.'
    inside = None
    if capture['task'] == 'stack':
        assessment = stack_verdict(trace, cal.world['cube']['size'])
        report['stack_assessment'] = assessment
        outcome = assessment['outcome']
    elif capture['task'] == 'unstack':
        mat = cal.world['mat']
        assessment = unstack_verdict(trace, cal.world['cube']['size'], cal.world['placement']['position'],
                                     mat['position'][2] + mat['size'][2]/2, _table_support(model, trace))
        report['unstack_assessment'] = assessment
        outcome = assessment['outcome']
    else:
        inside = containment(model, trace['qpos'][-1], trace['cube_pose'][-1], cal, config)
        outcome = score(trace, inside)
    collision = {key: value for key, value in trace['audit']['contact_audit'].items() if key != 'cube'}
    tracking = np.abs(adapter.to_native(trace['qpos'][1:]) - actions)
    checks = bool(trace['audit']['initial_max_penetration_m'] <= .001
                  and all(value['max_penetration_m'] <= .001 for value in collision.values())
                  and np.all(np.max(tracking, axis=0) <= np.array([10.]*5 + [25.])))
    report.update(outcome=outcome, transfer_checks_passed=checks, simulation_only=True, executed=False,
                  physical_model_validated=False, collision_audit=collision, audit=trace['audit'], bowl_containment=inside,
                  max_cube_height_m=float(trace['cube_pose'][:, 2].max()), max_tracking_error=np.max(tracking, axis=0).tolist(),
                  simulated_action_hash=array_hash(trace['native_action_applied']), simulated_timestamp_hash=array_hash(times),
                  simulated_speed_scale=report['speed_scale'], generation_provenance={'scene': 'generation-scene.json',
                  'config': 'generation-config.json', 'base_config_sha256': scene['base_config_sha256'],
                  'config_sha256': hashlib.sha256((folder / 'generation-config.json').read_bytes()).hexdigest(),
                  'scene_sha256': hashlib.sha256((folder / 'generation-scene.json').read_bytes()).hexdigest(),
                  'physics_draws': scene['physics_draws'], 'layout_draws': scene['layout_draws'],
                  'layout_sampling': scene['layout_sampling']})
    np.savez_compressed(folder / 'trace.npz', qpos=trace['qpos'], cube_pose=trace['cube_pose'],
                        target_pose=trace['target_pose'], target_velocity=trace['target_velocity'],
                        cube_velocity=trace['cube_velocity'], native=adapter.to_native(trace['qpos']), timestamps=trace['sample_times'])
    mujoco.mj_saveLastXML(str(folder / 'scene.xml'), model)
    _save(folder / 'plan.json', report)
    return report
