"""Validate estimated New Run scenes before manipulation, never during replay.

Only the unmeasured lean of a cylinder standing inside a container may be
initialized from passive physics. Measured object poses remain unchanged.
"""
import hashlib
import json
import copy
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

MAX_SECONDS = 6.
STABLE_SECONDS = .5
LINEAR_SPEED_M_S = .001
ANGULAR_SPEED_RAD_S = .02
MEASURED_DRIFT_M = .002
PENETRATION_M = .001


def model_fingerprint(model):
    """Include every compiled model field, including qpos0, solver and motor history."""
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def scene_fingerprint(model, config, scene):
    from pickplace import load_registry
    payload = dict(model=model_fingerprint(model), config=config, scene=scene, registry=load_registry())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _record_digest(report):
    return hashlib.sha256(json.dumps({k: v for k, v in report.items() if k != 'result_hash'},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def _apply(scene, report):
    for obj in scene['objects']:
        pose = report['accepted_poses'].get(obj['id'])
        if pose is None:
            continue
        obj['position'] = pose[:3]
        obj['initial_quaternion'] = pose[3:]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(pose[3:]))
        axis = rotation.reshape(3, 3)[:, 0]
        obj['yaw'] = float(np.arctan2(axis[1], axis[0]))
        obj['lean'] = float(np.arccos(np.clip(axis[2], -1, 1)))
    scene['initialization'] = report


def initialize_scene(model, scene, registry, arm, record_path=None):
    """Check a stationary-arm scene; return audited inferred poses or reject it.

    The inferred-position bound is half the supporting cavity radius, derived
    from its geometry. It is not an allowance in the manipulation verdict.
    All measured bodies must stay within 2 mm of their captured positions.
    """
    key_data = dict(model=model_fingerprint(model), scene=scene, registry=registry,
                    arm=np.asarray(arm).tolist(), policy=Path(__file__).read_text(),
                    limits=[MAX_SECONDS, STABLE_SECONDS, LINEAR_SPEED_M_S, ANGULAR_SPEED_RAD_S, MEASURED_DRIFT_M, PENETRATION_M])
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = Path(record_path) if record_path else None
    if path and path.exists():
        try:
            saved = json.loads(path.read_text())
            if saved['input_hash'] == key and saved['status'] == 'stable' and saved.get('result_hash') == _record_digest(saved):
                _apply(scene, saved)
                return saved
        except (ValueError, KeyError, TypeError):
            pass

    # The measured supports are observations, whereas the standing cylinder's
    # lean was guessed. Infer that missing pose with supports fixed, then require
    # the entire scene to be stable with EVERY original free joint restored.
    inferred = {o['id'] for o in scene['objects'] if o.get('lean')
                and registry['objects'][o['kind']]['shape']['type'] == 'cylinder'}
    try:
        if inferred:
            with tempfile.TemporaryDirectory(prefix='pickplace-init-') as directory:
                filename = Path(directory) / 'scene.xml'
                mujoco.mj_saveLastXML(str(filename), model)
                root = ET.parse(filename).getroot()
                for obj in scene['objects']:
                    if obj['id'] in inferred:
                        continue
                    body = root.find(f".//body[@name='{obj['body']}']")
                    for joint in list(body):
                        if joint.tag == 'freejoint' or joint.tag == 'joint' and joint.get('type') == 'free':
                            body.remove(joint)
                fixed_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
            report = _relax(fixed_model, scene, registry, arm, path, infer=True)
            validation_model = copy.copy(model)
            by_id = {o['id']: o for o in scene['objects']}
            for identifier, pose in report['accepted_poses'].items():
                body = validation_model.body(by_id[identifier]['body']).id
                address = validation_model.jnt_qposadr[validation_model.body_jntadr[body]]
                validation_model.qpos0[address:address + 7] = pose
            validation = _relax(validation_model, scene, registry, arm, path, infer=False)
            report['free_scene_validation'] = validation
        else:
            report = _relax(model, scene, registry, arm, path, infer=False)
        report['input_hash'] = key
        report['method'] = 'Infer missing contained-cylinder poses with measured supports fixed; validate with all original free joints restored'
        report['result_hash'] = _record_digest(report)
        if path:
            path.write_text(json.dumps(report, indent=2, allow_nan=False))
        _apply(scene, report)
        return report
    except ValueError:
        # _relax saves its rejection evidence before raising; no pose is applied.
        raise


def _relax(model, scene, registry, arm, path, infer):
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    data.qpos[model.jnt_qposadr[joints]] = arm
    data.ctrl[:] = arm
    for actuator, count in enumerate(model.actuator_history[:, 0]):
        if count:
            mujoco.mj_initCtrlHistory(model, data, actuator, None, np.full((int(count), 1), arm[actuator]))
    mujoco.mj_forward(model, data)
    objects = {o['id']: o for o in scene['objects']}
    bodies = {int(model.body(o['body']).id): o['id'] for o in scene['objects']}
    dynamic = {}
    report = dict(status='checking', phase='inference' if infer else 'free_scene_validation', duration_s=0., stability_window_s=STABLE_SECONDS,
                  accepted_poses={}, objects={}, contact_pairs=[], max_penetration_m=0.,
                  method='Passive physics before planning; only inferred contained-cylinder poses are corrected')
    for body, identifier in bodies.items():
        joint = int(model.body_jntadr[body])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        obj = objects[identifier]
        support = objects.get(obj.get('support'))
        shape = registry['objects'][obj['kind']]['shape']
        inferred = bool(infer and obj.get('lean') and shape['type'] == 'cylinder' and support is not None)
        bound = MEASURED_DRIFT_M
        if inferred:
            from pickplace import surface, surface_radius_at_rim
            inferred = surface(registry, support, obj.get('support_surface'))['type'] == 'inside'
            if inferred:
                bound = surface_radius_at_rim(registry, support) / 2
        address, velocity = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        dynamic[identifier] = (body, address, velocity, inferred, bound)
        report['objects'][identifier] = dict(inferred=inferred, correction_bound_m=bound,
            initial_pose=data.qpos[address:address + 7].tolist(), max_displacement_m=0.)

    contacts = set()

    def reject(reason):
        report.update(status='rejected', reason=reason, duration_s=float(data.time), contact_pairs=sorted(contacts))
        if path:
            path.write_text(json.dumps(report, indent=2, allow_nan=False))
        raise ValueError(f'Initial scene is inconsistent: {reason}. Capture and check the object supports again')

    def contact_check():
        support_contacts = set()
        for contact in data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            penetration = max(0., -float(contact.dist))
            report['max_penetration_m'] = max(report['max_penetration_m'], penetration)
            if penetration > PENETRATION_M:
                reject(f'{model.geom(g1).name} overlaps {model.geom(g2).name} by {penetration * 1000:.1f} mm')
            if contact.dist > 0:
                continue
            bits = int(model.geom_contype[g1]) | int(model.geom_contype[g2])
            if bits & (2 | 8):
                reject(f'robot contact between {model.geom(g1).name} and {model.geom(g2).name}')
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            id1, id2 = bodies.get(b1), bodies.get(b2)
            if id1 is not None or id2 is not None:
                contacts.add(tuple(sorted((model.geom(g1).name, model.geom(g2).name))))
            if id1 is not None and id2 is not None:
                if objects[id1].get('support') == id2:
                    support_contacts.add(id1)
                elif objects[id2].get('support') == id1:
                    support_contacts.add(id2)
                else:
                    reject(f'{id1} touches {id2} outside its declared support')
            for identifier, other in ((id1, id2), (id2, id1)):
                if identifier in dynamic and dynamic[identifier][3] and other is None:
                    reject(f'{identifier} fell out of its declared container')
        return support_contacts

    contact_check()
    stable_for = 0.
    while data.time < MAX_SECONDS:
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reject('physics produced a non-finite state')
        supported = contact_check()
        stable = True
        for identifier, (body, address, velocity, inferred, bound) in dynamic.items():
            item = report['objects'][identifier]
            displacement = float(np.linalg.norm(data.qpos[address:address + 3] - item['initial_pose'][:3]))
            item['max_displacement_m'] = max(item['max_displacement_m'], displacement)
            if displacement > bound:
                reject(f'{identifier} drifts {displacement * 1000:.1f} mm (initial-pose bound {bound * 1000:.1f} mm)')
            if not inferred:
                angle = 2 * np.arccos(np.clip(abs(np.dot(data.qpos[address + 3:address + 7], item['initial_pose'][3:])), 0., 1.))
                if angle > np.deg2rad(2):
                    reject(f'measured object {identifier} rotates before the task')
            speed = float(np.linalg.norm(data.qvel[velocity:velocity + 3]))
            spin = float(np.linalg.norm(data.qvel[velocity + 3:velocity + 6]))
            item.update(final_pose=data.qpos[address:address + 7].tolist(), linear_speed_m_s=speed, angular_speed_rad_s=spin)
            stable &= speed < LINEAR_SPEED_M_S and spin < ANGULAR_SPEED_RAD_S
            if objects[identifier].get('support', 'mat') != 'mat':
                stable &= identifier in supported
        stable_for = stable_for + model.opt.timestep if stable else 0.
        if data.time >= 1. and stable_for >= STABLE_SECONDS:
            break
    else:
        reject(f'objects do not settle on their declared support within {MAX_SECONDS:g} seconds')

    for identifier, (body, address, velocity, inferred, bound) in dynamic.items():
        if not inferred:
            continue
        # Contact alone is insufficient: ensure the inferred cylinder remains inside
        # the supporting cavity, not resting on its outside wall.
        from pickplace import _vertices, bottom_offset
        obj = objects[identifier]
        support = objects[obj['support']]
        shape = registry['objects'][support['kind']]['shape']
        frame = data.body(support['body'])
        local = (_vertices(model, data, body) - frame.xpos) @ frame.xmat.reshape(3, 3)
        local[:, 2] += bottom_offset(shape)
        below = local[:, 2] <= shape['height_m']
        allowed = shape['bottom_outer_radius_m'] - shape['wall_m'] + (shape['outer_radius_m'] - shape['bottom_outer_radius_m']) * local[:, 2] / shape['height_m']
        if not np.any(below) or np.any(np.linalg.norm(local[below, :2], axis=1) > allowed[below] + .001) or local[:, 2].min() < shape['floor_m'] - .001:
            reject(f'{identifier} is outside its declared container')
        report['accepted_poses'][identifier] = data.qpos[address:address + 7].tolist()
    report.update(status='stable', duration_s=float(data.time), contact_pairs=sorted(contacts))
    return report
