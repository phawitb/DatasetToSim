"""Persisted local simulation jobs. No hardware interface and no Hub upload."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from studio_generation_scene import TASKS


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def trajectory_group(scene, action_hash):
    physical = dict(scene['world'])
    for key in ('cube', 'target_cube'):
        physical[key] = {k: v for k, v in physical[key].items() if k != 'color'}
    return hashlib.sha256(json.dumps({'world': physical, 'physics': scene['physics_draws']}, sort_keys=True).encode() + action_hash.encode()).hexdigest()


def validate_request(body):
    from studio_generation_visuals import VISUAL_DEFAULTS, validate_visual_options
    defaults = dict(name='synthetic-demonstrations', tasks=list(TASKS), episodes_per_task=1, object_zones={},
                    seed=0, motion_technique='natural', timing_profile='recorded', speed_scale=1., randomize_layout=True,
                    visual_randomization=True, camera_jitter=False, sensor_augmentation=True,
                    physics_randomization=False, real_background=False, real_image_processing=False, validation_fraction=.2, unstack_top='red', parent_job_id=None, target_counts=None, layout_strategy="uniform") | VISUAL_DEFAULTS
    if not isinstance(body, dict) or set(body) - set(defaults):
        raise ValueError('Unknown generation options')
    result = defaults | body
    if result['parent_job_id'] is not None and (not isinstance(result['parent_job_id'], str) or not re.fullmatch('[a-f0-9]{32}', result['parent_job_id'])):
        raise ValueError('Invalid parent generation identifier')
    if not isinstance(result['name'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', result['name']):
        raise ValueError('Use 1–64 letters, numbers, hyphens or underscores for the dataset name')
    tasks = result['tasks']
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) or t not in TASKS for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('Select one or more distinct supported tasks')
    for key, low, high in (('episodes_per_task', 1, 1000), ('seed', 0, 2**31-1)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'{key} must be an integer between {low} and {high}')
    if result['layout_strategy'] not in ('uniform', 'balanced'):
        raise ValueError('Choose uniform or balanced layout sampling')
    if result['layout_strategy'] == 'balanced' and result['randomize_layout'] is False:
        raise ValueError('Balanced workspace zones require Randomize object layout')
    targets = result['target_counts']
    if targets is not None and (not isinstance(targets, dict) or not set(targets) <= set(TASKS) or (('tasks' in body or result['parent_job_id'] is None) and set(targets) != set(tasks)) or any(type(v) is not int or not 1 <= v <= 1000 for v in targets.values())):
        raise ValueError('target_counts must give a total from 1 to 1000 for every selected task')
    for key in ('randomize_layout', 'visual_randomization', 'camera_jitter', 'sensor_augmentation', 'physics_randomization', 'real_background', 'real_image_processing'):
        if type(result[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    for key, low, high in (('speed_scale', .5, 1.5), ('validation_fraction', 0, .4)):
        if type(result[key]) not in (float, int) or not math.isfinite(result[key]) or not low <= result[key] <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
    if result['timing_profile'] not in ('recorded', 'conservative'):
        raise ValueError('Choose recorded or conservative generation timing')
    from studio_motion import validate_technique
    validate_technique(result['motion_technique'])
    if result['unstack_top'] not in ('red', 'green'):
        raise ValueError('Unstack top must be red or green')
    from studio_generation_zones import validate_object_zones, layout_zone_roles, ALL_ZONES
    result['object_zones'] = validate_object_zones(result['object_zones'])
    if not result['randomize_layout'] and any(z != ALL_ZONES for z in result['object_zones'].values()):
        raise ValueError('Object zones require Randomize object layout')
    for task in tasks if 'tasks' in body or not result['parent_job_id'] else []:
        layout_zone_roles(task, result)
    return validate_visual_options(result)


def seed_highwater(job):
    """Exclusive consumed seed bound, including interrupted or rejected attempts."""
    start = job.get('options', {}).get('seed', 0)
    return max(start + job.get('progress', {}).get('attempts', 0),
               job.get('lineage', {}).get('seed_highwater', 0),
               max((item.get('seed', -1) + 1 for item in job.get('episodes', [])), default=0))


def episode_urls(identifier, index):
    base = f'/media/generated/{identifier}/episodes/{index:05d}'
    return dict(preview_url=base + '/top.jpg', video_urls={view: base + f'/{view}.mp4' for view in ('top', 'wrist')})


class GenerationManager:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.process = None; self.active = None; self.lock = threading.RLock()

    def recover_interrupted(self):
        """Run on server startup, never as an import-time filesystem mutation."""
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text())
                if job['status'] in ('queued', 'running'):
                    job.update(status='cancelled', phase='Interrupted by server restart')
                    job.setdefault('errors', []).append('Server restarted before this batch completed')
                    save(path, job)
            except (OSError, ValueError, KeyError):
                pass

    def folder(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch('[a-f0-9]{32}', identifier):
            raise ValueError('Invalid generation identifier')
        folder = self.root / identifier
        if not (folder / 'job.json').is_file():
            raise ValueError('Generation job not found')
        return folder

    def get(self, identifier):
        with self.lock:
            job = json.loads((self.folder(identifier) / 'job.json').read_text())
            if identifier == self.active and self.process is not None and self.process.poll() is not None:
                job = json.loads((self.folder(identifier) / 'job.json').read_text())
                if job['status'] in ('queued', 'running'):
                    job.update(status='failed', phase='Worker exited unexpectedly')
                    job['errors'].append('Inspect worker.log for diagnostic details')
                    save(self.folder(identifier) / 'job.json', job)
            accepted = [item for item in job.get('episodes', []) if item.get('status') == 'accepted']
            total_frames = job.get('export', {}).get('total_frames')
            if total_frames is None:
                total_frames = 0
                for index, item in enumerate(job.get('episodes', [])):
                    if item.get('status') != 'accepted': continue
                    frames = item.get('frames')
                    if frames is None:
                        try: frames = json.loads((self.folder(identifier) / 'episodes' / f'{index:05d}' / 'plan.json').read_text())['frames']
                        except (OSError, ValueError, KeyError): frames = 0
                    total_frames += frames
            job['summary'] = {'fps': 30, 'cameras': ['top', 'wrist'], 'resolution': [640, 480],
                              'total_frames': total_frames, 'total_duration_s': total_frames / 30,
                              'accepted_episodes': len(accepted), 'format': 'LeRobot v3',
                              'timing_profile': job.get('options', {}).get('timing_profile', 'conservative')}
            job['next_seed'] = self.next_seed(job)
            return job

    def next_seed(self, job):
        root_id = job.get('lineage', {}).get('root_job_id', job['id'])
        highwater = seed_highwater(job)
        # Failed sibling extensions consume seeds too. Read their journals without
        # ever modifying a completed parent's artifacts or metadata.
        for path in self.root.glob('*/job.json'):
            try:
                other = json.loads(path.read_text())
                if other.get('lineage', {}).get('root_job_id', other.get('id')) == root_id:
                    highwater = max(highwater, seed_highwater(other))
            except (OSError, ValueError, TypeError):
                continue
        return highwater

    def parent_artifacts(self, identifier):
        """Check accepted originals against the successful portable export."""
        from studio_generation_export import _sha
        folder = self.folder(identifier)
        try:
            job = json.loads((folder / 'job.json').read_text())
            if job['status'] != 'completed' or job.get('export', {}).get('valid') is not True:
                raise ValueError('Parent must be a completed valid dataset')
            manifest = json.loads((folder / 'dataset/export_manifest.json').read_text())
            audit = json.loads((folder / 'dataset/export_audit.json').read_text())
            records = {str(Path(item['source']).resolve()): item for item in manifest['episodes']}
            accepted = [(index, item) for index, item in enumerate(job['episodes']) if item.get('status') == 'accepted']
            if not accepted or len(records) != len(accepted) or audit.get('valid') is not True or audit.get('total_episodes') != len(accepted):
                raise ValueError('Parent episode count or export audit is invalid')
            splits, sources = {}, []
            for index, item in accepted:
                source = folder / 'episodes' / f'{index:05d}'
                record = records[str(source.resolve())]
                meta = json.loads((source / 'episode.json').read_text())
                group, split = record['group_id'], record['split']
                if (split not in ('train', 'val') or group in splits and splits[group] != split
                        or meta.get('group_id') != group or meta.get('action_hash') != record['action_hash']
                        or meta.get('outcome') != 'success' or meta.get('transfer_checks_passed') is not True):
                    raise ValueError('Parent episode validation or group split is invalid')
                for file, expected in [('plan.npz', record['source_plan_sha256']), ('trace.npz', record['source_trace_sha256'])] + [(f'{view}.mp4', record['video_sha256'][view]) for view in ('top', 'wrist')]:
                    if _sha(source / file) != expected:
                        raise ValueError('Parent episode artifact integrity check failed')
                for file, provenance in record.get('provenance_files', {}).items():
                    if Path(file).name != file or _sha(source / file) != provenance['sha256']:
                        raise ValueError('Parent episode provenance integrity check failed')
                splits[group] = split
                sources.append((source, item))
            return job, sources, splits
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Parent dataset is missing valid export artifacts') from exc

    def list(self):
        result = []
        for path in sorted(self.root.glob('*/job.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try: result.append(self.get(path.parent.name))
            except (ValueError, OSError): pass
        return result

    def busy(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def resolve_options(self, body, parent=None):
        """Resolve editable parent defaults for generation and appearance previews."""
        validated = validate_request(body)
        if not body.get('parent_job_id'):
            return validated
        parent = parent if parent is not None else self.get(body['parent_job_id'])
        if parent['status'] != 'completed' or parent.get('export', {}).get('valid') is not True:
            raise ValueError('Parent must be a completed valid dataset')
        public_fields = validate_request({})
        inherited = {key: value for key, value in parent['options'].items() if key in public_fields}
        inherited.setdefault('timing_profile', 'conservative')
        # Add episodes is additive unless this request explicitly supplies totals.
        inherited['target_counts'] = None
        next_seed = self.next_seed(parent)
        options = validate_request(inherited | {'seed': next_seed} | body)
        if options['seed'] < next_seed:
            raise ValueError(f'Continuation seed must be at least {next_seed}; earlier seeds were already consumed')
        return options

    def start(self, body):
        # Validate caller keys before merging trusted inherited defaults.
        validate_request(body)
        with self.lock:
            if self.busy(): raise ValueError('A generation job is already running')
            parent, sources, splits = None, [], {}
            if body.get('parent_job_id'):
                parent, sources, splits = self.parent_artifacts(body['parent_job_id'])
            options = self.resolve_options(body, parent)
            from studio_generation_balance import task_requests
            requests = task_requests(options, [item for _, item in sources])
            identifier = uuid.uuid4().hex
            folder = self.root / identifier; folder.mkdir()
            try:
                from studio_generation_assets import snapshot_background
                background_asset = snapshot_background(options['background_id'], folder / 'assets')
                inherited_episodes = []
                for index, (source, item) in enumerate(sources):
                    shutil.copytree(source, folder / 'episodes' / f'{index:05d}')
                    inherited_episodes.append(item | episode_urls(identifier, index) | {'inherited_from_job_id': parent['id']})
            except Exception:
                shutil.rmtree(folder)
                raise
            inherited_count = len(inherited_episodes)
            lineage = dict(parent_job_id=parent['id'] if parent else None,
                           root_job_id=parent.get('lineage', {}).get('root_job_id', parent['id']) if parent else identifier,
                           inherited_count=inherited_count, seed_highwater=options['seed'])
            job = dict(id=identifier, name=options['name'], status='queued', phase='Preparing simulation',
                       created_at=time.time(), options=options, background_asset=background_asset, lineage=lineage,
                       parent_job_id=lineage['parent_job_id'], inherited_group_splits=splits,
                       progress=dict(attempts=0, accepted=inherited_count, inherited_count=inherited_count, rejected=0,
                       requested=inherited_count + sum(requests.values()), task_id=None),
                       episodes=inherited_episodes, errors=[])
            save(folder / 'job.json', job)
            try:
                with (folder / 'worker.log').open('w') as log:
                    self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', str(folder)], stdout=log, stderr=log, start_new_session=True, env=os.environ | {'STUDIO_GENERATION_PARENT': str(os.getpid())})
            except OSError as exc:
                job.update(status='failed', phase='Unable to start generation worker')
                job['errors'].append(str(exc)); save(folder / 'job.json', job)
                raise ValueError('Unable to start generation worker') from exc
            self.active = identifier
            return job

    def cancel(self, identifier):
        with self.lock:
            job = self.get(identifier)
            if identifier == self.active and self.busy():
                from studio_runtime import terminate_group
                terminate_group(self.process)
                job = json.loads((self.folder(identifier) / 'job.json').read_text())
            if job['status'] in ('queued', 'running'):
                job.update(status='cancelled', phase='Cancelled; incomplete batch is not exported')
                save(self.folder(identifier) / 'job.json', job)
            return job

    def close(self):
        if self.active and self.busy(): self.cancel(self.active)


def work(folder):
    # Own session permits cancellation of renderer/ffmpeg together. A crashed
    # parent must not leave a generation worker running after Studio restarts.
    parent = int(os.environ.get('STUDIO_GENERATION_PARENT', os.getppid()))
    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return
    threading.Thread(target=watch_parent, daemon=True).start()
    from studio_generation_scene import generate_episode
    from studio_generation_render import render_episode
    from studio_generation_export import export_dataset
    import shutil
    import traceback
    folder = Path(folder)
    job = json.loads((folder / 'job.json').read_text())
    options = job['options']
    render_options = options | {'_background_asset': job.get('background_asset')}
    from studio_generation_balance import task_requests, BalancedLayouts
    requests = task_requests(options, job['episodes'])
    accepted = [folder / 'episodes' / f'{index:05d}' for index, item in enumerate(job['episodes']) if item.get('status') == 'accepted']
    def update(phase):
        job.update(status='running', phase=phase)
        save(folder / 'job.json', job)
    try:
        update('Planning physical demonstrations')
        for task_id in options['tasks']:
            requested = requests[task_id]
            if not requested: continue
            balancer = None
            if options.get('layout_strategy') == 'balanced' and options['randomize_layout'] and task_id in ('red_cube_to_green_cube', 'green_cube_to_red_cube'):
                world = json.loads((ROOT / 'calibration/native-action-current.json').read_text())['base_calibration']['world']
                inherited = [path for path, item in zip([folder/'episodes'/f'{i:05d}' for i in range(len(job['episodes']))], job['episodes']) if item.get('status') == 'accepted' and item['task_id'] == task_id]
                from studio_generation_zones import layout_zone_roles
                zones = layout_zone_roles(task_id, options)
                balancer = BalancedLayouts(world, options['seed'], len(inherited) + requested,
                                           source_cells=zones['source_cube'], target_cells=zones['other_cube'])
                for path in inherited: balancer.accept_world(json.loads((path/'generation-scene.json').read_text())['world'])
                balancer.prepare_additions(requested)
            count = 0
            for attempt in range(max(30, requested * 30) if options['randomize_layout'] else requested * 4):
                if count >= requested: break
                choice = balancer.choose() if balancer else None
                episode_options = options | ({'_layout_cell': choice} if choice else {})
                if balancer: job.setdefault('coverage', {})[task_id] = balancer.summary()
                serial = len(job['episodes'])
                attempt_index = job['progress']['attempts']
                seed = options['seed'] + attempt_index
                episode = folder / 'episodes' / f'{serial:05d}'
                job['progress'].update(task_id=task_id, attempts=attempt_index + 1)
                job.setdefault('lineage', {})['seed_highwater'] = seed + 1
                instruction = TASKS[task_id][2]
                if task_id == 'unstack' and options.get('unstack_top') == 'green':
                    instruction = 'Lift the green cube off the red cube and place it on the empty mat.'
                update(f'Planning and validating · {instruction}')
                item = dict(task_id=task_id, seed=seed, status='rejected', outcome='error', reason='')
                try:
                    report = generate_episode(episode, task_id, seed, episode_options)
                    item['outcome'] = report['outcome']
                    if report['outcome'] != 'success' or not report['transfer_checks_passed']:
                        raise ValueError('Physical task or collision/tracking checks failed')
                except Exception as exc:
                    item['reason'] = str(exc)
                    if balancer: balancer.reject(choice)
                    job['progress']['rejected'] += 1
                    job['episodes'].append(item)
                    update('Rejected attempt; trying another layout')
                    continue
                update('Rendering synchronized Top and Wrist observations')
                try:
                    render_episode(episode, seed, render_options, lambda i, n: update(f'Rendering Top + Wrist · {i}/{n} frames'))
                except ValueError as exc:
                    # An occluded layout is a rejected observation, while broken
                    # assets/configuration and renderer/system errors stay fatal.
                    if str(exc) != 'A task cube is never visible in the rendered reference views':
                        raise
                    item.update(reason=str(exc), rejection_stage='rendering')
                    if balancer: balancer.reject(choice)
                    job['progress']['rejected'] += 1
                    # Preserve this attempt's serial and partial artifacts so the
                    # next layout cannot overwrite them or include them in export.
                    job['episodes'].append(item)
                    update('Rejected invisible task cube; trying another layout')
                    continue
                capture = json.loads((episode / 'capture.json').read_text())
                # Same physical trajectory and all its visual variants share a
                # group; this is stable across repeated batches, not job IDs.
                scene = json.loads((episode / 'generation-scene.json').read_text())
                group = trajectory_group(scene, report['action_hash'])
                metadata = dict(task_id=task_id, instruction=capture['instruction'], seed=seed, group_id=group,
                                action_hash=report['action_hash'], outcome='success', transfer_checks_passed=True,
                                simulation_only=True, physical_model_validated=False, options=options)
                save(episode / 'episode.json', metadata)
                item.update(status='accepted', instruction=capture['instruction'], frames=report['frames'], duration=report['duration'], **episode_urls(job['id'], serial))
                job['episodes'].append(item); accepted.append(episode); count += 1
                job['progress']['accepted'] += 1
                if balancer:
                    balancer.accept_world(scene['world'])
                    job.setdefault('coverage', {})[task_id] = balancer.summary()
                update('Accepted physical demonstration')
            if count < requested:
                raise ValueError(f'Only {count}/{requested} valid episodes for {instruction} after bounded retries; batch not exported')
        update('Exporting and auditing LeRobot v3 dataset')
        exported = export_dataset(accepted, folder / 'dataset', name=options['name'], validation_fraction=options['validation_fraction'],
                                  inherited_group_splits=job.get('inherited_group_splits'), lineage=job.get('lineage'))
        update('Packaging dataset with provenance')
        shutil.make_archive(str(folder / 'dataset'), 'zip', root_dir=folder, base_dir='dataset')
        job.update(status='completed', phase='LeRobot dataset ready', export=exported | {'url': f'/api/generation/{job["id"]}/download'}, completed_at=time.time())
        save(folder / 'job.json', job)
    except Exception as exc:
        traceback.print_exc()
        job.update(status='failed', phase='Generation failed; incomplete batch is not exported')
        job['errors'].append(str(exc)); save(folder / 'job.json', job)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['worker']); parser.add_argument('folder', type=Path)
    args = parser.parse_args(); work(args.folder)
