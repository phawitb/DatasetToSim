"""Asynchronous dataset builds from verified original recorded native actions.

The calibration and physics traces are frozen. Appearance changes only affect
rendering. Failed physical tasks may be inspected but never enter training data.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from studio_replay_dataset_source import ReplaySource, SOURCE_ID, TASK_ID, INSTRUCTION
from studio_generation_visuals import VISUAL_DEFAULTS, validate_visual_options


def save(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)); temporary.replace(path)


def validate_request(body):
    defaults = dict(name='real-action-simulation',source_id=SOURCE_ID,episode_ids=[],success_only=True,
                    seed=0,validation_fraction=.2,visual_randomization=False,camera_jitter=False,
                    sensor_augmentation=False,real_background=True,real_image_processing=True,parent_job_id=None) | VISUAL_DEFAULTS
    if not isinstance(body,dict) or set(body) - set(defaults):
        raise ValueError('Unknown replay options; only appearance changes are supported')
    result = defaults | body
    if result['source_id'] != SOURCE_ID:
        raise ValueError('This source has no validated shared replay calibration')
    ids = result['episode_ids']
    if not isinstance(ids,list) or not ids or any(type(i) is not int or not 0 <= i < 72 for i in ids):
        raise ValueError('Select one or more calibrated real episode IDs (0–71)')
    if len(ids) != len(set(ids)):
        raise ValueError('Episode selection must not contain duplicates')
    if not isinstance(result['name'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',result['name']):
        raise ValueError('Use 1–64 letters, numbers, hyphens or underscores for the dataset name')
    if type(result['seed']) is not int or not 0 <= result['seed'] < 2**31:
        raise ValueError('Seed must be an integer from 0 to 2147483647')
    for key in ('success_only','visual_randomization','camera_jitter','sensor_augmentation','real_background','real_image_processing'):
        if type(result[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    parent = result['parent_job_id']
    if parent is not None and (not isinstance(parent,str) or not re.fullmatch('[a-f0-9]{32}',parent)):
        raise ValueError('Invalid parent dataset identifier')
    fraction = result['validation_fraction']
    if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0 <= fraction <= .4:
        raise ValueError('validation_fraction must be between 0 and 0.4')
    return validate_visual_options(result)


class ReplayDatasetManager:
    def __init__(self, root, *, media_prefix='/media/generated'):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.media_prefix = media_prefix.rstrip('/')
        self.process = None; self.active = None; self.lock = threading.RLock()

    def sources(self):
        return [ReplaySource().catalog()]

    def folder(self, identifier):
        if not isinstance(identifier,str) or not re.fullmatch('[a-f0-9]{32}',identifier):
            raise ValueError('Invalid replay dataset identifier')
        folder = self.root / identifier
        if not (folder / 'job.json').is_file():
            raise ValueError('Replay dataset job not found')
        if json.loads((folder / 'job.json').read_text()).get('source_type') != 'real_replay':
            raise ValueError('This is not a real replay dataset job')
        return folder

    def busy(self):
        return self.process is not None and self.process.poll() is None

    def get(self, identifier):
        with self.lock:
            folder = self.folder(identifier)
            job = json.loads((folder / 'job.json').read_text())
            if identifier == self.active and self.process is not None and self.process.poll() is not None and job['status'] in ('queued','running'):
                job.update(status='failed',phase='Replay worker exited unexpectedly')
                job.setdefault('errors',[]).append('Inspect worker.log for diagnostic details')
                save(folder / 'job.json',job)
            return job

    def list(self):
        jobs = []
        for path in sorted(self.root.glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            try:
                jobs.append(self.get(path.parent.name))
            except (ValueError,OSError):
                continue
        return jobs

    def recover_interrupted(self):
        for job in self.list():
            if job['status'] in ('queued','running'):
                job.update(status='cancelled',phase='Interrupted by server restart')
                save(self.folder(job['id']) / 'job.json',job)

    def start(self, body):
        return self._start(body,preview=False)

    def start_preview(self, body):
        return self._start(body,preview=True)

    def _start(self, body, *, preview):
        options = validate_request(body)
        if preview and len(options['episode_ids']) > 5:
            options['episode_ids'] = options['episode_ids'][:5]
        with self.lock:
            if self.busy():
                raise ValueError('A real replay dataset operation is already running')
            source = ReplaySource()
            inherited = self.parent_splits(options['parent_job_id']) if options['parent_job_id'] and not preview else {}
            if preview:
                options['parent_job_id'] = None
            selected = [source.reports[i] for i in options['episode_ids']]
            included = [item for item in selected if not options['success_only'] or item['outcome'] == 'success']
            if not included:
                raise ValueError('No successful selected episodes; disable success-only to inspect known failures')
            identifier = uuid.uuid4().hex; folder = self.root / identifier; folder.mkdir()
            try:
                from studio_generation_assets import snapshot_background
                asset = snapshot_background(options['background_id'],folder / 'assets')
            except Exception:
                shutil.rmtree(folder); raise
            skipped = [dict(source_episode_index=item['episode'],outcome=item['outcome'],reason='Excluded by success-only filter') for item in selected if item not in included]
            options['tasks'] = [TASK_ID]
            job = dict(id=identifier,name=options['name'],source_type='real_replay',kind='preview' if preview else 'dataset',
                       status='queued',phase='Verifying selected real action traces',created_at=time.time(),
                       options=options,background_asset=asset,media_prefix=self.media_prefix,
                       selected_episode_ids=options['episode_ids'],included_episode_ids=[item['episode'] for item in included],
                       progress=dict(requested=len(included),attempts=0,accepted=0,rejected=0),
                       episodes=[],images=[],skipped=skipped,errors=[],parent_job_id=options['parent_job_id'],
                       inherited_group_splits=inherited,
                       note='Original recorded actions; frozen physics traces; appearance-only variation. Failed tasks are reported and excluded from training export. No hardware commands.')
            save(folder / 'job.json',job)
            try:
                with (folder / 'worker.log').open('w') as log:
                    self.process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker',str(folder)],
                        stdout=log,stderr=log,start_new_session=True,env=os.environ | {'STUDIO_REPLAY_PARENT':str(os.getpid())})
            except OSError as exc:
                job.update(status='failed',phase='Unable to start replay worker'); job['errors'].append(str(exc)); save(folder / 'job.json',job)
                raise ValueError('Unable to start replay worker') from exc
            self.active = identifier
            return job

    def parent_splits(self, identifier):
        """Train/val assignment of a completed replay dataset, so revisions never move a recording across splits."""
        job = self.get(identifier)
        if job['status'] != 'completed' or job.get('kind') != 'dataset' or job.get('export',{}).get('valid') is not True:
            raise ValueError('Only a completed, validated replay dataset can be extended')
        manifest = json.loads((self.folder(identifier) / 'dataset/export_manifest.json').read_text())
        return {str(item['group_id']): item['split'] for item in manifest['episodes']}

    def cancel(self, identifier):
        with self.lock:
            job = self.get(identifier)
            if identifier == self.active and self.busy():
                from studio_runtime import terminate_group
                terminate_group(self.process)
                job = json.loads((self.folder(identifier) / 'job.json').read_text())
            if job['status'] in ('queued','running'):
                job.update(status='cancelled',phase='Cancelled; incomplete batch is not exported')
                save(self.folder(identifier) / 'job.json',job)
            return job

    def close(self):
        if self.active and self.busy():
            self.cancel(self.active)

    def download(self, identifier):
        job = self.get(identifier)
        if job['status'] != 'completed' or not job.get('export',{}).get('valid'):
            raise ValueError('No completed successful training export is available')
        return self.folder(identifier) / 'dataset.zip'

    def episode(self, identifier, index):
        from studio_generation_view import episode_data
        job = self.get(identifier)
        if type(index) is not int or index < 0 or index >= len(job['episodes']):
            raise ValueError('Episode not found')
        item = job['episodes'][index]
        if item['status'] != 'accepted':
            return item | {'status':'diagnostic','note':'Failed task is excluded from training data'}
        return episode_data(self.folder(identifier),index) | item


def work(folder):
    parent = int(os.environ.get('STUDIO_REPLAY_PARENT',os.getppid()))
    if 'STUDIO_REPLAY_PARENT' in os.environ:
        def watch_parent():
            while True:
                time.sleep(1)
                if os.getppid() != parent:
                    os.killpg(os.getpgrp(),signal.SIGTERM); return
        threading.Thread(target=watch_parent,daemon=True).start()
    import traceback
    from studio_replay_dataset_render import render_episode
    from studio_generation_export import export_dataset
    folder = Path(folder); job = json.loads((folder / 'job.json').read_text())
    options = job['options']; preview = job['kind'] == 'preview'
    def update(phase):
        job.update(status='running',phase=phase); save(folder / 'job.json',job)
    try:
        source = ReplaySource(); accepted = []
        for index,episode_id in enumerate(job['included_episode_ids']):
            update(f'Verifying real episode {episode_id}')
            destination = folder / 'episodes' / f'{index:05d}'
            seed = options['seed'] + index
            metadata = source.prepare(episode_id,destination,seed=seed)
            update(f'Rendering real episode {episode_id} ({index+1}/{len(job["included_episode_ids"])})')
            render_episode(destination,seed,options | {'_background_asset':job['background_asset']},preview=preview,
                           progress=lambda i,n:update(f'Rendering real episode {episode_id}: frame {i+1}/{n}'))
            base = f'{job["media_prefix"]}/{job["id"]}/episodes/{index:05d}'
            success = metadata['outcome'] == 'success'
            item = dict(index=index,source_episode_index=episode_id,source_type='real_replay',task_id=TASK_ID,
                        instruction=INSTRUCTION,seed=seed,frames=metadata['frames'],outcome=metadata['outcome'],
                        status='preview' if preview else ('accepted' if success else 'rejected'),rendered=True,
                        preview_url=base+'/top.jpg',top_image=base+'/top.jpg',wrist_image=base+'/wrist.jpg',
                        video_urls={} if preview else {view:base+f'/{view}.mp4' for view in ('top','wrist')})
            job['episodes'].append(item); job['images'].append(item)
            job['progress']['attempts'] += 1
            job['progress']['accepted' if success else 'rejected'] += 1
            if success and not preview:
                accepted.append(destination)
            save(folder / 'job.json',job)
        if accepted:
            update('Writing and auditing LeRobot v3 export')
            groups = {json.loads((path/'episode.json').read_text()).get('group_id') for path in accepted}
            inherited = {group: split for group, split in job.get('inherited_group_splits',{}).items() if group in groups}
            exported = export_dataset(accepted,folder/'dataset',name=options['name'],validation_fraction=options['validation_fraction'],
                                      inherited_group_splits=inherited,
                                      lineage={'source_type':'real_replay','source_id':SOURCE_ID,'parent_job_id':job.get('parent_job_id'),
                                               'source_settings_sha256':source.settings_hash,'selected_episode_ids':options['episode_ids']})
            # The full build report includes explicit failures even though the
            # validated training export intentionally contains successes only.
            save(folder/'dataset/replay-build-report.json',{'source_type':'real_replay','episodes':job['episodes'],'skipped':job['skipped'],
                 'physical_transfer_validated':False,'source_settings_sha256':source.settings_hash,'options':options})
            update('Packaging validated dataset')
            shutil.make_archive(str(folder/'dataset'),'zip',root_dir=folder/'dataset')
            job['export'] = exported | {'url':f'/api/replay-datasets/{job["id"]}/download'}
        job.update(status='completed',completed_at=time.time(),phase='Appearance preview ready' if preview else ('LeRobot dataset ready' if accepted else 'Diagnostic rendering complete; no successful episodes to export'))
        job['summary'] = dict(fps=30,cameras=['top','wrist'],resolution=[640,480],format='LeRobot v3',
                             total_frames=sum(item['frames'] for item in job['episodes'] if item['status']=='accepted'),
                             accepted_episodes=len(accepted),rejected_episodes=job['progress']['rejected'])
        save(folder/'job.json',job)
    except Exception as exc:
        traceback.print_exc()
        job.update(status='failed',phase='Replay build failed; incomplete batch is not exported')
        job['errors'].append(f'{type(exc).__name__}: {exc}'); save(folder/'job.json',job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['worker']); parser.add_argument('folder',type=Path)
    args = parser.parse_args(); work(args.folder)
