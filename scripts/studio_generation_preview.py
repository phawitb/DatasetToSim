"""Bounded isolated workers for approximate, non-physical still previews."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

from studio_generation_assets import snapshot_background
from studio_runtime import terminate_group

TIMEOUT_SECONDS = 120


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class PreviewManager:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.process = None; self.active = None; self.lock = threading.RLock()

    def folder(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch('[a-f0-9]{32}', identifier):
            raise ValueError('Invalid preview identifier')
        folder = self.root / identifier
        if not (folder / 'preview.json').is_file():
            raise ValueError('Preview not found')
        return folder

    def get(self, identifier):
        with self.lock:
            folder = self.folder(identifier)
            job = json.loads((folder / 'preview.json').read_text())
            if identifier == self.active and job['status'] in ('queued', 'running'):
                if self.process and self.process.poll() is None and time.time() - job['created_at'] > TIMEOUT_SECONDS:
                    terminate_group(self.process)
                if self.process and self.process.poll() is not None:
                    # Worker may complete between the first read and poll.
                    job = json.loads((folder / 'preview.json').read_text())
                    if job['status'] in ('queued', 'running'):
                        job.update(status='failed', error='Preview worker stopped. Try Preview again; inspect preview worker.log if it persists.')
                        save(folder / 'preview.json', job)
            return job

    def busy(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def start(self, options):
        with self.lock:
            if self.busy():
                self.close()  # The latest settings replace an obsolete preview.
            identifier = uuid.uuid4().hex
            folder = self.root / identifier; folder.mkdir()
            frozen = json.loads(json.dumps(options))
            asset = snapshot_background(frozen.get('background_id', 'calibrated-real'), folder / 'assets')
            digest = hashlib.sha256(json.dumps({'options': frozen, 'background_sha256': asset['sha256']}, sort_keys=True).encode()).hexdigest()
            job = {'id': identifier, 'status': 'queued', 'created_at': time.time(),
                   'options': frozen, 'background_asset': asset, 'settings_hash': digest,
                   'images': [], 'note': 'Approximate initial-layout and appearance preview; reachability and task success are checked during generation.'}
            save(folder / 'preview.json', job)
            try:
                with (folder / 'worker.log').open('w') as log:
                    self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(folder)],
                        stdout=log, stderr=log, start_new_session=True,
                        env=os.environ | {'STUDIO_PREVIEW_PARENT': str(os.getpid())})
            except OSError as exc:
                job.update(status='failed', error='Unable to start preview worker')
                save(folder / 'preview.json', job)
                raise ValueError(job['error']) from exc
            self.active = identifier
            return job

    def close(self):
        with self.lock:
            if self.busy():
                terminate_group(self.process)
            if self.active:
                folder = self.folder(self.active); job = json.loads((folder / 'preview.json').read_text())
                if job['status'] in ('queued', 'running'):
                    job.update(status='cancelled', error='Preview interrupted')
                    save(folder / 'preview.json', job)
            self.process = None

    def recover_interrupted(self):
        for path in self.root.glob('*/preview.json'):
            try:
                job = json.loads(path.read_text())
                if job['status'] in ('queued', 'running'):
                    job.update(status='cancelled', error='Preview interrupted by server restart')
                    save(path, job)
            except (OSError, ValueError, KeyError):
                continue


def work(folder):
    parent = int(os.environ.get('STUDIO_PREVIEW_PARENT', os.getppid()))
    started = time.monotonic()
    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid() != parent or time.monotonic() - started > TIMEOUT_SECONDS:
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return
    threading.Thread(target=watch_parent, daemon=True).start()
    # Graphics and simulation imports belong exclusively to this subprocess.
    from studio_generation_scene import initialize_generation_scene
    from studio_generation_render import render_initial_pair
    folder = Path(folder); job = json.loads((folder / 'preview.json').read_text())
    options = job['options'] | {'_background_asset': job['background_asset']}
    try:
        job['status'] = 'running'; save(folder / 'preview.json', job)
        for index in range(5):
            task = options['tasks'][index % len(options['tasks'])]
            seed = options['seed'] + index
            scene = folder / f'scene-{index}'
            initialize_generation_scene(scene, task, seed, options)
            render_initial_pair(scene, seed, options)
            base = f'/api/generation-preview/{job["id"]}/images/scene-{index}'
            job['images'].append({'index': index, 'task_id': task, 'seed': seed,
                                  'top_url': base + '/top.jpg', 'wrist_url': base + '/wrist.jpg'})
            save(folder / 'preview.json', job)
        job.update(status='completed', completed_at=time.time()); save(folder / 'preview.json', job)
    except Exception as exc:
        job.update(status='failed', error=str(exc)); save(folder / 'preview.json', job)
        raise


if __name__ == '__main__':
    work(Path(sys.argv[1]))
