"""Imported real datasets: automatic calibration + strict replay jobs run as subprocesses."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]
SELECTABLE = ('eraser', 'cube', 'tape', 'green-bowl')


def _read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default


class ImportManager:
    def __init__(self, root=ROOT / 'output/imports'):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock(); self.processes = {}

    def library(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from import_calibration import library_objects
        specs = library_objects()
        return [{'id': key, 'name': specs[key]['name'], 'role': specs[key]['role'],
                 'thumbnail_url': f"/api/objects/{ {'cube': 'real-red-cube'}.get(key, key)}/thumbnail",
                 'dimensions_m': specs[key].get('size_m') or [2 * specs[key].get('top_radius_m', specs[key].get('radius_m', 0))] * 2 + [specs[key]['height_m']],
                 'replay_supported': specs[key]['role'] == 'container' or specs[key]['shape'] == 'box'} for key in SELECTABLE]

    def folder(self, identifier):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', identifier or ''): raise ValueError('Import not found')
        folder = self.root / identifier
        if not (folder / 'request.json').is_file(): raise ValueError('Import not found')
        return folder

    def busy(self):
        with self.lock: return any(p.poll() is None for p in self.processes.values())

    def start(self, body):
        path = Path(str(body.get('dataset_path', '')).strip()).expanduser()
        if not path.is_absolute() or not (path / 'meta/info.json').is_file():
            raise ValueError('Enter the absolute path of a LeRobot v3 dataset folder (it must contain meta/info.json)')
        objects = body.get('objects')
        if not isinstance(objects, list) or not objects or any(o not in SELECTABLE for o in objects):
            raise ValueError('Select the objects that appear in the images')
        sys.path.insert(0, str(ROOT / 'scripts'))
        from import_calibration import library_objects
        specs = library_objects(); roles = [specs[o]['role'] for o in objects]
        if roles.count('manipulated') != 1 or roles.count('container') != 1:
            raise ValueError('Select exactly one object that is picked and one bowl')
        with self.lock:
            if any(p.poll() is None for p in self.processes.values()): raise ValueError('Wait for the current import to finish')
            identifier = re.sub(r'[^A-Za-z0-9_.-]', '_', path.name)[:40] + '-' + uuid.uuid4().hex[:6]
            folder = self.root / identifier; folder.mkdir()
            request = {'id': identifier, 'dataset_path': str(path.resolve()), 'name': path.name, 'objects': objects,
                       'created': datetime.now(timezone.utc).isoformat(timespec='seconds')}
            (folder / 'request.json').write_text(json.dumps(request, indent=2))
            log = open(folder / 'import.log', 'w')
            self.processes[identifier] = subprocess.Popen(
                [sys.executable, str(ROOT / 'scripts/import_dataset.py'), request['dataset_path'], '--objects', *objects, '--output', str(folder)],
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
            log.close()
        return self.get(identifier)

    def cancel(self, identifier):
        folder = self.folder(identifier)
        with self.lock:
            process = self.processes.get(identifier)
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        progress = _read(folder / 'progress.json', {}) | {'status': 'cancelled'}
        (folder / 'progress.json').write_text(json.dumps(progress, indent=2))
        return self.get(identifier)

    def get(self, identifier, detail=True):
        folder = self.folder(identifier); request = _read(folder / 'request.json')
        progress = _read(folder / 'progress.json', {'status': 'running', 'stage': 'starting', 'log': []})
        with self.lock: process = self.processes.get(identifier)
        if progress.get('status') == 'running' and (process is None or process.poll() is not None):
            progress['status'] = 'interrupted' if process is None else 'error'
        results = _read(folder / 'results.json'); calibration = _read(folder / 'calibration.json')
        item = dict(request, status=progress.get('status'), stage=progress.get('stage'),
                    successes=results and results['successes'], completed=results and results['completed'],
                    total=progress.get('total') or (results and results['total']), error=progress.get('error'),
                    camera_mask_iou=calibration and calibration['top_camera_report'].get('mean_mask_iou'))
        if not detail: return item
        media = lambda name: f'/media/imports/{identifier}/{name}' if (folder / name).is_file() else None
        episodes = []
        for row in (results or {}).get('results', []):
            name = f'episode-{row["episode"]:03d}'
            episodes.append(row | {'video': media(f'{name}/real-vs-sim.mp4'), 'scene_fit': media(f'scene-fit-{row["episode"]:03d}.jpg')})
        summary = None
        if calibration:
            summary = {'camera': calibration['top_camera'], 'camera_report': {k: v for k, v in calibration['top_camera_report'].items() if k != 'coarse_candidates'},
                       'mat': calibration['mat'], 'mat_report': calibration['mat_report'], 'object_reports': calibration['object_reports'],
                       'objects': calibration['objects']}
        return item | {'log': progress.get('log', [])[-40:], 'camera_fit_image': media('camera-fit.jpg'),
                       'calibration': summary, 'episodes': episodes, 'counts': results and results.get('counts')}

    def list(self):
        items = []
        for folder in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (folder / 'request.json').is_file():
                try: items.append(self.get(folder.name, detail=False))
                except ValueError: pass
        return items

    def close(self):
        with self.lock:
            for process in self.processes.values():
                if process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
