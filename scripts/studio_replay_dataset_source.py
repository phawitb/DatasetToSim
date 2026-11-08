"""Read-only bridge from the audited real-action replay to portable datasets."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.strict_replay import NativeJointAdapter, array_hash

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = 'SIMCUBE_1O1B_VF'
DATASET = Path('/Users/phawit/Projects/VLA-Benchmark/data') / SOURCE_ID
RUN = ROOT / 'output/shared-calibration-v10'
TASK_ID = 'red_cube_to_green_bowl'
INSTRUCTION = 'Put the red cube in the green bowl.'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def verify_trace(episode, trace, report, settings):
    """Reject changed commands, state alignment, adapter targets or timing."""
    def exact(actual, expected, label):
        if not np.array_equal(actual, expected):
            raise ValueError('Archived ' + label + ' differs from the real recording')
    n = episode.frame_count
    exact(trace['native_action_applied'], episode.action_deg, 'action')
    exact(trace['recorded_native_state'], episode.state_deg, 'recorded state')
    exact(trace['source_timestamps'], episode.timestamps, 'timestamps')
    exact(trace['sample_times'][:-1], episode.timestamps, 'sample times')
    adapter = NativeJointAdapter(**settings['adapter'])
    exact(trace['model_targets'], adapter.to_model(episode.action_deg), 'adapter targets')
    exact(trace['step_controls'], trace['model_targets'][trace['step_action_index']], 'step controls')
    if trace['qpos'].shape != (n + 1, 6) or trace['cube_pose'].shape != (n + 1, 7):
        raise ValueError('Trace must contain N+1 integrated states')
    if not np.allclose(trace['simulated_native_state'], adapter.to_native(trace['qpos']), atol=1e-10, rtol=0):
        raise ValueError('Native state conversion differs from archived model states')
    if not np.isfinite(trace['qpos']).all() or not np.isfinite(trace['cube_pose']).all():
        raise ValueError('Trace contains non-finite states')
    if report['native_action_sha256'] != array_hash(episode.action_deg):
        raise ValueError('Native action provenance hash mismatch')
    if not np.isclose(trace['sample_times'][-1], episode.timestamps[-1] + 1 / 30, atol=1e-10):
        raise ValueError('Final action interval is missing')
    if not report.get('action_equality_verified') or not report.get('initial_collision_screen_passed'):
        raise ValueError('Archive lacks required action and initial collision checks')


class ReplaySource:
    def __init__(self, dataset_path=DATASET, run=RUN):
        self.dataset_path, self.run = Path(dataset_path).resolve(), Path(run).resolve()
        self.settings = json.loads((self.run / 'settings.json').read_text())
        self.results = json.loads((self.run / 'results.json').read_text())
        self.verification = json.loads((self.run / 'verification.json').read_text())
        self.settings_hash = hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode()).hexdigest()
        if self.results.get('mode') != 'strict_native_action' or self.results.get('native_action_changed') is not False or self.results.get('trajectory_repair') is not False:
            raise ValueError('Only unchanged native-action physical replay is supported')
        if self.results['settings_sha256'] != self.settings_hash or self.verification['same_settings_sha256'] != self.settings_hash:
            raise ValueError('Archived shared settings hash mismatch')
        for key in ('all_native_actions_exactly_equal', 'all_step_targets_exactly_equal_to_fixed_adapter', 'all_action_boundaries_verified', 'all_saved_sources_and_dataset_hashes_verified'):
            if self.verification.get(key) is not True:
                raise ValueError('Strict archive verification is incomplete')
        self.dataset = DatasetIndex(self.dataset_path)
        self.reports = {int(row['episode']): row for row in self.results['results']}
        self._verified_files = set()

    def catalog(self):
        episodes = []
        for index, report in sorted(self.reports.items()):
            # Source ranges/statistics can be stale. Resolve physical parquet rows.
            episode = self.dataset.episode(index)
            episodes.append(dict(episode_index=index, frames=episode.frame_count,
                                 duration_s=episode.frame_count / 30, outcome=report['outcome'],
                                 success=report['outcome'] == 'success'))
        successes = sum(item['success'] for item in episodes)
        return dict(id=SOURCE_ID, name=SOURCE_ID, supported=True, episode_count=len(episodes),
                    successes=successes, failures=len(episodes)-successes, episodes=episodes,
                    note='Frozen shared calibration; 68/72 calibration-set task successes. Original native actions and integrated physics states; no hardware-transfer validation.')

    def read(self, index):
        if type(index) is not int or index not in self.reports:
            raise ValueError('Episode is not in the calibrated real source')
        folder = self.run / f'episode-{index:02d}'
        report = json.loads((folder / 'report.json').read_text())
        if report != self.reports[index] or report['settings_sha256'] != self.settings_hash:
            raise ValueError('Episode report differs from the audited run')
        for file, key in [('strict_trace.npz','trace_archive_sha256'), ('scene.xml','model_archive_sha256')]:
            if sha(folder / file) != report[key]:
                raise ValueError('Archived ' + file + ' hash mismatch')
        # Verify the real metadata/data and selected video assets before claiming
        # their provenance. Cache only within one immutable worker source object.
        episode = self.dataset.episode(index)
        paths = [self.dataset_path / 'meta/info.json', *self.dataset._data_paths[index],
                 *sorted((self.dataset_path / 'meta/episodes').glob('chunk-*/*.parquet')),
                 *(item.path for item in episode.video_slices.values())]
        for path in paths:
            relative = str(path.relative_to(self.dataset_path))
            if relative not in self._verified_files:
                expected = self.settings['dataset_input_sha256'].get(relative)
                if expected is None or sha(path) != expected:
                    raise ValueError('Real source has changed since calibration: ' + relative)
                self._verified_files.add(relative)
        for path in sorted((self.run / 'assets').glob('*.stl')):
            key = 'mesh:' + path.name
            if key not in self._verified_files:
                expected = [value for name,value in self.settings['robot_asset_sha256'].items() if Path(name).name == path.name]
                if len(expected) != 1 or sha(path) != expected[0]:
                    raise ValueError('Archived mesh asset changed: ' + path.name)
                self._verified_files.add(key)
        with np.load(folder / 'strict_trace.npz', allow_pickle=False) as archive:
            trace = {key: archive[key].copy() for key in archive.files}
        verify_trace(episode, trace, report, self.settings)
        return episode, trace, report, folder

    def prepare(self, index, destination, *, seed):
        episode, trace, report, source = self.read(index)
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        # This bridge changes file layout only, never native actions or physics.
        np.savez_compressed(destination / 'plan.npz', actions=episode.action_deg, timestamps=episode.timestamps)
        np.savez_compressed(destination / 'trace.npz', **trace, native=trace['simulated_native_state'])
        tree = ET.parse(source / 'scene.xml')
        tree.getroot().find('compiler').set('meshdir', str(self.run / 'assets'))
        tree.write(destination / 'scene.xml', encoding='unicode')
        metadata = dict(source_type='real_replay', source_id=SOURCE_ID, source_episode_index=index,
                        source_dataset=str(self.dataset_path), source_replay=str(self.run),
                        source_model_sha256=report['model_archive_sha256'], source_trace_sha256=report['trace_archive_sha256'],
                        source_settings_sha256=self.settings_hash,
                        source_video_slices={view:dict(path=str(item.path), from_timestamp=item.start_time,
                            to_timestamp=item.end_time, sha256=self.settings['dataset_input_sha256'][str(item.path.relative_to(self.dataset_path))])
                            for view,item in episode.video_slices.items()},
                        task_id=TASK_ID, instruction=INSTRUCTION, seed=seed, frames=episode.frame_count,
                        group_id='real-' + SOURCE_ID + '-' + str(index), action_hash=array_hash(episode.action_deg),
                        adapter=self.settings['adapter'], outcome=report['outcome'],
                        transfer_checks_passed=True, physical_transfer_validated=False,
                        checks_note='Integrity, native channel/timing checks and initial collision screen only; no physical hardware or dynamics validation.',
                        native_action_changed=False, trajectory_repair=False, state_following=False,
                        observation_alignment='pre-action integrated state k, rendered image k, original real command k; terminal state excluded',
                        source_camera_delay_frames=self.settings.get('camera_delay_frames', {}),
                        exported_camera_delay_frames={'top':0,'wrist':0},
                        camera_timing_note='Export images use simultaneous pre-action simulator states; source camera delay is retained as provenance, not applied to training observations.')
        write_json(destination / 'episode.json', metadata)
        write_json(destination / 'plan.json', metadata)
        write_json(destination / 'report.json', report | {'transfer_checks_passed':True})
        write_json(destination / 'generation-config.json', self.settings)
        world = json.loads((source / 'calibration.json').read_text())['world']
        world['cube']['position'] = trace['cube_pose'][0, :3].tolist()
        world['cube']['quaternion'] = trace['cube_pose'][0, 3:].tolist()
        write_json(destination / 'generation-scene.json', {
            'source_type': 'real_replay', 'source_episode_index': index,
            'world': world, 'note': 'Archived per-episode initial geometry; cube pose taken directly from integrated trace state zero.'})
        return metadata
