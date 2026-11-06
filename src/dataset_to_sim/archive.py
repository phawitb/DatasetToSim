"""Load the model archive declared by a replay, never an obsolete sibling file."""
from pathlib import Path
import hashlib
import mujoco


def load_saved_model(folder,settings,expected_sha256=None):
    folder=Path(folder)
    xml_archive=settings.get('model_archive','').startswith('self-contained per-episode XML')
    path=folder/'scene.xml' if xml_archive else (folder/'scene.mjb' if (folder/'scene.mjb').exists() else folder/'scene.xml')
    if expected_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError(f'Saved model hash mismatch: {path}')
    return mujoco.MjModel.from_xml_path(str(path)) if path.suffix=='.xml' else mujoco.MjModel.from_binary_path(str(path))
