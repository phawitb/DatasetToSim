import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_to_sim.dataset import DatasetFormatError, DatasetIndex


JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def write_dataset(root, joint_names=JOINTS, metadata_indices=(7, 9)):
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.top/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.wrist/chunk-000").mkdir(parents=True)
    info = {
        "fps": 30,
        "total_episodes": 1,
        "features": {
            "observation.state": {"names": joint_names, "shape": [6]},
            "action": {"names": joint_names, "shape": [6]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0, 10, 20, 30, 40, 50], [1, 11, 21, 31, 41, 51]],
                "action": [[2, 12, 22, 32, 42, 52], [3, 13, 23, 33, 43, 53]],
                "timestamp": [0.0, 1 / 30],
                "episode_index": [0, 0],
                "frame_index": [0, 1],
                "index": [7, 8],
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "length": [2],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "dataset_from_index": [metadata_indices[0]],
                "dataset_to_index": [metadata_indices[1]],
                "videos/observation.images.top/chunk_index": [0],
                "videos/observation.images.top/file_index": [0],
                "videos/observation.images.top/from_timestamp": [5.0],
                "videos/observation.images.top/to_timestamp": [5.0 + 2 / 30],
                "videos/observation.images.wrist/chunk_index": [0],
                "videos/observation.images.wrist/file_index": [0],
                "videos/observation.images.wrist/from_timestamp": [5.0],
                "videos/observation.images.wrist/to_timestamp": [5.0 + 2 / 30],
            }
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    for view in ("top", "wrist"):
        (root / f"videos/observation.images.{view}/chunk-000/file-000.mp4").touch()


def test_episode_loads_exact_parquet_and_video_slice(tmp_path):
    write_dataset(tmp_path)

    episode = DatasetIndex(tmp_path).episode(0)

    assert episode.index == 0
    assert episode.frame_indices.tolist() == [0, 1]
    assert episode.dataset_indices.tolist() == [7, 8]
    assert episode.timestamps.tolist() == pytest.approx([0.0, 1 / 30])
    assert episode.state_deg.tolist() == [[0, 10, 20, 30, 40, 50], [1, 11, 21, 31, 41, 51]]
    assert episode.video_slices["top"].start_time == pytest.approx(5.0)
    assert episode.video_slices["top"].path.name == "file-000.mp4"


def test_loader_rejects_joint_contract_mismatch(tmp_path):
    write_dataset(tmp_path, joint_names=[*JOINTS[:-1], "pinch.pos"])

    with pytest.raises(DatasetFormatError, match="gripper.pos"):
        DatasetIndex(tmp_path)


def test_loader_uses_episode_identity_when_metadata_indices_follow_video_timeline(tmp_path):
    write_dataset(tmp_path, metadata_indices=(100, 102))

    episode = DatasetIndex(tmp_path).episode(0)

    assert episode.dataset_indices.tolist() == [7, 8]
    assert episode.frame_indices.tolist() == [0, 1]
