"""Strict reader for the LeRobot v3 shards used by SIMCUBE."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .joints import FEATURE_JOINT_NAMES


class DatasetFormatError(ValueError):
    """The dataset is incomplete or violates the replay input contract."""


@dataclass(frozen=True)
class VideoSlice:
    path: Path
    start_time: float
    end_time: float


@dataclass(frozen=True)
class Episode:
    index: int
    frame_indices: np.ndarray
    dataset_indices: np.ndarray
    timestamps: np.ndarray
    state_deg: np.ndarray
    action_deg: np.ndarray
    video_slices: dict[str, VideoSlice]

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)


def _field(row: dict[str, Any], name: str) -> Any:
    if name not in row:
        raise DatasetFormatError(f"episode metadata is missing {name!r}")
    return row[name]


class DatasetIndex:
    """Indexes source data without writing to it or decoding a full video."""

    VIEWS = ("top", "wrist")

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "meta/info.json"
        if not info_path.is_file():
            raise DatasetFormatError(f"missing metadata file: {info_path}")
        try:
            self.info = json.loads(info_path.read_text())
        except json.JSONDecodeError as error:
            raise DatasetFormatError(f"invalid JSON in {info_path}: {error}") from error
        self._validate_info()
        self._episodes = self._read_episode_metadata()
        self._data_paths = self._index_data_shards()
        if len(self._episodes) != int(self.info["total_episodes"]):
            raise DatasetFormatError(
                f"metadata declares {self.info['total_episodes']} episodes but found {len(self._episodes)}"
            )

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def _validate_info(self) -> None:
        if self.info.get("fps") != 30:
            raise DatasetFormatError(f"expected 30 Hz input, got {self.info.get('fps')!r}")
        features = self.info.get("features")
        if not isinstance(features, dict):
            raise DatasetFormatError("metadata has no features object")
        for key in ("observation.state", "action"):
            actual = features.get(key, {}).get("names")
            if actual != list(FEATURE_JOINT_NAMES):
                raise DatasetFormatError(
                    f"{key} joint names must be {list(FEATURE_JOINT_NAMES)!r}; got {actual!r}"
                )

    def _read_episode_metadata(self) -> dict[int, dict[str, Any]]:
        paths = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        if not paths:
            raise DatasetFormatError("no episode metadata parquet shards found")
        episodes: dict[int, dict[str, Any]] = {}
        for path in paths:
            for row in pq.read_table(path).to_pylist():
                index = int(_field(row, "episode_index"))
                if index in episodes:
                    raise DatasetFormatError(f"duplicate episode_index {index}")
                episodes[index] = row
        expected = set(range(len(episodes)))
        if set(episodes) != expected:
            raise DatasetFormatError("episode indices must be contiguous and start at zero")
        return episodes

    def _index_data_shards(self) -> dict[int, list[Path]]:
        """Map trajectory identity from parquet, not from video-timeline indices.

        Some real captures retain original video-global indices in episode
        metadata while state parquet rows have been compacted.  `episode_index`
        is the shared identity that survives that compaction.
        """
        paths = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not paths:
            raise DatasetFormatError("no data parquet shards found")
        locations: dict[int, list[Path]] = {}
        for path in paths:
            try:
                indices = pq.read_table(path, columns=["episode_index"]).column(0).to_pylist()
            except Exception as error:
                raise DatasetFormatError(f"cannot read episode_index from {path}: {error}") from error
            for index in set(map(int, indices)):
                locations.setdefault(index, []).append(path)
        missing = set(self._episodes) - set(locations)
        if missing:
            raise DatasetFormatError(f"trajectory rows missing for episodes {sorted(missing)}")
        return locations

    def episode(self, index: int) -> Episode:
        if index not in self._episodes:
            raise DatasetFormatError(f"episode {index} not found")
        metadata = self._episodes[index]
        rows = []
        for data_path in self._data_paths[index]:
            rows.extend(
                row
                for row in pq.read_table(data_path).to_pylist()
                if int(row["episode_index"]) == index
            )
        rows.sort(key=lambda row: int(row["frame_index"]))
        if len(rows) != int(_field(metadata, "length")):
            raise DatasetFormatError(
                f"episode {index} expected {_field(metadata, 'length')} rows, found {len(rows)}"
            )
        expected_frames = list(range(len(rows)))
        if [int(row["frame_index"]) for row in rows] != expected_frames:
            raise DatasetFormatError(f"episode {index} frame_index values must be contiguous from zero")
        video_slices = {view: self._video_slice(metadata, view, index) for view in self.VIEWS}
        return Episode(
            index=index,
            frame_indices=np.asarray([row["frame_index"] for row in rows], dtype=np.int64),
            dataset_indices=np.asarray([row["index"] for row in rows], dtype=np.int64),
            timestamps=np.asarray([row["timestamp"] for row in rows], dtype=float),
            state_deg=np.asarray([row["observation.state"] for row in rows], dtype=float),
            action_deg=np.asarray([row["action"] for row in rows], dtype=float),
            video_slices=video_slices,
        )

    def _video_slice(self, metadata: dict[str, Any], view: str, episode: int) -> VideoSlice:
        prefix = f"videos/observation.images.{view}"
        chunk = int(_field(metadata, f"{prefix}/chunk_index"))
        file_index = int(_field(metadata, f"{prefix}/file_index"))
        path = self.root / f"videos/observation.images.{view}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        if not path.is_file():
            raise DatasetFormatError(f"episode {episode} references missing {view} video {path}")
        return VideoSlice(
            path=path,
            start_time=float(_field(metadata, f"{prefix}/from_timestamp")),
            end_time=float(_field(metadata, f"{prefix}/to_timestamp")),
        )
