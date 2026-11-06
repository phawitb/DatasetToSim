"""Command-line workflow for inspecting, replaying and evaluating SIMCUBE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .calibration import Calibration
from .dataset import DatasetIndex
from .replay import ReplaySession
from .report import decode_video_frame, evaluate_calibration, image_metrics, save_contact_sheet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = PROJECT_ROOT / "calibration/simcube-draft.json"
DEFAULT_ROBOT_XML = Path("/Users/phawit/Projects/SO-101 MuJoCo 3D Manipulation Scene Editor/assets/so101/so101.xml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-to-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="validate and summarize a LeRobot v3 dataset")
    inspect.add_argument("--dataset", type=Path, required=True)
    for name, help_text in (("replay", "render one simulator frame"), ("report", "render and compare one real/sim frame")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--episode", type=int, default=0)
        command.add_argument("--frame", type=int, default=0, help="zero-based frame within episode")
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        command.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
        command.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
        command.add_argument("--mode", choices=("kinematic", "pd"), default="kinematic")
        command.add_argument("--no-render", action="store_true", help="validate simulation state without creating OpenGL frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = DatasetIndex(args.dataset)
    if args.command == "inspect":
        print(json.dumps({"dataset": str(dataset.root), "episode_count": dataset.episode_count, "fps": dataset.fps, "task": dataset.info.get("total_tasks")}, indent=2))
        return 0
    calibration = Calibration.load(args.calibration)
    if args.command == "report" and args.no_render:
        raise ValueError("report requires rendering; remove --no-render in a desktop graphics session")
    episode = dataset.episode(args.episode)
    with ReplaySession(episode, calibration, args.robot_xml) as session:
        replay = (
            session.kinematic_frame(args.frame, render=not args.no_render)
            if args.mode == "kinematic"
            else session.pd_frame(args.frame, render=not args.no_render)
        )
    frame_dir = args.output / f"episode-{args.episode:03d}-frame-{args.frame:04d}-{args.mode}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for view, image in replay.images.items():
        _write_rgb(frame_dir / f"sim-{view}.png", image)
    if args.command == "replay":
        print(json.dumps({"mode": replay.mode, "timestamp": replay.timestamp, "output": str(frame_dir), "calibration_status": calibration.status}, indent=2))
        return 0
    metrics_by_view = {}
    for view, simulation in replay.images.items():
        reference = decode_video_frame(episode.video_slices[view], replay.timestamp)
        metrics_by_view[view] = image_metrics(reference, simulation)
        _write_rgb(frame_dir / f"real-{view}.png", reference)
        save_contact_sheet(reference, simulation, frame_dir / f"compare-{view}.png", label=view)
    payload = {
        "episode": args.episode,
        "frame": args.frame,
        "mode": replay.mode,
        "timestamp": replay.timestamp,
        "calibration": str(calibration.path),
        "calibration_status": calibration.status,
        "metrics": metrics_by_view,
        "verdict": evaluate_calibration(calibration, metrics_by_view),
    }
    (frame_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


def _write_rgb(path: Path, image) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write image: {path}")
