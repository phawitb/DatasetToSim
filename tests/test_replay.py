import json

import numpy as np

from dataset_to_sim.calibration import Calibration
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.replay import ReplaySession
from tests.test_calibration import valid_config
from tests.test_dataset import write_dataset


def write_robot(path):
    joints = "".join(
        f"<joint name='{name}' type='hinge' axis='0 0 1' range='-3.14 3.14'/>"
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    )
    actuators = "".join(
        f"<position name='{name}' joint='{name}' kp='50' ctrlrange='-3.14 3.14'/>"
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    )
    path.write_text(
        f"<mujoco model='test'><compiler angle='radian'/><option timestep='.001'/><worldbody>"
        f"<body name='base'>{joints}<geom type='sphere' size='.05' mass='1'/>"
        f"<camera name='wrist_cam' pos='0 0 1'/></body>"
        f"</worldbody><actuator>{actuators}</actuator></mujoco>"
    )


def make_session(tmp_path):
    dataset_root = tmp_path / "dataset"
    write_dataset(dataset_root)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(valid_config()))
    robot_xml = tmp_path / "so101.xml"
    write_robot(robot_xml)
    return ReplaySession(DatasetIndex(dataset_root).episode(0), Calibration.load(calibration_path), robot_xml)


def test_kinematic_replay_sets_exact_converted_joint_state_and_timestamp(tmp_path):
    session = make_session(tmp_path)

    result = session.kinematic_frame(1, render=False)

    np.testing.assert_allclose(result.qpos, np.deg2rad([1, 11, 21, 31, 41, 51]))
    assert result.timestamp == 1 / 30
    assert result.mode == "kinematic"
    assert result.images == {}


def test_pd_replay_is_separately_labelled_and_advances_to_requested_time(tmp_path):
    session = make_session(tmp_path)

    result = session.pd_frame(1, render=False)

    assert result.mode == "pd"
    assert result.timestamp == 1 / 30
