import xml.etree.ElementTree as ET

from dataset_to_sim.calibration import Calibration
from dataset_to_sim.scene import build_scene_xml
from tests.test_calibration import valid_config


def test_scene_adds_observed_workspace_and_calibrated_top_camera(tmp_path):
    robot_xml = tmp_path / "so101.xml"
    robot_xml.write_text(
        "<mujoco model='test'><compiler/><worldbody><body name='base'/></worldbody>"
        "<actuator><position name='shoulder_pan'/><position name='shoulder_lift'/>"
        "<position name='elbow_flex'/><position name='wrist_flex'/>"
        "<position name='wrist_roll'/><position name='gripper'/></actuator></mujoco>"
    )
    calibration_path = tmp_path / "calibration.json"
    import json
    calibration_path.write_text(json.dumps(valid_config()))

    root = ET.fromstring(build_scene_xml(Calibration.load(calibration_path), robot_xml))

    assert root.find(".//geom[@name='table']") is not None
    assert root.find(".//geom[@name='work_mat']") is not None
    assert root.find(".//body[@name='cube']") is not None
    assert root.find(".//body[@name='bowl']") is not None
    top = root.find(".//camera[@name='top_cam']")
    assert top is not None
    assert top.get("fovy") == "62.0"
    assert root.find("compiler").get("meshdir") == str((tmp_path / "assets").resolve())
