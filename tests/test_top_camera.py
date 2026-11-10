import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import top_camera as T  # noqa: E402


def _reference():
    try:
        return T.reference()
    except Exception as exc:  # the recorded dataset lives outside the repository
        pytest.skip(f'dataset reference unavailable: {exc}')


def test_dataset_view_matches_itself():
    image = _reference()
    A, result = T.measure(image)
    assert A is not None and result['moved_px'] < .2


def test_moved_camera_is_lined_up_with_the_dataset_view(tmp_path):
    image = _reference()
    shifted = cv2.warpAffine(image, cv2.getRotationMatrix2D((320, 240), .6, 1.)+np.float32([[0, 0, 7], [0, 0, -4]]), (640, 480), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(tmp_path / 'top.jpg'), shifted)
    result = T.align(tmp_path)
    assert result['aligned'] and result['moved_px'] > 5
    assert result['residual_px'] < 1.
    assert (tmp_path / 'top-raw.jpg').exists()


def test_camera_moved_too_far_blocks(tmp_path):
    image = _reference()
    cv2.imwrite(str(tmp_path / 'top.jpg'), cv2.warpAffine(image, np.float32([[1, 0, 40], [0, 1, 0]]), (640, 480), borderMode=cv2.BORDER_REPLICATE))
    result = T.align(tmp_path)
    assert result['ok'] is False
