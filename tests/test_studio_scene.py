import sys
from pathlib import Path
import cv2,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_scene import detect_bowl

@pytest.mark.parametrize('x',[210,470])
def test_live_bowl_detection_covers_both_sides_of_mat(x):
    rgb=np.full((480,640,3),40,np.uint8)
    cv2.ellipse(rgb,(x,185),(60,55),0,0,360,(140,165,160),-1)
    assert np.linalg.norm(np.array(detect_bowl(rgb))-[x,185])<1

def test_live_bowl_rejects_missing_object():
    with pytest.raises(ValueError,match='bowl'):detect_bowl(np.zeros((480,640,3),np.uint8))


def test_pale_bowl_interior_and_shadow_keep_stable_center():
    rgb=np.full((480,640,3),40,np.uint8)
    cv2.ellipse(rgb,(205,172),(62,58),0,0,360,(155,180,175),-1)
    # Interior can be warmer than the rim; blue need not exceed red.
    cv2.ellipse(rgb,(205,180),(48,43),0,0,360,(207,217,205),-1)
    assert np.linalg.norm(np.array(detect_bowl(rgb))-[205,172])<1


def test_occlusion_requires_visible_bowl_and_arm_not_just_detection_failure():
    from studio_scene import verify_bowl_occlusion
    frame=np.full((480,640,3),40,np.uint8)
    cv2.rectangle(frame,(185,110),(225,235),(240,140,60),-1)
    with pytest.raises(ValueError):verify_bowl_occlusion(frame,[205,172])
    with pytest.raises(ValueError):verify_bowl_occlusion(np.zeros_like(frame),[205,172])
    cv2.ellipse(frame,(205,172),(60,56),0,0,360,(140,170,160),-1)
    cv2.rectangle(frame,(185,110),(225,235),(240,140,60),-1)
    assert verify_bowl_occlusion(frame,[205,172])['visible_bowl_pixels']>450
