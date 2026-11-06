import importlib.util
from pathlib import Path
import numpy as np
import cv2

spec=importlib.util.spec_from_file_location('episode0',Path(__file__).parents[1]/'scripts/episode0.py')
recipe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)

def test_mat_calibration_reprojects_corners_within_eight_pixels():
    K,R,p,xyz,uv=recipe.camera_fit()
    projected=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],-R@p,K,None)[0].reshape(-1,2)
    assert np.max(np.linalg.norm(projected-uv,axis=1))<8

def test_image_ray_backprojection_returns_requested_plane():
    K,R,p,_,_=recipe.camera_fit()
    point=recipe.ground_point([401,243],K,R,p,.016)
    assert abs(point[2]-.016)<1e-10
    projected=cv2.projectPoints(point[None],cv2.Rodrigues(R)[0],-R@p,K,None)[0].reshape(2)
    np.testing.assert_allclose(projected,[401,243],atol=1e-7)
