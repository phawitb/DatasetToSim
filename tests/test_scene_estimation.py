import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from dataset_to_sim.scene_estimation import fit_cube_silhouette, project_cube


def test_cube_pose_from_initial_rgb_is_metric_and_respects_support_plane():
    camera={'position':[0,0,.6],'quaternion':[1,0,0,0],'fovy':45.}
    truth=np.array([.07,-.04,.41]);edge=.0195;support=.005
    polygon=project_cube(camera,truth,edge,support)
    rgb=np.full((480,640,3),75,dtype=np.uint8)
    cv2.fillConvexPoly(rgb,np.int32(np.round(polygon)),(210,40,50))
    x,y,w,h=cv2.boundingRect(polygon.astype(np.float32));scene={'cube_bbox':[x,y,w,h,w*h]}
    pose,report=fit_cube_silhouette(rgb,camera,scene,truth+[.004,-.003,.15],edge,support)
    np.testing.assert_allclose(pose[:2],truth[:2],atol=.001)
    assert abs(pose[2]-truth[2])<.07
    assert report['boundary_rmse_px']<1.


def test_bright_top_face_disambiguates_cube_yaw_from_outer_silhouette():
    camera={'position':[0,-.2,.6],'quaternion':Rotation.from_euler('x',.32).as_quat()[[3,0,1,2]].tolist(),'fovy':45.}
    truth=np.array([.06,.02,.42]);edge=.0195;support=.003
    hull=project_cube(camera,truth,edge,support);rgb=np.full((480,640,3),70,dtype=np.uint8)
    cv2.fillConvexPoly(rgb,np.int32(np.round(hull)),(125,20,30))
    local=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*edge/2
    top=np.c_[local,np.full(4,edge)]@Rotation.from_euler('z',truth[2]).as_matrix().T+[truth[0],truth[1],support]
    r=Rotation.from_quat(np.array(camera['quaternion'])[[1,2,3,0]]).as_matrix();p=(top-camera['position'])@r@np.diag([1,-1,-1]);f=240/np.tan(np.deg2rad(camera['fovy']/2));uv=p[:,:2]/p[:,2:]*f+[320,240]
    cv2.fillConvexPoly(rgb,np.int32(np.round(uv)),(225,50,65))
    x,y,w,h=cv2.boundingRect(hull.astype(np.float32));scene={'cube_bbox':[x,y,w,h,w*h]}
    pose,report=fit_cube_silhouette(rgb,camera,scene,[truth[0],truth[1],np.pi/2-truth[2]],edge,support,use_top_face=True)
    delta=np.angle(np.exp(4j*(pose[2]-truth[2])))/4
    assert abs(delta)<.09
    assert report['top_face_orientation']['used']
