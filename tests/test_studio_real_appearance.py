from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_real_appearance import background_maps,camera_matrix,project_world,robust_stats


def test_top_camera_background_projection_is_pixel_aligned():
    c={'position':[0,0,1.],'quaternion':[1.,0,0,0],'fovy':50.}
    model=SimpleNamespace(camera=lambda name:SimpleNamespace(id=0),cam_fovy=np.array([50.]))
    data=SimpleNamespace(cam_xpos=np.array([[0,0,1.]]),cam_xmat=np.array([np.eye(3).ravel()]))
    x,y,valid=background_maps(data,model,'top',{'camera':c},.003)
    yy,xx=np.mgrid[:480,:640]
    np.testing.assert_allclose(x,xx,atol=1e-5);np.testing.assert_allclose(y,yy,atol=1e-5)
    assert valid[200,300]


def test_wrist_camera_motion_changes_the_background_projection():
    c={'position':[0,0,1.],'quaternion':[1.,0,0,0],'fovy':50.}
    model=SimpleNamespace(camera=lambda name:SimpleNamespace(id=0),cam_fovy=np.array([50.]))
    data=SimpleNamespace(cam_xpos=np.array([[0.,0,1.]]),cam_xmat=np.array([np.eye(3).ravel()]))
    a=background_maps(data,model,'wrist',{'camera':c},0.)[0]
    data.cam_xpos[0,0]=.1
    b=background_maps(data,model,'wrist',{'camera':c},0.)[0]
    assert b[240,320]>a[240,320]+40


def test_color_statistics_are_finite_for_flat_surfaces():
    stats=robust_stats(np.tile([100,120,80],(200,1)))
    assert np.isfinite(stats['mean']).all()
    assert min(stats['std'])>=1


def test_background_completion_does_not_mix_materials_or_keep_foreground():
    from studio_real_appearance import complete_background
    rgb=np.full((240,320,3),220,dtype='uint8');mat=np.zeros((240,320),bool);mat[:,80:240]=True;rgb[mat]=60
    valid=np.ones((240,320),bool);valid[70:110,135:165]=False;rgb[~valid]=[255,0,0]
    clean,observed,_=complete_background(rgb,valid,mat)
    np.testing.assert_allclose(clean[90,150],60,atol=1)
    np.testing.assert_allclose(clean[90,60],220,atol=1)
    assert not observed[90,150]


def test_background_only_preserves_robot_and_objects():
    from studio_real_appearance import RealAppearance
    a=object.__new__(RealAppearance);a.background=True;a.processing=False;a.classes={0:'robot',1:'red'}
    rgb=np.tile(np.array([255,50,20],dtype='uint8'),(12,12,1));ids=np.zeros((12,12),int);ids[:,6:]=1
    np.testing.assert_array_equal(a.apply(rgb,ids,None,'top'),rgb)


def test_wrist_floor_uses_table_reference_when_top_cannot_see_floor():
    from studio_real_appearance import RealAppearance
    from unittest.mock import patch
    stats={'mean':[100.,128.,128.],'std':[10.,10.,10.]}
    a=object.__new__(RealAppearance);a.background=True;a.processing=True;a.classes={0:'floor'};a.maps={}
    a.plate=np.full((480,640,3),100,dtype='uint8')
    a.model=SimpleNamespace(geom_size=np.zeros((1,3)),geom_type=np.zeros(1,int))
    a.profile={'appearance':{'top':{'table':{'real':stats}},'wrist':{'floor':{'real':stats,'sim':stats}}},'surface_rgb':{'table':[100,100,100]}}
    data=SimpleNamespace(geom_xpos=np.zeros((1,3)));rgb=a.plate.copy();ids=np.zeros((480,640),int)
    yy,xx=np.mgrid[:480,:640]
    with patch('studio_real_appearance.background_maps',return_value=(xx.astype('float32'),yy.astype('float32'),np.ones((480,640),bool))):
        result=a.apply(rgb,ids,data,'wrist')
    assert result.shape==rgb.shape
    assert np.isfinite(result).all()


def test_plane_grid_spacing_is_not_surface_height():
    from studio_real_appearance import surface_height
    model=SimpleNamespace(geom_type=np.array([0,6]),geom_size=np.array([[3,3,.01],[1,1,.4]]))
    data=SimpleNamespace(geom_xpos=np.array([[0,0,-.8],[0,0,-.399]]))
    assert surface_height(model,data,0)==-.8
    assert abs(surface_height(model,data,1)-.001)<1e-12
