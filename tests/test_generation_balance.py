import sys,json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_generation_balance import task_requests,BalancedLayouts
from studio_generation import validate_request

def test_total_quota_counts_existing_by_task_without_overshooting():
 episodes=[{'status':'accepted','task_id':'red_cube_to_green_cube'}]*40+[{'status':'accepted','task_id':'green_cube_to_red_cube'}]*5
 options=validate_request({'tasks':['red_cube_to_green_cube','green_cube_to_red_cube'],'episodes_per_task':300,'target_counts':{'red_cube_to_green_cube':300,'green_cube_to_red_cube':300}})
 assert task_requests(options,episodes)=={'red_cube_to_green_cube':260,'green_cube_to_red_cube':295}
 assert task_requests(options|{'target_counts':None},episodes)=={'red_cube_to_green_cube':300,'green_cube_to_red_cube':300}
 with pytest.raises(ValueError):validate_request({'target_counts':{'unstack':-1}})

def test_balancer_counts_acceptance_not_attempts_and_meets_both_marginals():
 world={'mat':{'position':[.23,0,0],'size':[.405,.405,.002]}}
 b=BalancedLayouts(world,seed=12,target=300)
 for i in range(300):
  choice=b.choose()
  b.accept_cells(choice['source_cell'],choice['target_cell'])
 assert max(b.source)-min(b.source)<=1
 assert max(b.target)-min(b.target)<=1
 assert sum(b.source)==sum(b.target)==300
 with pytest.raises(ValueError):b.choose()

def test_rejections_change_pair_not_coverage():
 world={'mat':{'position':[.23,0,0],'size':[.405,.405,.002]}}
 b=BalancedLayouts(world,seed=5,target=300)
 a=b.choose();b.reject(a);new=b.choose()
 assert a!=new
 assert sum(b.source)==sum(b.target)==0


def test_balanced_requires_layout_randomization():
    with pytest.raises(ValueError, match='Randomize'):
        validate_request({'layout_strategy':'balanced','randomize_layout':False})

def test_balancing_never_leaves_only_an_identical_cell_pair_when_avoidable():
    world={'mat':{'position':[.23,0,0],'size':[.405,.405,.002]}}
    for seed in range(30):
        b=BalancedLayouts(world,seed=seed,target=9)
        b.source[:]=1;b.source[[0,1]]=0
        b.target[:]=1;b.target[[0,2]]=0
        for _ in range(2):
            choice=b.choose()
            assert choice['source_cell']!=choice['target_cell']
            b.accept_cells(choice['source_cell'],choice['target_cell'])
