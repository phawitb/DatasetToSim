import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_home import validate_occluded_reference


def test_hidden_bowl_requires_recent_completed_matching_execution():
    old={'captured_at':100};actual={'completed':True,'samples':[{'measured':[0]*6}]}
    validate_occluded_reference(old,actual,[0]*6,now=150)
    for changed,pose,now in [(actual,[11,0,0,0,0,0],150),(actual,[0]*6,701),({'completed':False,'samples':actual['samples']},[0]*6,150),({},[0]*6,150)]:
        with pytest.raises(ValueError):validate_occluded_reference(old,changed,pose,now=now)
