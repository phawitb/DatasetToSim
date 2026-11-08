import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import studio_generation_assets as assets


def png(size=(800, 600), color=(40, 90, 100)):
    out=io.BytesIO();Image.new('RGB',size,color).save(out,format='PNG');return out.getvalue()


def test_upload_is_normalized_deduplicated_and_snapshotted(tmp_path, monkeypatch):
    monkeypatch.setattr(assets,'ASSET_ROOT',tmp_path/'library')
    a=assets.upload_background(png(),'../my-background.png')
    b=assets.upload_background(png(),'second.png')
    assert a['id']==b['id'] and a['name']=='my-background.png'
    with Image.open(a['path']) as image:assert image.size==(640,480) and image.mode=='RGB'
    assert len(assets.list_backgrounds())==2
    frozen=assets.snapshot_background(a['id'],tmp_path/'job'/'assets')
    assert Path(frozen['path']).read_bytes()==Path(a['path']).read_bytes()
    assert Path(frozen['path']).parent==tmp_path/'job'/'assets'
    assert frozen['mode']=='uploaded'


def test_upload_rejects_bad_large_or_animated_input(tmp_path, monkeypatch):
    monkeypatch.setattr(assets,'ASSET_ROOT',tmp_path)
    with pytest.raises(ValueError,match='image'):assets.upload_background(b'<svg/>','test.png')
    monkeypatch.setattr(assets,'MAX_UPLOAD_BYTES',10)
    with pytest.raises(ValueError,match='12 MB'):assets.upload_background(png(),'a.png')
    monkeypatch.setattr(assets,'MAX_UPLOAD_BYTES',12*1024**2)
    monkeypatch.setattr(assets,'MAX_PIXELS',100)
    with pytest.raises(ValueError,match='pixels'):assets.upload_background(png((11,10)),'a.png')
    assert not list(tmp_path.glob('*.json'))


def test_resolver_rejects_paths_and_modified_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(assets,'ASSET_ROOT',tmp_path)
    for identifier in ['../secret','/tmp/a.png','bg-x']:
        with pytest.raises(ValueError):assets.resolve_background(identifier)
    a=assets.upload_background(png(),'a.png')
    Path(a['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):assets.resolve_background(a['id'])


def test_public_catalog_never_exposes_filesystem_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(assets,'ASSET_ROOT',tmp_path)
    assets.upload_background(png(),'a.png')
    for item in assets.list_backgrounds():
        assert 'path' not in item
        assert item['url'].startswith('/api/generation-assets/')
