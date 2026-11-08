"""Local, immutable image library for generated workcell backgrounds."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import threading
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / 'output/generation-assets'
REFERENCE = ROOT / 'calibration/real-appearance-v1/background.png'
MAX_UPLOAD_BYTES = 12 * 1024**2
MAX_PIXELS = 20_000_000
_LOCK = threading.Lock()


def public_background(item):
    return {key: value for key, value in item.items() if key != 'path'} | {
        'url': f'/api/generation-assets/{item["id"]}/image'}


def resolve_background(identifier):
    if identifier == 'calibrated-real':
        if not REFERENCE.is_file():
            raise ValueError('The calibrated workcell background is unavailable')
        return {'id': identifier, 'name': 'Calibrated real workcell',
                'mode': 'calibrated', 'path': str(REFERENCE),
                'sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                'note': 'Foreground-cleaned real workcell; calibrated Top and moving Wrist projection.'}
    if not isinstance(identifier, str) or not re.fullmatch(r'bg-[a-f0-9]{64}', identifier):
        raise ValueError('Choose a background from the image library')
    path = ASSET_ROOT / (identifier + '.png')
    metadata = ASSET_ROOT / (identifier + '.json')
    if not path.is_file() or not metadata.is_file() or path.is_symlink() or metadata.is_symlink():
        raise ValueError('Background image not found')
    item = json.loads(metadata.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item.get('sha256') or identifier != 'bg-' + digest:
        raise ValueError('Stored background image changed; upload it again')
    return item | {'path': str(path.resolve())}


def list_backgrounds():
    items = [public_background(resolve_background('calibrated-real'))]
    for metadata in sorted(ASSET_ROOT.glob('bg-*.json')):
        try:
            items.append(public_background(resolve_background(metadata.stem)))
        except (ValueError, OSError):
            continue
    return items


def upload_background(content, filename):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError('Choose an image up to 12 MB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in ('PNG', 'JPEG', 'WEBP'):
                    raise ValueError('Choose a PNG, JPEG or WebP image')
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError('Image exceeds 20 million pixels')
                if getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('Choose a still image, not an animation')
                oriented = ImageOps.exif_transpose(source)
                rgba = oriented.convert('RGBA')
                clean = Image.new('RGBA', rgba.size, (235, 235, 235, 255))
                clean.alpha_composite(rgba)
                clean = ImageOps.fit(clean.convert('RGB'), (640, 480), method=Image.Resampling.LANCZOS)
                buffer = io.BytesIO(); clean.save(buffer, format='PNG')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Cannot read this image; choose a valid PNG, JPEG or WebP') from exc
    normalized = buffer.getvalue()
    digest = hashlib.sha256(normalized).hexdigest(); identifier = 'bg-' + digest
    name = Path(str(filename).replace('\\', '/')).name
    name = ''.join(c for c in name if c.isprintable()).strip()[:120] or 'Uploaded background'
    item = {'id': identifier, 'name': name, 'mode': 'uploaded', 'sha256': digest,
            'source_sha256': hashlib.sha256(content).hexdigest(), 'width': 640, 'height': 480,
            'note': 'Uploaded image cropped to 4:3 and mapped to the workcell plane; not camera-calibrated. Use an empty background to avoid duplicate objects.'}
    with _LOCK:
        ASSET_ROOT.mkdir(parents=True, exist_ok=True)
        path = ASSET_ROOT / (identifier + '.png'); metadata = path.with_suffix('.json')
        if not path.exists() or not metadata.exists():
            temporary = path.with_suffix('.png.tmp'); temporary.write_bytes(normalized); temporary.replace(path)
            temporary = metadata.with_suffix('.json.tmp'); temporary.write_text(json.dumps(item, indent=2)); temporary.replace(metadata)
    return resolve_background(identifier)


def snapshot_background(identifier, folder):
    item = resolve_background(identifier)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'background.png'
    shutil.copyfile(item['path'], destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('Background changed while preparing the job')
    snapshot = item | {'path': str(destination.resolve())}
    (folder / 'background.json').write_text(json.dumps(snapshot, indent=2))
    return snapshot
