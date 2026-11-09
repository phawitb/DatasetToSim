"""Fully automatic workcell calibration for an imported SO-101 LeRobot dataset.

Robot, wrist camera, joint/gripper adapter and physics come unchanged from the
shared strict-replay configuration. Only the scene that changes between
recording setups is fitted, from the dataset images alone:

1. Top camera extrinsics + field of view: render-and-compare of the robot's
   orange printed parts against colour masks of the real frames, at several
   recorded joint states. Multi-start CMA-ES on a symmetric chamfer distance,
   then a silhouette-IoU refinement.
2. Mat: the dark work surface is segmented, back-projected through the fitted
   camera onto the table plane, and a rectangle (size, centre, yaw) is fitted.
3. Objects: every model the user picked from the Model library is located in
   the first frame of each episode by segmenting bright, non-robot blobs on the
   mat and fitting the model's projected silhouette (x, y, yaw).

No manual clicks, no per-episode tuning and no trajectory or outcome inputs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from dataset_to_sim.calibration import Calibration  # noqa: E402
from dataset_to_sim.strict_replay import NativeJointAdapter  # noqa: E402

BASE_CONFIG = ROOT / 'calibration/native-action-current.json'
ORANGE_RGBA = np.array([1, .38, .055])


# ---------------------------------------------------------------- image masks
def robot_mask(rgb):
    """Orange printed SO-101 parts. Saturation/value floors reject warm-lit mats and tables."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 0] > 3) & (hsv[..., 0] < 26) & (hsv[..., 1] > 125) & (hsv[..., 2] > 150)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask.astype(bool)


def read_frames(episode, view, indices):
    """Decode selected frames of one episode view (RGB)."""
    wanted = sorted(set(int(i) for i in indices)); out = {}
    cap = cv2.VideoCapture(str(episode.video_slices[view].path))
    start = round(episode.video_slices[view].start_time * 30)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(wanted[-1] + 1):
        ok, bgr = cap.read()
        if not ok: raise RuntimeError(f'Incomplete {view} video for episode {episode.index}')
        if f in wanted: out[f] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return [out[int(i)] for i in indices]


# ------------------------------------------------------------- camera algebra
def look_at(params):
    """(target x, y, distance, azimuth, elevation, roll, fovy) -> position, MuJoCo quat wxyz, fovy."""
    tx, ty, dist, az, el, roll, fovy = params
    target = np.array([tx, ty, 0.])
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    position = target + dist * direction
    back = direction  # camera +Z points away from the scene
    right = np.cross([0, 0, 1.], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    R = np.column_stack([right, up, back]) @ Rotation.from_rotvec([0, 0, roll]).as_matrix()
    return position, Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]], float(fovy)


def camera_matrices(camera, width, height):
    """World->OpenCV camera rotation, position and intrinsics for a MuJoCo camera dict."""
    R_cam = Rotation.from_quat(np.array(camera['quaternion'])[[1, 2, 3, 0]]).as_matrix()
    R = np.diag([1, -1, -1]) @ R_cam.T
    f = height / 2 / np.tan(np.deg2rad(camera['fovy'] / 2))
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.]])
    return K, R, np.asarray(camera['position'], float)


def project(points, camera, width, height):
    K, R, p = camera_matrices(camera, width, height)
    local = (np.asarray(points) - p) @ R.T
    uv = local @ K.T
    return uv[:, :2] / uv[:, 2:3], local[:, 2]


def backproject(uv, camera, width, height, z):
    K, R, p = camera_matrices(camera, width, height)
    uv = np.atleast_2d(np.asarray(uv, float))
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T @ R
    scale = (z - p[2]) / rays[:, 2]
    return p + rays * scale[:, None]


# ----------------------------------------------------------------- CMA-ES
def cma_es(cost, mean, sigma, lower, upper, iterations=60, population=None, seed=0):
    """Small (mu/mu_w, lambda) CMA-ES with box clipping; returns best x and value."""
    rng = np.random.default_rng(seed); n = len(mean)
    lam = population or 4 + int(3 * np.log(n)); mu = lam // 2
    w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); w /= w.sum(); mueff = 1 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n); cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff); cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs; chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    scale = np.asarray(sigma, float); m = (np.asarray(mean, float) - lower) / scale
    lo, hi = np.zeros(n), (upper - lower) / scale
    C = np.eye(n); pc = np.zeros(n); ps = np.zeros(n); s = 1.
    best, best_x = np.inf, None
    for generation in range(iterations):
        vals, eig = np.linalg.eigh(C); vals = np.maximum(vals, 1e-20); B = eig; D = np.sqrt(vals)
        z = rng.standard_normal((lam, n)); y = z * D @ B.T; x = np.clip(m + s * y, lo, hi)
        y = (x - m) / s
        f = np.array([cost(lower + xi * scale) for xi in x]); order = np.argsort(f)
        if f[order[0]] < best: best, best_x = float(f[order[0]]), lower + x[order[0]] * scale
        yw = w @ y[order[:mu]]; m = m + s * yw
        invsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrt @ yw
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        artmp = y[order[:mu]]
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C) + cmu * (artmp.T * w) @ artmp
        s *= np.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        if s * np.sqrt(vals.max()) < 1e-4: break
    return best_x, best


# ------------------------------------------------------------ robot renderer
def load_config(path=BASE_CONFIG):
    return json.loads(Path(path).read_text())


def robot_model(config):
    """Strict replay model of the shared robot (objects are irrelevant for the camera fit)."""
    from strict_action_scene import build_model
    data = copy.deepcopy(config['base_calibration'])
    data['world']['cube']['position'] = [2., 2., .05]; data['world']['bowl']['position'] = [2.5, 2.5, 0.]
    return build_model(Calibration(ROOT / 'calibration/import.json', data), config)


class RobotSilhouette:
    """Renders the robot's orange printed parts from a candidate top camera."""

    def __init__(self, model, width, height):
        self.model, self.width, self.height = model, width, height
        self.data = mujoco.MjData(model)
        self.camera = model.camera('top_cam').id
        orange = np.linalg.norm(model.mat_rgba[:, :3] - ORANGE_RGBA, axis=1) < .02
        geom_orange = np.zeros(model.ngeom, bool)
        for g in range(model.ngeom):
            if model.geom_matid[g] >= 0: geom_orange[g] = orange[model.geom_matid[g]]
            else: geom_orange[g] = np.linalg.norm(model.geom_rgba[g, :3] - ORANGE_RGBA) < .02
        self.orange_geoms = np.flatnonzero(geom_orange & (model.geom_group <= 2))
        self.renderer = mujoco.Renderer(model, height, width)
        self.renderer.enable_segmentation_rendering()
        self.option = mujoco.MjvOption(); self.option.geomgroup[3:] = 0; self.option.sitegroup[:] = 0

    def set_camera(self, position, quat, fovy):
        self.model.cam_pos[self.camera] = position; self.model.cam_quat[self.camera] = quat; self.model.cam_fovy[self.camera] = fovy

    def render(self, qpos):
        self.data.qpos[:6] = qpos; mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        seg = self.renderer.render()
        ids = seg[..., 0]; types = seg[..., 1]
        return (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(ids, self.orange_geoms)

    def points(self, qpos, limit=4000):
        """World-space mesh vertices of the orange parts at a joint state."""
        self.data.qpos[:6] = qpos; mujoco.mj_forward(self.model, self.data); clouds = []
        for g in self.orange_geoms:
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
            mesh = self.model.geom_dataid[g]; start = self.model.mesh_vertadr[mesh]
            v = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mesh]]
            clouds.append(v @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g])
        cloud = np.concatenate(clouds)
        return cloud[np.random.default_rng(0).choice(len(cloud), min(limit, len(cloud)), replace=False)]

    def close(self): self.renderer.close()


def select_frames(dataset, adapter, count=10, per_episode=40):
    """Low-velocity frames with diverse arm poses across all episodes (farthest-point sampling)."""
    candidates = []
    for index in range(dataset.episode_count):
        ep = dataset.episode(index); q = ep.state_deg
        speed = np.r_[np.inf, np.linalg.norm(np.diff(q[:, :5], axis=0), axis=1)]
        speed = np.maximum(speed, np.r_[speed[1:], np.inf])
        still = np.flatnonzero(speed < 1.0)
        if len(still) > per_episode: still = still[np.linspace(0, len(still) - 1, per_episode).astype(int)]
        candidates += [(index, int(f), q[f]) for f in still]
    if not candidates: raise ValueError('No stationary robot frames found for camera calibration')
    poses = np.array([c[2][:5] for c in candidates])
    chosen = [int(np.argmin(np.abs(poses[:, 1] - poses[:, 1].max())))]
    distance = np.linalg.norm(poses - poses[chosen[0]], axis=1)
    while len(chosen) < min(count, len(candidates)):
        nxt = int(np.argmax(distance)); chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(poses - poses[nxt], axis=1))
    return [(candidates[i][0], candidates[i][1], candidates[i][2]) for i in chosen]


def fit_top_camera(dataset, config, frames=None, starts=12, log=print, scene_score=None, fov_grid=(30., 35., 42., 50.)):
    """Return the fitted top camera dict and a report with per-frame mask IoU."""
    adapter = NativeJointAdapter(**config['adapter'])
    frames = frames or select_frames(dataset, adapter)
    images = []
    for episode_index, frame, _ in frames:
        images.append(read_frames(dataset.episode(episode_index), 'top', [frame])[0])
    height, width = images[0].shape[:2]
    qpos = [adapter.to_model(native) for _, _, native in frames]
    targets = [robot_mask(im) for im in images]
    model = robot_model(config)

    def prepare(scale):
        w, h = int(width * scale), int(height * scale)
        masks = [cv2.resize(t.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0 for t in targets]
        transforms = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) / np.hypot(w, h) for m in masks]
        return w, h, masks, transforms

    def chamfer_cost(renderer, masks, transforms, params):
        renderer.set_camera(*look_at(params)); total = 0.
        for q, target, transform in zip(qpos, masks, transforms):
            sim = renderer.render(q)
            if sim.sum() < 5: total += 1.; continue
            sim_transform = cv2.distanceTransform((~sim).astype(np.uint8), cv2.DIST_L2, 3) / np.hypot(*sim.shape)
            total += transform[sim].mean() + sim_transform[target].mean()
        return total / len(qpos)

    def iou_cost(renderer, masks, params):
        renderer.set_camera(*look_at(params)); values = []
        for q, target in zip(qpos, masks):
            sim = renderer.render(q)
            values.append((sim & target).sum() / max(1, (sim | target).sum()))
        return 1 - float(np.mean(values)), values

    lower = np.array([-.3, -.6, .25, -np.pi, np.deg2rad(15), -np.pi, 25.])
    upper = np.array([.7, .6, 1.6, np.pi, np.deg2rad(89), np.pi, 90.])
    # Coarse stage: project mesh vertices of the orange parts (no rasterisation),
    # symmetric chamfer distance against the real masks. ~100x faster than rendering.
    w, h, masks, transforms = prepare(.5)
    probe = RobotSilhouette(model, w, h)
    try: clouds = [probe.points(q) for q in qpos]
    finally: probe.close()
    diagonal = np.hypot(w, h)
    real_pixels = [np.c_[np.nonzero(m)[1], np.nonzero(m)[0]] for m in masks]

    def point_cost(params):
        position, quat, fovy = look_at(params)
        camera = {'position': position, 'quaternion': quat, 'fovy': fovy}; total = 0.
        for cloud, transform, pixels in zip(clouds, transforms, real_pixels):
            uv, depth = project(cloud, camera, w, h)
            inside = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
            if inside.sum() < 10: total += 1.; continue
            px = np.round(uv[inside]).astype(int)
            forward = transform[px[:, 1], px[:, 0]].mean() * inside.mean() + .08 * (1 - inside.mean())
            canvas = np.ones((h, w), np.uint8); canvas[px[:, 1], px[:, 0]] = 0
            canvas = cv2.erode(canvas, np.ones((3, 3), np.uint8))
            backward = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)[pixels[:, 1], pixels[:, 0]].mean() / diagonal
            total += forward + backward
        return total / len(clouds)

    rng = np.random.default_rng(0); results = []
    for k in range(starts):
        # Azimuth sampled on a grid so every viewing side is tried.
        start = np.array([rng.uniform(0, .4), rng.uniform(-.25, .25), rng.uniform(.5, 1.0), -np.pi + 2 * np.pi * k / starts,
                          np.deg2rad(rng.uniform(40, 75)), rng.uniform(-np.pi, np.pi), 50.])
        x, value = cma_es(point_cost, start, np.array([.1, .1, .15, .5, .25, .8, 10.]), lower, upper, iterations=150, population=16, seed=k)
        results.append((value, x)); log(f'camera start {k + 1}/{starts}: chamfer {value:.4f}')
    results.sort(key=lambda r: r[0])
    polished = []
    for value, x in results[:4]:
        x2, v2 = cma_es(point_cost, x, np.array([.02, .02, .03, .08, .04, .08, 2.]), lower, upper, iterations=100, population=16, seed=99)
        polished.append((v2, x2))
    polished.sort(key=lambda r: r[0]); best = polished[0][1]
    log(f'camera coarse best chamfer {polished[0][0]:.4f}')
    w, h, masks, _ = prepare(1.)
    fine = RobotSilhouette(model, w, h)
    try:
        best, value = cma_es(lambda p: iou_cost(fine, masks, p)[0], best,
                             np.array([.006, .006, .01, .02, .01, .02, .6]), lower, upper, iterations=50, population=12, seed=7)
        candidates = [(best, 1 - value)]
        if scene_score is not None:
            # Robot silhouettes barely constrain focal length vs distance. Scan the
            # field of view at constant image scale, refine each, and let the
            # measured library objects (bowl, eraser) decide together with the robot.
            for fov in fov_grid:
                if abs(fov - best[6]) < 2: continue
                start = best.copy(); start[2] *= np.tan(np.deg2rad(best[6] / 2)) / np.tan(np.deg2rad(fov / 2)); start[6] = fov
                lo, hi = start - [.05, .05, .15, .2, .1, .2, 1e-3], start + [.05, .05, .15, .2, .1, .2, 1e-3]
                x, v = cma_es(lambda p: iou_cost(fine, masks, p)[0], start, np.array([.006, .006, .01, .02, .01, .02, 1e-4]), lo, hi,
                              iterations=35, population=12, seed=int(fov))
                candidates.append((x, 1 - v))
            scored = []
            for x, robot in candidates:
                position, quat, fovy = look_at(x)
                objects = scene_score({'position': position, 'quaternion': quat, 'fovy': fovy})
                scored.append((robot + objects, robot, objects, x)); log(f'fov {x[6]:.1f} deg: robot IoU {robot:.3f}, objects IoU {objects:.3f}')
            scored.sort(key=lambda r: -r[0]); best = scored[0][3]
            fov_scan = [{'fovy': float(r[3][6]), 'robot_iou': float(r[1]), 'object_iou': float(r[2])} for r in scored]
        else:
            fov_scan = None
        final, per_frame = iou_cost(fine, masks, best)
        overlays = []
        for q, target, image in zip(qpos, masks, images):
            sim = fine.render(q); overlay = image.copy()
            overlay[sim & ~target] = [0, 120, 255]; overlay[target & ~sim] = [255, 0, 0]; overlay[sim & target] = [0, 220, 0]
            overlays.append(overlay)
    finally:
        fine.close()
    position, quat, fovy = look_at(best)
    camera = {'position': position.tolist(), 'quaternion': quat.tolist(), 'fovy': fovy,
              'target': [float(best[0]), float(best[1]), 0.], 'image_size': [width, height]}
    report = {'method': 'robot orange-part silhouettes; multi-start CMA-ES chamfer, IoU refinement, field-of-view scan scored with library-object silhouettes',
              'frames': [[int(e), int(f)] for e, f, _ in frames], 'per_frame_mask_iou': [float(v) for v in per_frame],
              'mean_mask_iou': float(np.mean(per_frame)), 'look_at_parameters': best.tolist(), 'fov_scan': fov_scan,
              'coarse_candidates': [[float(v), x.tolist()] for v, x in results]}
    return camera, report, overlays


# ---------------------------------------------------------------- mat fit
def dark_surface_mask(rgb):
    """Dark work mat versus a bright table: Otsu on value, closed over wires and objects."""
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2]
    threshold, _ = cv2.threshold(cv2.GaussianBlur(value, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (value < threshold).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if count < 2: raise ValueError('No dark work mat found in the top camera image')
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest, float(threshold)


def rectangle_corners(params, z):
    cx, cy, yaw, sx, sy = params
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]]) / 2
    return np.c_[local @ np.array([[c, s], [-s, c]]) + [cx, cy], np.full(4, z)]


def fit_mat(images, robot_masks, camera, thickness=.002, table_height=.001, log=print):
    """Rectangle (centre, yaw, size) on the table plane that best explains the dark-surface pixels.

    Pixels covered by the robot are ignored. Sides that leave the image are not
    observed: an area penalty then keeps the smallest consistent rectangle, and
    the report says which sides were seen.
    """
    height, width = images[0].shape[:2]
    masks = [dark_surface_mask(im)[0] for im in images]
    mat = np.mean(masks, axis=0) > .5
    occluded = np.mean([cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for m in robot_masks], axis=0) > .2
    valid = ~occluded
    top = table_height + thickness
    filled = cv2.morphologyEx(mat.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    ys, xs = np.nonzero(filled & valid)
    ground = backproject(np.c_[xs, ys], camera, width, height, top)[:, :2]
    rect = cv2.minAreaRect(ground.astype(np.float32))
    (cx, cy), (sx, sy), angle = rect
    start = np.array([cx, cy, np.deg2rad(angle), max(sx, .1), max(sy, .1)])
    weight = valid.astype(float); total = max(1., (filled & valid).sum())

    def predicted(params):
        uv, depth = project(rectangle_corners(params, top), camera, width, height)
        if np.any(depth <= 0): return None
        canvas = np.zeros((height, width), np.uint8)
        cv2.fillPoly(canvas, [np.round(uv * 8).astype(np.int32)], 1, shift=3)
        return canvas.astype(bool)

    def cost(params):
        poly = predicted(params)
        if poly is None: return 10.
        mismatch = ((poly ^ filled) * weight).sum() / total
        return mismatch + .02 * params[3] * params[4] / .16

    lower = np.array([cx - .3, cy - .3, start[2] - np.pi / 2, .1, .1]); upper = np.array([cx + .3, cy + .3, start[2] + np.pi / 2, 1.2, 1.2])
    best, value = cma_es(cost, start, np.array([.03, .03, .1, .05, .05]), lower, upper, iterations=120, population=16, seed=3)
    best, value = cma_es(cost, best, np.array([.005, .005, .015, .008, .008]), lower, upper, iterations=80, population=16, seed=4)
    uv, _ = project(rectangle_corners(best, top), camera, width, height)
    inside = [bool(0 <= u < width and 0 <= v < height and not occluded[int(v), int(u)]) for u, v in uv]
    poly = predicted(best)
    iou = float(((poly & filled) * weight).sum() / max(1, ((poly | filled) * weight).sum()))
    cx, cy, yaw, sx, sy = map(float, best)
    world = {'position': [cx, cy, table_height + thickness / 2], 'size': [sx, sy, thickness],
             'quaternion': [float(np.cos(yaw / 2)), 0., 0., float(np.sin(yaw / 2))], 'yaw_rad': yaw}
    report = {'method': 'dark-surface Otsu mask, robot pixels ignored, rectangle on table plane by CMA-ES',
              'visible_mask_iou': iou, 'corners_visible': inside, 'corners_image_px': uv.tolist(),
              'thickness_m': thickness, 'thickness_source': 'assumed (same as the measured 2 mm mat)',
              'size_observed': all(inside) or sum(inside) >= 2,
              'note': 'Sides outside the image or under the robot are not observed; their extent is the smallest rectangle consistent with the visible mat.'}
    log(f'mat {sx * 1000:.0f} x {sy * 1000:.0f} mm, yaw {np.rad2deg(yaw):.1f} deg, visible IoU {iou:.3f}, corners visible {sum(inside)}/4')
    return world, report, filled


# ------------------------------------------------------------- object fit
def library_objects():
    """Physical shapes of Model-library items that an imported scene can contain."""
    eraser = json.loads((ROOT / 'assets/models/eraser/eraser.json').read_text())
    tape = json.loads((ROOT / 'assets/models/tape/tape.json').read_text())
    measured = json.loads((ROOT / 'calibration/real-measurements.json').read_text())
    bowl = measured['bowl']; cube = measured['cube']['size_xyz']
    return {
        'eraser': {'name': 'Eraser', 'role': 'manipulated', 'shape': 'box', 'size_m': eraser['size_m'], 'mass_kg': eraser['mass_kg'],
                   'friction': eraser['friction']},
        'cube': {'name': 'Cube (19.5 mm)', 'role': 'manipulated', 'shape': 'box', 'size_m': cube, 'mass_kg': 600. * np.prod(cube)},
        'tape': {'name': 'Tape roll', 'role': 'manipulated', 'shape': 'cylinder', 'radius_m': tape['outer_diameter_m'] / 2,
                 'height_m': tape['width_m'], 'mass_kg': tape['mass_kg']},
        'green-bowl': {'name': 'Bowl (measured)', 'role': 'container', 'shape': 'frustum', 'bottom_radius_m': bowl['bottom_diameter'] / 2,
                       'top_radius_m': bowl['top_rim_diameter'] / 2, 'height_m': bowl['height']},
    }


def object_points(spec, n=48):
    """Outline vertices (object frame, z from 0 at the support) whose projected hull is the silhouette."""
    if spec['shape'] == 'box':
        sx, sy, sz = spec['size_m']
        return np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (0, sz)])
    angles = np.arange(n) * 2 * np.pi / n
    if spec['shape'] == 'cylinder':
        r, h = spec['radius_m'], spec['height_m']; rings = [(r, 0), (r, h)]
    else:
        rings = [(spec['bottom_radius_m'], 0), (spec['top_radius_m'], spec['height_m'])]
    return np.array([[r * np.cos(a), r * np.sin(a), z] for r, z in rings for a in angles])


def silhouette(spec, pose, support, camera, shape):
    x, y, yaw = pose; c, s = np.cos(yaw), np.sin(yaw)
    points = object_points(spec) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [x, y, support]
    uv, depth = project(points, camera, shape[1], shape[0])
    canvas = np.zeros(shape, np.uint8)
    if np.all(depth > 0):
        hull = cv2.convexHull(np.round(uv * 8).astype(np.int32))
        cv2.fillConvexPoly(canvas, hull, 1, shift=3)
    return canvas.astype(bool)


def object_candidates(rgb, mat_mask, exclusion):
    """Bright blobs on the dark mat that are not robot (or the robot's jaw cover)."""
    hull = np.zeros(mat_mask.shape, np.uint8)
    contours, _ = cv2.findContours(mat_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(hull, [cv2.convexHull(max(contours, key=cv2.contourArea))], 1)
    region = cv2.dilate(hull, np.ones((7, 7), np.uint8)).astype(bool)
    # Anything whose colour differs from the mat's own (robust median) colour:
    # bright rubber, a pale bowl rim and its shaded outer wall alike.
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (3, 3), 0), cv2.COLOR_RGB2LAB).astype(float)
    # Reference colour from mat pixels that are dark in this very frame (arm, cover and objects excluded).
    surface = mat_mask & hull.astype(bool) & dark_surface_mask(rgb)[0] & ~exclusion
    reference = np.median(lab[surface], axis=0); spread = np.linalg.norm(lab[surface] - reference, axis=1)
    distance = np.linalg.norm(lab - reference, axis=2)
    bright = (distance > max(18., np.percentile(spread, 97) * 1.5)) & region & ~exclusion
    bright = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bright = cv2.drawContours(np.zeros_like(bright), contours, -1, 1, -1)  # fill holes
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    # A blob merged into the table outside the mat is not an object on the mat.
    outside = ~hull.astype(bool)
    blobs = []
    for i in range(1, count):
        blob = labels == i
        if stats[i, cv2.CC_STAT_AREA] < 12: continue
        if (blob & outside).sum() > .5 * blob.sum(): continue
        blobs.append(blob)
    return blobs, hull.astype(bool)


def fit_object(spec, blob, support, camera, valid):
    """Pose (x, y, yaw) of a library model whose projected hull matches the blob."""
    shape = blob.shape
    ys, xs = np.nonzero(blob)
    top = support + (spec['size_m'][2] if spec['shape'] == 'box' else spec['height_m'])
    centre = backproject([[xs.mean(), ys.mean()]], camera, shape[1], shape[0], support + .6 * (top - support))[0, :2]

    def cost(pose):
        sil = silhouette(spec, pose, support, camera, shape)
        return 1 - ((sil & blob) & valid).sum() / max(1, ((sil | blob) & valid).sum())
    symmetric = spec['shape'] != 'box'
    best = None
    for yaw in ([0.] if symmetric else np.linspace(0, np.pi, 6, endpoint=False)):
        start = np.array([*centre, yaw])
        lower, upper = start - [.04, .04, .6], start + [.04, .04, .6]
        x, value = cma_es(cost, start, np.array([.006, .006, .25 if not symmetric else 1e-6]), lower, upper, iterations=45, population=12, seed=int(yaw * 10))
        if best is None or value < best[1]: best = (x, value)
    pose, value = best
    if not symmetric: pose[2] = (pose[2] + np.pi / 2) % np.pi - np.pi / 2  # box yaw is defined modulo 180 deg
    else: pose[2] = 0.
    return pose, 1 - value


def fit_objects(rgb, selected, specs, camera, mat_world, mat_mask, exclusion, table_height=.001, blobs=None):
    """Assign blobs to the selected library models and fit each pose. Larger models are matched first."""
    shape = rgb.shape[:2]
    if blobs is None: blobs, _ = object_candidates(rgb, mat_mask, exclusion)
    support = mat_world['position'][2] + mat_world['size'][2] / 2
    valid = ~exclusion
    order = sorted(selected, key=lambda key: -(np.ptp(object_points(specs[key])[:, 0]) * np.ptp(object_points(specs[key])[:, 1])))
    result = {}; used = set()
    for key in order:
        spec = specs[key]; best = None
        for i, blob in enumerate(blobs):
            if i in used: continue
            pose, iou = fit_object(spec, blob, support, camera, valid)
            # Expected size check: the fitted silhouette area must resemble the blob area.
            area = silhouette(spec, pose, support, camera, shape).sum()
            ratio = blob.sum() / max(1, area)
            if not .35 < ratio < 2.5: continue
            if best is None or iou > best[1]: best = (pose, iou, i, ratio)
        if best is None:
            raise ValueError(f'{spec["name"]} was not found on the mat in the first frame')
        pose, iou, index, ratio = best; used.add(index)
        result[key] = {'pose_xy_yaw': [float(v) for v in pose], 'support_z_m': support, 'mask_iou': float(iou), 'blob_area_ratio': float(ratio),
                       'blob_pixels': int(blobs[index].sum())}
    return result, blobs


# ------------------------------------------------------- SAM mask refinement
class SamRefiner:
    """Replaces colour blobs by SAM 2.1 masks prompted automatically at each blob's interior point.

    Colour blobs miss shaded parts (a bowl's outer wall). SAM recovers the
    whole object; it is only accepted when it stays consistent with the blob.
    Missing SAM weights or worker fall back to the colour blobs.
    """
    def __init__(self):
        self.client = None; self.available = True; self.folder = ROOT / 'output/studio/import-sam'; self.folder.mkdir(parents=True, exist_ok=True)

    def refine(self, rgb, blobs, tag):
        if not self.available or not blobs: return blobs, ['color'] * len(blobs)
        try:
            from studio_sam_client import SAMClient, decode_mask
            if self.client is None: self.client = SAMClient()
            h, w = rgb.shape[:2]; path = self.folder / f'{tag}.png'
            cv2.imwrite(str(path), cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 480), interpolation=cv2.INTER_CUBIC))
            refined, sources = [], []
            for blob in blobs:
                distance = cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 3)
                y, x = np.unravel_index(np.argmax(distance), distance.shape)  # deepest interior point
                try: mask = decode_mask(self.client.predict(path, [float(x) * 640 / w, float(y) * 480 / h]))
                except ValueError as error:
                    if 'worker exited' in str(error) or 'timed out' in str(error): raise
                    refined.append(blob); sources.append('color'); continue
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                covered = (mask & blob).sum() / blob.sum()
                if covered > .7 and blob.sum() * .8 <= mask.sum() <= blob.sum() * 3:
                    refined.append(mask | blob); sources.append('sam2.1')
                else:
                    refined.append(blob); sources.append('color')
            path.unlink(missing_ok=True)
            return refined, sources
        except Exception as error:  # noqa: BLE001 - optional accelerator; colour masks remain valid
            print(f'SAM unavailable, using colour masks: {error}', flush=True); self.available = False
            return blobs, ['color'] * len(blobs)

    def close(self):
        if self.client: self.client.close()
