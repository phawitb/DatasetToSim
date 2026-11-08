"""Seeded full-mat proposals; planning and contact audits decide acceptance."""
import numpy as np


EDGE_CLEARANCE_M = .002
OBJECT_CLEARANCE_M = .008
# Initial centers must leave room for the open, asymmetric SO-101 jaws during
# approach (35 native opening) and release (40). This is a proposal margin,
# not a collision guarantee: every trajectory still requires the contact audit.
# The calibrated pad-center span at nominal grasp orientation is 58--65 mm;
# cube-only footprint rejection would allow centers only 36 mm apart.
STACKING_CENTER_SEPARATION_M = .08


def workspace_grid_cells(world, grid_shape=(3, 3)):
    """Row-major XY cells partitioning the entire calibrated mat, x varying first."""
    if (not isinstance(grid_shape, (list, tuple)) or len(grid_shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0
                   for n in grid_shape)):
        raise ValueError('grid_shape must contain two positive integer cell counts')
    center = np.asarray(world['mat']['position'][:2], dtype=float)
    half = np.asarray(world['mat']['size'][:2], dtype=float) / 2
    lower = center - half
    width = 2 * half / np.asarray(grid_shape)
    return [{'cell_id': y * grid_shape[0] + x,
             'bounds_xy_m': [(lower + [x, y] * width).tolist(),
                             (lower + [x + 1, y + 1] * width).tolist()]}
            for y in range(grid_shape[1]) for x in range(grid_shape[0])]


def sample_workspace_layout(world, rng, base_bounds, task, layout_cell=None):
    """Draw footprints over the whole calibrated mat, without reachability bias.

    Circumscribed circles conservatively separate objects, while yaw-aware box
    extents keep each cube on the mat. Base bounds come from the robot geometry.
    The unstack placement is reserved empty space, not a third physical cube.
    """
    mat = world['mat']
    center = np.asarray(mat['position'][:2], dtype=float)
    half = np.asarray(mat['size'][:2], dtype=float) / 2
    low, high = center - half, center + half
    edge = float(world['cube']['size'])
    occupied = []
    draws = {}
    base_low, base_high = np.asarray(base_bounds, dtype=float)
    cells = {}
    if layout_cell is not None:
        if task != 'stack' or not isinstance(layout_cell, dict):
            raise ValueError('layout_cell must be a cell descriptor for stacking')
        grid_shape = layout_cell.get('grid_shape', [3, 3])
        grid = workspace_grid_cells(world, grid_shape)
        for name, key in [('source_cube', 'source_cell'), ('other_cube', 'target_cell')]:
            cell_id = layout_cell.get(key)
            if isinstance(cell_id, bool) or not isinstance(cell_id, (int, np.integer)) or not 0 <= cell_id < len(grid):
                raise ValueError(f'{key} must be a valid cell index in grid {list(grid_shape)}')
            cells[name] = grid[cell_id]

    class SamplingExhausted(ValueError):
        pass

    def draw(name, radius, yaw=0., extent=None, minimum_separation=0.):
        extent = np.full(2, radius) if extent is None else np.asarray(extent)
        lower, upper = low + extent + EDGE_CLEARANCE_M, high - extent - EDGE_CLEARANCE_M
        if name in cells:
            cell_low, cell_high = np.asarray(cells[name]['bounds_xy_m'])
            lower, upper = np.maximum(lower, cell_low), np.minimum(upper, cell_high)
        if np.any(lower >= upper):
            raise ValueError(f'{name} does not fit on the calibrated mat in requested cell {cells.get(name)}')
        for attempt in range(128 if cells else 1000):
            xy = rng.uniform(lower, upper)
            nearest_base = np.clip(xy, base_low, base_high)
            if np.linalg.norm(xy - nearest_base) < radius + OBJECT_CLEARANCE_M:
                continue
            if any(np.linalg.norm(xy - p) < max(radius + r + OBJECT_CLEARANCE_M, minimum_separation)
                   for p, r in occupied):
                continue
            occupied.append((xy, radius))
            draws[name] = {'position_xy_m': xy.tolist(), 'yaw_rad': float(yaw),
                           'center_bounds_xy_m': [lower.tolist(), upper.tolist()],
                           'proposal_count': attempt + 1, 'footprint_radius_m': float(radius)}
            if name in cells:
                draws[name].update(cell_id=int(cells[name]['cell_id']), cell_bounds_xy_m=cells[name]['bounds_xy_m'])
            return xy, yaw
        raise SamplingExhausted(f'Could not sample a separated {name} on the calibrated mat in requested cell {cells.get(name)}')

    def cube(name, separation=0.):
        yaw = float(rng.uniform(-.10, .10))
        extent = edge / 2 * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
        return draw(name, edge / np.sqrt(2), yaw, [extent, extent], separation)

    for pair_attempt in range(64 if cells else 1):
        occupied.clear()
        draws.clear()
        try:
            source = cube('source_cube')
            destination = cube('other_cube', STACKING_CENTER_SEPARATION_M if cells else .08 if task == 'unstack' else 0.)
            break
        except SamplingExhausted:
            if not cells or pair_attempt == 63:
                raise
    bowl = draw('bowl', float(world['bowl']['outer_radius']))[0] if task == 'bowl' else np.asarray(world['bowl']['position'][:2])
    policy = {'version': 2, 'distribution': 'uniform_xy_over_full_calibrated_mat_with_footprint_rejection',
              'mat_bounds_xy_m': [low.tolist(), high.tolist()],
              'robot_base_bounds_xy_m': [base_low.tolist(), base_high.tolist()],
              'edge_clearance_m': EDGE_CLEARANCE_M, 'object_clearance_m': OBJECT_CLEARANCE_M,
              'reachability_filter': 'none_at_sampling; planner_and_contact_audit_must_pass',
              'acceptance_note': 'Accepted episodes are the reachable, physically valid subset of full-mat proposals.'}
    if cells:
        policy.update(version=3, distribution='uniform_xy_within_requested_full_mat_grid_cells_with_footprint_rejection',
                      grid_shape=list(map(int, grid_shape)), grid_cells=grid,
                      source_cell=int(cells['source_cube']['cell_id']), target_cell=int(cells['other_cube']['cell_id']),
                      minimum_center_separation_m=STACKING_CENTER_SEPARATION_M,
                      separation_note='Conservative initial open-jaw clearance; full planned-motion contact audit remains mandatory.',
                      pair_proposal_count=pair_attempt + 1,
                      acceptance_note='Requested cells balance accepted source and target coverage; IK and contact failures remain rejected proposals.')
    return source, destination, bowl, draws, policy
