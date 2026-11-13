"""Time-optimal speed along a fixed joint path, with velocity, acceleration and jerk limits.

Path-velocity decomposition (Bobrow et al.; the reachability idea of Pham's TOPP-RA,
discretised): the geometric path (Natural's quintic B-spline through the task
waypoints) is kept exactly; only the speed along it changes. Forward/backward passes
on x = sdot^2 give the fastest speed profile under per-joint velocity and acceleration
limits. A Gaussian smoothing of s(t) then rounds the acceleration switches so the
sampled jerk stays within its limit too (the acceleration budget is lowered until all
three limits hold on the actual 30 Hz commands). Endpoints are exact and at rest.

One uniform time scale for a whole curve (Natural's original timing) is set by the
curve's single most demanding point; here only that stretch is slow.
"""
import numpy as np


def _profile(d1, d2, ds, vmax, amax):
    """Largest x = sdot^2 per grid point, starting and ending at rest."""
    with np.errstate(divide='ignore', invalid='ignore'):
        xv = np.min(np.where(np.abs(d1) > 1e-12, (vmax / np.abs(d1)) ** 2, np.inf), axis=1)
    moving = np.abs(d1) > 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(moving, 1 / d1, 0.)

    def bounds(k, x):
        # |d1 sdd + d2 x| <= a for every joint -> one interval for sdd.
        b = d2[k] * x; m = moving[k]
        u = (-amax - b) * inv[k]; v = (amax - b) * inv[k]
        return np.max(np.minimum(u, v)[m], initial=-np.inf), np.min(np.maximum(u, v)[m], initial=np.inf)

    n = len(xv); fwd = np.zeros(n); back = np.zeros(n)
    for k in range(n - 1):
        lo, hi = bounds(k, fwd[k])
        fwd[k + 1] = min(xv[k + 1], max(0., fwd[k] + 2 * ds * hi)) if np.isfinite(hi) else xv[k + 1]
    for k in range(n - 1, 0, -1):
        lo, hi = bounds(k, back[k])
        back[k - 1] = min(xv[k - 1], max(0., back[k] - 2 * ds * lo)) if np.isfinite(lo) else xv[k - 1]
    x = np.minimum(fwd, back); x[0] = x[-1] = 0.
    return x


MAX_DURATION_S = 30.  # a stalled profile (speed ~0 mid-path) must not allocate hours of samples


def _times(x, ds):
    v = np.sqrt(np.maximum(x, 0.))
    return np.r_[0., np.cumsum(2 * ds / np.maximum(v[1:] + v[:-1], 1e-9))]


def _derivatives(q, rate):
    vel = np.max(np.abs(np.diff(q, axis=0)), axis=0) * rate
    acc = np.max(np.abs(np.diff(q, n=2, axis=0)), axis=0) * rate ** 2 if len(q) > 2 else np.zeros(q.shape[1])
    jerk = np.max(np.abs(np.diff(q, n=3, axis=0)), axis=0) * rate ** 3 if len(q) > 3 else np.zeros(q.shape[1])
    return vel, acc, jerk


def time_optimal(curve, limits, rate=30., grid=400, minimum=0., joints=None):
    """Positions at `rate` Hz along `curve(s)`, s in [0, 1], within `limits` (3 x dof:
    velocity, acceleration, jerk), lasting at least `minimum` seconds. `joints` selects
    the columns the limits apply to (others follow the path). Returns
    (positions, times, info), or None when the path does not move."""
    limits = np.asarray(limits, float)
    s = np.linspace(0., 1., grid); ds = s[1] - s[0]
    d1 = curve.derivative(1)(s); d2 = curve.derivative(2)(s)
    cols = np.arange(d1.shape[1]) if joints is None else np.asarray(joints)
    d1, d2, lim = d1[:, cols], d2[:, cols], limits[:, cols]
    if np.max(np.abs(d1)) < 1e-6:
        return None
    fine = 240.
    last = None
    for fraction in (.85, .7, .55, .45, .36, .28, .22, .17, .13, .1):
        x = _profile(d1, d2, ds, lim[0] * .95, lim[1] * fraction)
        if np.any(x[5:-5] <= 1e-12):
            continue  # stalls mid-path (the ends start and stop at rest)
        x[1:-1] = np.maximum(x[1:-1], 1e-10)
        t = _times(x, ds)
        if not np.isfinite(t[-1]) or t[-1] > MAX_DURATION_S or minimum > MAX_DURATION_S:
            continue
        for sigma in (.06, .1, .15):
            # s(t) on a fine clock, smoothed with a Gaussian; ends held so they stay at rest.
            total = max(t[-1], minimum)
            pad = 4 * sigma
            tt = np.arange(-pad, total + pad + 1 / fine, 1 / fine)
            st = np.interp(tt, t * (total / max(t[-1], 1e-9)), s, left=0., right=1.)
            k = np.arange(-int(3 * sigma * fine), int(3 * sigma * fine) + 1) / fine
            kernel = np.exp(-.5 * (k / sigma) ** 2); kernel /= kernel.sum()
            smooth = np.convolve(np.r_[np.zeros(len(k)), st, np.ones(len(k))], kernel, mode='same')[len(k):-len(k)]
            # Trim the rest spans the smoothing added at both ends (keep exact 0 and 1).
            moving = np.flatnonzero((smooth > 1e-7) & (smooth < 1 - 1e-7))
            smooth = smooth[max(0, moving[0] - 1):moving[-1] + 2]
            duration = (len(smooth) - 1) / fine
            ticks = max(1, int(np.ceil(duration * rate - 1e-9)))
            times = np.arange(ticks + 1) / rate
            s_out = np.clip(np.interp(times, np.arange(len(smooth)) / fine, smooth), 0., 1.); s_out[0], s_out[-1] = 0., 1.
            q = curve(s_out)
            vel, acc, jerk = _derivatives(q[:, cols], rate)
            ok = np.all(vel <= lim[0] + 1e-6) and np.all(acc <= lim[1] + 1e-6) and np.all(jerk <= lim[2] + 1e-6)
            last = (q, times, {'acceleration_fraction': fraction, 'sigma_s': sigma, 'duration': float(times[-1]), 'within_limits': bool(ok),
                               'max_velocity': vel.tolist(), 'max_acceleration': acc.tolist(), 'max_jerk': jerk.tolist()})
            if ok:
                return last
    return last
