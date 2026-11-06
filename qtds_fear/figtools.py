# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Shared computations for the publication figures.

DRAFT mode (environment variable QTDS_DRAFT=1) lowers every resolution so a figure renders in seconds for checking.
Final mode (default) uses full resolution.
"""
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp
from .model import Params, rhs_log
from .equilibria import interior, classify, raccoon_free_stable
from . import style as S

DRAFT = os.environ.get('QTDS_DRAFT', '0') == '1'


def res(final, draft):
    return draft if DRAFT else final


# ---------------------------------------------------------------- colour maps (built from the palette)
def cmap(name):
    maps = {
        'fear': ['#f3f0fd', '#c9c0f4', S.FEAR, '#2e2380'],
        'prey': ['#effaf5', '#a6e3c9', S.PREY, '#0b5c3e'],
        'meso': ['#fff3ec', '#f7b899', S.MESO, '#8a2f0c'],
        'time': ['#d9e7f9', S.BLUE, S.FEAR, '#2b1a5c'],
        'speed': ['#ffffff', '#c9c9c4', '#7b7a74', '#2c2c2a'],
        'regime': [S.TINT_FREE, S.TINT_COEX, S.TINT_BISTABLE, S.TINT_CYCLE],
    }
    return LinearSegmentedColormap.from_list(name, maps[name])


# ---------------------------------------------------------------- vectorised integrator for basins
def rhs_log_vec(W, p: Params):
    x, y = np.exp(W[0]), np.exp(W[1])
    s = p.s
    g = x / (s + x)
    phi = (1 + p.c * y) / (1 + p.c * y + p.f)
    return np.stack([(1 - x) - y / (s + x), p.delta * (g * phi - p.mu)])


def basin_grid(p: Params, xs, ys, t_end, dt):
    """Final state of a whole grid of initial conditions (RK4 in log variables, vectorised).
    Returns (raccoons_persist[bool grid], final_x, final_y)."""
    X, Y = np.meshgrid(xs, ys)
    W = np.log(np.stack([X, Y]))
    n = int(t_end / dt)
    for _ in range(n):
        k1 = rhs_log_vec(W, p); k2 = rhs_log_vec(W + 0.5 * dt * k1, p)
        k3 = rhs_log_vec(W + 0.5 * dt * k2, p); k4 = rhs_log_vec(W + dt * k3, p)
        W = W + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        W = np.clip(W, -60, 5)
    xf, yf = np.exp(W)
    eq = interior(p)
    ysad = eq[0][1] if len(eq) == 2 else 1e-3
    return yf > 0.3 * ysad, xf, yf


def field(p: Params, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    s = p.s; g = X / (s + X); phi = (1 + p.c * Y) / (1 + p.c * Y + p.f)
    U = X * (1 - X) - Y * g
    V = p.delta * Y * (g * phi - p.mu)
    return X, Y, U, V


def predator_nullcline_y(p: Params, xs, ys):
    """Return the grid of y' / (delta y) whose zero contour is the predator nullcline."""
    X, Y = np.meshgrid(xs, ys)
    s = p.s; g = X / (s + X); phi = (1 + p.c * Y) / (1 + p.c * Y + p.f)
    return X, Y, g * phi - p.mu


def orbit(p: Params, z0, t_end, n=4000, f_of_t=None):
    def F(t, w):
        q = p if f_of_t is None else p.with_(f=float(f_of_t(t)))
        return rhs_log(t, w, q)
    ts = np.linspace(0, t_end, n)
    sol = solve_ivp(F, (0, t_end), np.log(z0), t_eval=ts, method='LSODA', rtol=1e-9, atol=1e-11)
    return sol.t, np.exp(sol.y)


def saddle_manifolds(p: Params, t_end=3000.0, eps=1e-6):
    """Stable (backward in time) and unstable (forward) manifold branches of the interior saddle."""
    eq = interior(p)
    if len(eq) != 2:
        return [], []
    from .model import jacobian
    Sd = np.array(eq[0]); ev, V = np.linalg.eig(jacobian(*Sd, p))
    vs, vu = V[:, np.argmin(ev.real)].real, V[:, np.argmax(ev.real)].real

    def stop(t, w):
        return 6.0 - np.exp(w[1])
    stop.terminal = True
    branches = {'stable': [], 'unstable': []}
    for name, vec, sign in (('stable', vs, -1.0), ('unstable', vu, 1.0)):
        for sgn in (1.0, -1.0):
            z0 = Sd + sgn * eps * vec
            if np.any(z0 <= 0):
                continue
            sol = solve_ivp(lambda t, w: sign * rhs_log(t, w, p), (0, t_end), np.log(z0), method='LSODA',
                            rtol=1e-9, atol=1e-11, events=stop, max_step=t_end / 3000)
            branches[name].append(np.exp(sol.y))
    return branches['stable'], branches['unstable']


def regime(p: Params):
    """0 raccoon-free, 1 stable coexistence, 2 bistable (E1 + stable state), 3 oscillation (no stable interior state,
    E1 unstable), 4 bistable with oscillation (E1 stable, upper state unstable focus)."""
    eq = interior(p); e1 = raccoon_free_stable(p)
    st = [classify(e, p)['stable'] for e in eq]
    if not eq:
        return 0
    if not e1:
        return 1 if st[-1] else 3
    return 2 if st[-1] else 4
