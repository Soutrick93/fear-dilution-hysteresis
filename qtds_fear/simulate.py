# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Time integration helpers: trajectories, quasi-static ramps, basins of attraction, limit-cycle amplitude."""
import numpy as np
from scipy.integrate import solve_ivp
from .model import Params, rhs


def trajectory(p: Params, z0, t_end, n=2000, immigration=0.0, f_of_t=None, rtol=1e-9, atol=1e-12):
    """Integrate; f_of_t (optional) makes the fear cue time dependent; immigration adds a tiny raccoon inflow."""
    def F(t, z):
        q = p if f_of_t is None else p.with_(f=float(f_of_t(t)))
        d = rhs(t, z, q)
        d[1] += immigration
        return d
    ts = np.linspace(0.0, t_end, n)
    sol = solve_ivp(F, (0.0, t_end), z0, t_eval=ts, method='LSODA', rtol=rtol, atol=atol)
    return sol.t, sol.y


def quasi_static_ramp(p: Params, f_values, z0, settle=30000.0, immigration=1e-6):
    """Slowly step f through f_values, letting the state settle at each step. Returns final (x, y) per step."""
    z = np.array(z0, float); out = []
    for f in f_values:
        q = p.with_(f=float(f))
        sol = solve_ivp(lambda t, w: rhs(t, w, q) + np.array([0.0, immigration]), (0, settle), z,
                        method='LSODA', rtol=1e-9, atol=1e-12)
        z = sol.y[:, -1]; out.append(z.copy())
    return np.array(out)


def basin_grid(p: Params, xs, ys, t_end=4000.0, y_threshold=1e-3):
    """1 if the orbit from (x, y) ends with raccoons present (coexistence or cycle), 0 if raccoon-free."""
    B = np.zeros((len(ys), len(xs)))
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            sol = solve_ivp(lambda t, w: rhs(t, w, p), (0, t_end), [x0, y0], method='LSODA', rtol=1e-7, atol=1e-10)
            B[i, j] = 1.0 if sol.y[1, -1] > y_threshold else 0.0
    return B


def cycle_amplitude(p: Params, z0, t_transient=20000.0, t_record=5000.0):
    """Peak-to-peak amplitude of y on the attractor reached from z0 (0 for an equilibrium)."""
    sol = solve_ivp(lambda t, w: rhs(t, w, p), (0, t_transient + t_record), z0, method='LSODA',
                    rtol=1e-9, atol=1e-12, dense_output=True)
    ts = np.linspace(t_transient, t_transient + t_record, 20000)
    yy = sol.sol(ts)[1]; xx = sol.sol(ts)[0]
    return float(yy.max() - yy.min()), float(xx.max() - xx.min()), sol
