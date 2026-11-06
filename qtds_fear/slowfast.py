# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Slow-fast structure (delta -> 0).

Fast subsystem (raccoons frozen):  x' = x(1-x) - x y/(s+x).  Its nonzero equilibria form the critical manifold
    C0 : y = (1-x)(s+x),  attracting for x > x_h = (1-s)/2, repelling for x < x_h (fold of C0 at x_h).
A 28-day playback changes s (foraging) on the fast time scale only: crabs move along a fast fibre (y fixed)
to the new C0. Raccoons then drift slowly along C0 towards the new equilibrium.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .model import Params, rhs


def critical_manifold(p: Params, n=400):
    x = np.linspace(0.0, 1.0, n)
    y = (1.0 - x) * (p.s + x)
    xh = (1.0 - p.s) / 2.0
    return x, y, xh


def fast_response(p_before: Params, p_after: Params, x0, y0, t_end):
    """Crab trajectory with raccoons held at y0 (fast subsystem), after the fear cue switches."""
    s = p_after.s
    sol = solve_ivp(lambda t, z: [z[0] * (1 - z[0]) - z[0] * y0 / (s + z[0])], (0, t_end), [x0],
                    dense_output=True, rtol=1e-10, atol=1e-12)
    return sol


def time_to_gain(p_before: Params, p_after: Params, x0, y0, gain, t_max=200.0):
    """Fast time (units 1/rho) for crabs to reach gain * x0 with raccoons frozen."""
    sol = fast_response(p_before, p_after, x0, y0, t_max)
    ts = np.linspace(0, t_max, 400001)
    xs = sol.sol(ts)[0]
    i = np.argmax(xs >= gain * x0)
    return ts[i] if xs[i] >= gain * x0 else np.nan


def cycle_amplitudes(p: Params, f_values, z_near, t_transient=30000.0, t_record=8000.0):
    """Peak-to-peak raccoon amplitude on the attractor for each f (continuation by reusing the last end state)."""
    amps = []
    z = np.array(z_near, float)
    for f in f_values:
        q = p.with_(f=float(f))
        sol = solve_ivp(lambda t, w: rhs(t, w, q), (0, t_transient + t_record), z, method='LSODA',
                        rtol=1e-10, atol=1e-13, dense_output=True)
        ts = np.linspace(t_transient, t_transient + t_record, 40000)
        Y = sol.sol(ts)
        amps.append((float(Y[1].max() - Y[1].min()), float(Y[0].max() - Y[0].min())))
        z = sol.y[:, -1]
    return np.array(amps)
