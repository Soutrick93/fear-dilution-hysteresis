# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Homoclinic orbit to the interior saddle, located by the manifold-splitting method (no MATCONT needed).

Idea. A homoclinic loop exists when one branch of the saddle's unstable manifold W^u comes back exactly along a
branch of its stable manifold W^s. Take the section Sigma = half-line from the coexistence focus E pointing away
from the saddle S. Follow W^u forward and W^s backward to their first crossing of Sigma and record the distances
from E, r_u and r_s. The splitting function

        Delta(f) = r_u(f) - r_s(f)

changes sign exactly at the homoclinic value f_hom. Brent's method on Delta is well conditioned, unlike bisection
on the long-run fate of single orbits.

Cross-check. As f -> f_hom from the cycle side, the cycle period grows like T ~ -(1/lambda_u) ln|f - f_hom|,
where lambda_u is the saddle's unstable eigenvalue.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from .model import Params, rhs, rhs_log, jacobian
from .equilibria import interior


def saddle_and_focus(p: Params):
    eq = interior(p)
    if len(eq) != 2:
        return None
    S, E = np.array(eq[0]), np.array(eq[1])     # sorted by y: lower = saddle, upper = coexistence state
    J = jacobian(*S, p)
    ev, V = np.linalg.eig(J)
    if not (ev.real.min() < 0 < ev.real.max()):
        return None
    iu, is_ = int(np.argmax(ev.real)), int(np.argmin(ev.real))
    return S, E, ev[iu].real, V[:, iu].real, ev[is_].real, V[:, is_].real


def splitting(p: Params, eps=1e-7, a0_frac=0.05, t_max=None):
    """Signed return distance of the saddle's unstable manifold W^u across its stable manifold W^s.

    Coordinates near the saddle S: z - S = a v_s + b v_u. Follow the branch of W^u that leaves towards the focus E
    once around the loop, and stop where it comes back to the line a = a0 (a0 small, on the side from which the inner
    branch of W^s enters S). Return b there, signed so that b > 0 means W^u came back INSIDE (towards E, onto the cycle)
    and b < 0 OUTSIDE (it then escapes to the raccoon-free state). b = 0 is the homoclinic orbit. If W^u never comes
    back near S, return +1 if it was captured inside (a cycle or the focus) and -1 if it ended raccoon-free.
    Log variables are used so orbits passing within 1e-15 of x = 0 stay accurate."""
    out = saddle_and_focus(p)
    if out is None:
        return np.nan
    S, E, lu, vu, ls, vs = out
    toE = E - S
    if np.dot(vu, toE) < 0: vu = -vu            # unstable branch that leaves towards the focus side
    if np.dot(vs, toE) < 0: vs = -vs            # inner stable branch enters S from the focus side
    M = np.column_stack([vs, vu]); Minv = np.linalg.inv(M)
    a0 = a0_frac * np.linalg.norm(toE)
    t_max = t_max or 60.0 / max(abs(lu), 1e-4) + 6000.0 / p.delta

    def ev(t, w):
        a, b = Minv @ (np.exp(w) - S)
        return a - a0
    ev.direction = -1.0                          # a decreasing: coming back towards S along v_s
    ev.terminal = True
    sol = solve_ivp(lambda t, w: rhs_log(t, w, p), (0, t_max), np.log(S + eps * vu), method='LSODA',
                    rtol=1e-11, atol=1e-12, events=ev)
    if sol.t_events[0].size == 0:
        # never came back near S: inside (captured by the cycle or focus) unless it ended raccoon-free (outside)
        # relative test: near f_T the saddle itself has y ~ 1e-3, so an absolute threshold misclassifies
        # escaped = heading to the raccoon-free state E1 = (1, 0): crabs near 1 and raccoons below the saddle level.
        # (Near f_T, E1 attracts only weakly, so y decays very slowly; a pure y-threshold misreads this as captured.)
        xe, ye = np.exp(sol.y[:, -1])
        return -1.0 if (xe > 0.9 and ye < 0.5 * S[1]) or ye < 1e-3 * S[1] else 1.0
    a, b = Minv @ (np.exp(sol.y_events[0][0]) - S)
    return float(b) / a0                         # b > 0: inside (same side as the departing branch, towards E)


def find_homoclinic(p: Params, f_lo, f_hi, n_scan=12, tol=1e-9):
    """Scan the splitting sign on [f_lo, f_hi] and bisect each sign change down to tol.
    Returns (roots, scanned f values, scanned splitting values)."""
    fs = np.linspace(f_lo, f_hi, n_scan)
    D = np.array([splitting(p.with_(f=float(f))) for f in fs])
    roots = []
    for i in range(n_scan - 1):
        if np.isfinite(D[i]) and np.isfinite(D[i + 1]) and np.sign(D[i]) != np.sign(D[i + 1]):
            lo, hi, slo = fs[i], fs[i + 1], np.sign(D[i])
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                sm = np.sign(splitting(p.with_(f=float(mid))))
                if sm == slo: lo = mid
                else: hi = mid
            roots.append(0.5 * (lo + hi))
    return roots, fs, D


def cycle_period(p: Params, z0=None, n_cross=4, t_max=None, chunk=None):
    """Period of the attracting cycle, from successive upward crossings of the line x = x_E (focus).
    Starts on the saddle's unstable manifold (on top of the cycle near f_hom, so no long transient) and integrates in
    chunks, stopping as soon as n_cross + 2 crossings are found (fast) or the orbit collapses to raccoon-free."""
    eq = interior(p)
    xE = eq[-1][0]
    if z0 is None:
        S, E, lu, vu, ls, vs = saddle_and_focus(p)
        if np.dot(vu, E - S) < 0: vu = -vu
        z0 = S + 1e-7 * vu
    t_max = t_max or 40000.0 / p.delta
    chunk = chunk or 1000.0 / p.delta
    evl = lambda t, w: np.exp(w[0]) - xE
    evl.direction = 1.0
    w, t, te = np.log(z0), 0.0, []
    while t < t_max:
        sol = solve_ivp(lambda tt, ww: rhs_log(tt, ww, p), (t, t + chunk), w, method='LSODA', rtol=1e-10,
                        atol=1e-12, events=evl)
        te.extend(sol.t_events[0].tolist()); w = sol.y[:, -1]; t = sol.t[-1]
        if np.exp(w[1]) < 1e-4:                  # fell to the raccoon-free state: no cycle
            return np.nan
        if len(te) >= n_cross + 2:
            return float(np.mean(np.diff(te[-n_cross:])))
    return np.nan
