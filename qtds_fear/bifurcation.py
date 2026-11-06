# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Bifurcations of the fear-diluted mesopredator model.

Closed forms (proved in symbolic.py):
  f_T : transcritical of E1 = (1,0):  mu (1 + f)(1 + s0(1 + q f)) = 1
  c*  : the transcritical is backward (hysteresis) iff c > c* = s_T (1 + f_T) / (f_T (1 + s_T)^2)
Numerical (exact roots of the cubic, refined by bisection / Brent):
  f_F : saddle-node (fold) closing the bistable window
  f_H : Hopf (trace zero, det > 0) and its first Lyapunov coefficient l1
"""
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from .model import Params, jacobian
from .equilibria import interior, classify


# ---------------------------------------------------------------- transcritical and c*
def f_transcritical(p: Params):
    A = p.mu * p.s0 * p.q
    B = p.mu * (1.0 + p.s0 + p.s0 * p.q)
    C = p.mu * (1.0 + p.s0) - 1.0
    if C >= 0:          # raccoons cannot persist even without fear
        return None
    return (-B + np.sqrt(B * B - 4.0 * A * C)) / (2.0 * A)


def c_star(p: Params):
    fT = f_transcritical(p)
    if fT is None:
        return None
    sT = p.s0 * (1.0 + p.q * fT)
    return sT * (1.0 + fT) / (fT * (1.0 + sT) ** 2)


def transcritical_slope(p: Params):
    """dy/df of the coexistence branch at (1, 0, f_T): negative = forward (reversible), positive = backward."""
    fT = f_transcritical(p)
    q = p.with_(f=fT)
    s = q.s
    # exact expression from symbolic.transcritical_slope():
    num = -q.mu * (fT * q.q * q.s0 + q.s0 + 1.0) * (2.0 * fT * q.q * q.s0 + q.q * q.s0 + q.s0 + 1.0)
    den = q.c * (1.0 + s) * (q.mu * (1.0 + s) - 1.0) + 1.0 - q.mu * (1.0 + fT)
    return num / den


# ---------------------------------------------------------------- scanning along f
def branch_scan(p: Params, f_grid):
    """Interior equilibria (with stability) at each f.  Returns a list of lists of dicts."""
    out = []
    for f in f_grid:
        q = p.with_(f=float(f))
        out.append([classify(e, q) for e in interior(q)])
    return out


def _count(p, f):
    return len(interior(p.with_(f=f)))


def fold_points(p: Params, fmax=6.0, n=1200, fmin=1e-6):
    """f values where the number of interior equilibria changes by two (saddle-node), excluding f_T.
    If none is found on the coarse grid but the transcritical is backward, a fine search right of f_T is made
    (near c* the window is very narrow)."""
    found = _fold_scan(p, fmin, fmax, n)
    cs = c_star(p)
    if not found and cs is not None and p.c > cs:
        ff = fold_by_discriminant(p, fmax)
        if ff is not None:
            found = [ff]
    return found


def _discriminant(p):
    from .equilibria import cubic_coeffs
    d, c_, b, a = cubic_coeffs(p)          # lowest degree first
    return 18 * a * b * c_ * d - 4 * b ** 3 * d + b * b * c_ * c_ - 4 * a * c_ ** 3 - 27 * a * a * d * d


def fold_by_discriminant(p: Params, fmax=6.0):
    """Saddle-node right of f_T for c > c*: the first f > f_T where the cubic's discriminant changes sign.
    Robust even when the bistable window is far narrower than any grid (near c*)."""
    fT = f_transcritical(p)
    D = lambda f: _discriminant(p.with_(f=f))
    lo = fT * (1 + 1e-13) + 1e-15
    if D(lo) <= 0:
        lo = fT + 1e-10
    step = 1e-9
    while lo + step < fmax:
        hi = lo + step
        if np.sign(D(hi)) != np.sign(D(lo)):
            return brentq(D, lo, hi, xtol=1e-15)
        lo, step = hi, step * 1.6
    return None


def _fold_scan(p, fmin, fmax, n):
    fs = np.linspace(fmin, fmax, n)
    cnt = [_count(p, f) for f in fs]
    fT = f_transcritical(p)
    folds = []
    for i in range(n - 1):
        if cnt[i] != cnt[i + 1]:
            lo, hi = fs[i], fs[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _count(p, mid) == cnt[i]:
                    lo = mid
                else:
                    hi = mid
            fc = 0.5 * (lo + hi)
            if fT is None or abs(fc - fT) > 1e-6:
                folds.append(fc)
    return folds


def hopf_points(p: Params, fmax=6.0, n=1200):
    """Hopf points along the upper (largest-y) interior branch, with l1."""
    fs = np.linspace(1e-6, fmax, n)
    def tr_up(f):
        q = p.with_(f=f)
        eq = interior(q)
        if not eq:
            return np.nan, np.nan
        J = jacobian(*eq[-1], q)
        return np.trace(J), np.linalg.det(J)
    vals = [tr_up(f) for f in fs]
    out = []
    for i in range(n - 1):
        (t0, d0), (t1, d1) = vals[i], vals[i + 1]
        if np.isfinite(t0) and np.isfinite(t1) and d0 > 0 and d1 > 0 and np.sign(t0) != np.sign(t1):
            fh = brentq(lambda f: tr_up(f)[0], fs[i], fs[i + 1], xtol=1e-13)
            q = p.with_(f=fh)
            e = interior(q)[-1]
            l1, om = lyap1(e, q)
            out.append(dict(f=fh, x=e[0], y=e[1], l1=l1, omega=om))
    return out


# ---------------------------------------------------------------- first Lyapunov coefficient (Kuznetsov 2004, eq. 3.20)
_x, _y = sp.symbols('x y')
_s0, _q, _c, _f, _mu, _dl = sp.symbols('s0 q c f mu delta')
_s = _s0 * (1 + _q * _f)
_F = sp.Matrix([_x * (1 - _x) - _x * _y / (_s + _x),
                _dl * _y * (_x / (_s + _x) * (1 + _c * _y) / (1 + _c * _y + _f) - _mu)])
_X = sp.Matrix([_x, _y])
_v = sp.symbols('v1:3'); _w = sp.symbols('w1:3'); _u = sp.symbols('u1:3')
_B = [sum(sp.diff(_F[i], _X[j], _X[k]) * _v[j] * _w[k] for j in range(2) for k in range(2)) for i in range(2)]
_C = [sum(sp.diff(_F[i], _X[j], _X[k], _X[m]) * _v[j] * _w[k] * _u[m]
          for j in range(2) for k in range(2) for m in range(2)) for i in range(2)]
_args = (_x, _y, _s0, _q, _c, _f, _mu, _dl)
_Bn = sp.lambdify(_args + tuple(_v) + tuple(_w), _B, 'numpy')
_Cn = sp.lambdify(_args + tuple(_v) + tuple(_w) + tuple(_u), _C, 'numpy')


def lyap1(e, p: Params):
    """First Lyapunov coefficient at a Hopf point e (eigenvalues +- i omega). l1 < 0: supercritical."""
    A = jacobian(e[0], e[1], p)
    ev, V = np.linalg.eig(A)
    i = int(np.argmax(ev.imag))
    om = ev[i].imag
    qv = V[:, i]
    evT, W = np.linalg.eig(A.T)
    j = int(np.argmin(np.abs(evT + 1j * om)))
    pv = W[:, j]
    pv = pv / np.conj(np.vdot(pv, qv))
    a = (e[0], e[1], p.s0, p.q, p.c, p.f, p.mu, p.delta)
    B = lambda v1, v2: np.array(_Bn(*a, *v1, *v2), dtype=complex)
    C = lambda v1, v2, v3: np.array(_Cn(*a, *v1, *v2, *v3), dtype=complex)
    h11 = np.linalg.solve(A, B(qv, np.conj(qv)))
    h20 = np.linalg.solve(2j * om * np.eye(2) - A, B(qv, qv))
    val = np.vdot(pv, C(qv, qv, np.conj(qv)) - 2.0 * B(qv, h11) + B(np.conj(qv), h20))
    return float(val.real / (2.0 * om)), float(om)


def summary(p: Params, fmax=6.0):
    """All codimension-1 points along f for parameter set p."""
    fT = f_transcritical(p)
    return dict(f_T=fT, c_star=c_star(p), backward=(p.c > (c_star(p) or np.inf)),
                folds=fold_points(p, fmax), hopf=hopf_points(p, fmax))


def opening_coefficient(p: Params):
    """Coefficient C in f_F - f_T ~ C (c - c*)^2 (Proposition 'Opening of the window')."""
    fT = f_transcritical(p); cs = c_star(p)
    if fT is None or cs is None:
        return None
    mu = p.mu; sT = p.with_(f=fT).s
    k = sT * fT / (1 + fT)
    a2 = sT / (1 + sT) * (2 * (1 - mu) - sT / (1 + sT))
    Lp = mu * (1 + sT) + mu * (1 + fT) * p.s0 * p.q
    return k ** 2 / (4 * a2 * cs * sT * Lp)
