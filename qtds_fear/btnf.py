# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Bogdanov-Takens points and their quadratic normal-form coefficients (Kuznetsov 2004, Sec. 8.4)."""
import numpy as np
from .model import Params, jacobian
from .equilibria import interior
from .bifurcation import fold_points, _Bn


def bt_point(s0, c):
    p = Params(s0=s0, c=c, delta=1.0)
    fl = fold_points(p, fmax=8.0, n=3000)
    if not fl:
        return None
    fF = fl[0]; eq = interior(p.with_(f=fF - 1e-11))
    e = tuple(np.mean(eq, axis=0)) if len(eq) == 2 else eq[0]
    J = jacobian(*e, p.with_(f=fF)); J11, j22 = J[0, 0], J[1, 1]
    if not (J11 < 0 < j22):
        return None
    return p.with_(f=fF, delta=-J11 / j22), e


def coefficients(p, e):
    A = jacobian(*e, p)
    B = lambda v, w: np.array(_Bn(e[0], e[1], p.s0, p.q, p.c, p.f, p.mu, p.delta, *v, *w), dtype=float)
    # Jordan chain: q0 spans ker A, q1 solves A q1 = q0 (least squares, then made orthogonal-free below)
    U, S, Vt = np.linalg.svd(A); q0 = Vt[-1]
    if q0[0] < 0: q0 = -q0          # fix orientation: (a, b) -> (-a, -b) under q0 -> -q0, so only a canonical
                                    # orientation makes a and b continuous along the BT curve
    q1 = np.linalg.lstsq(A, q0, rcond=None)[0]
    U2, S2, Vt2 = np.linalg.svd(A.T); p1 = Vt2[-1]
    p0 = np.linalg.lstsq(A.T, p1, rcond=None)[0]
    # normalise: <p0,q0>=1 and <p1,q1>=1 ; <p1,q0>=0 holds automatically; fix <p0,q1>=0 by p0 -> p0 - k p1
    sc = p1 @ q1; p1 = p1 / sc; p0 = p0 / sc
    p0 = p0 - (p0 @ q1) * p1
    sc0 = p0 @ q0; p0 = p0 / sc0; p1 = p1 / sc0
    checks = dict(p0q0=p0 @ q0, p1q1=p1 @ q1, p0q1=p0 @ q1, p1q0=p1 @ q0, residual=float(np.abs(A @ q1 - q0).max()),
                  eigenvalues=[complex(v).__repr__() for v in np.linalg.eigvals(A)])
    a = 0.5 * p1 @ B(q0, q0)
    b = p0 @ B(q0, q0) + p1 @ B(q0, q1)
    return float(a), float(b), checks


