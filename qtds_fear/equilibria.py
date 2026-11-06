# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Equilibria and their stability.

Boundary:  E0 = (0, 0) (saddle),  E1 = (1, 0) (raccoon-free).
Interior: with u = 1 + c y, the predator nullcline gives x = s mu (u + f) / D,  D = (1 - mu) u - mu f,
and the prey nullcline y = (1 - x)(s + x) becomes the cubic (derivation in symbolic.py):

    P(u) = (u - 1) D(u)^2 - c s u [ D(u) - s mu (u + f) ] = 0 .

Admissible roots: u > 1 (y > 0) and D > 0 with 0 < x < 1. For c = 0 the interior equilibrium is unique and explicit.
"""
import numpy as np
from numpy.polynomial import polynomial as Pn
from .model import Params, jacobian


def boundary(p: Params):
    return [(0.0, 0.0), (1.0, 0.0)]


def raccoon_free_stable(p: Params):
    """E1 = (1,0) is stable iff mu (1 + f)(1 + s) > 1 (eigenvalues -1 and delta[1/((1+s)(1+f)) - mu])."""
    return p.mu * (1.0 + p.f) * (1.0 + p.s) > 1.0


def cubic_coeffs(p: Params):
    """Coefficients of P(u), lowest degree first."""
    s, mu, f, c = p.s, p.mu, p.f, p.c
    a, b = 1.0 - mu, mu * f
    D = np.array([-b, a])                                  # D(u) = a u - b
    term1 = Pn.polymul(np.array([-1.0, 1.0]), Pn.polymul(D, D))
    inner = Pn.polysub(D, np.array([s * mu * f, s * mu]))  # D - s mu (u + f)
    term2 = c * s * Pn.polymul(np.array([0.0, 1.0]), inner)
    return Pn.polysub(term1, term2)


def interior(p: Params, tol=1e-10):
    """All admissible interior equilibria, sorted by y (raccoons)."""
    s, mu, f, c = p.s, p.mu, p.f, p.c
    out = []
    if c == 0.0:
        G = mu * (1.0 + f)
        if G < 1.0:
            x = s * G / (1.0 - G)
            if 0.0 < x < 1.0:
                out.append((x, (1.0 - x) * (s + x)))
        return out
    for u in Pn.polyroots(cubic_coeffs(p)):
        if abs(u.imag) > 1e-9:
            continue
        u = u.real
        D = (1.0 - mu) * u - mu * f
        if u <= 1.0 + tol or D <= 0.0:
            continue
        x = s * mu * (u + f) / D
        y = (u - 1.0) / c
        if 0.0 < x < 1.0 and y > 0.0 and abs(y - (1.0 - x) * (s + x)) < 1e-7 * max(1.0, y):
            out.append((x, y))
    return sorted(out, key=lambda e: e[1])


def classify(e, p: Params):
    J = jacobian(e[0], e[1], p)
    tr, det = np.trace(J), np.linalg.det(J)
    ev = np.linalg.eigvals(J)
    if det < 0:
        kind = 'saddle'
    elif tr < 0:
        kind = 'stable focus' if np.any(np.abs(ev.imag) > 1e-12) else 'stable node'
    else:
        kind = 'unstable focus' if np.any(np.abs(ev.imag) > 1e-12) else 'unstable node'
    return dict(x=e[0], y=e[1], tr=tr, det=det, eig=ev, kind=kind, stable=(det > 0 and tr < 0))


def all_equilibria(p: Params):
    rows = []
    for e in boundary(p) + interior(p):
        rows.append(classify(e, p))
    return rows
