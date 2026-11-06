# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""3-D extension: a living top carnivore P replaces the playback cue, f = kappa P.

    x' = x(1-x) - x y/(s+x),                         s = s0 (1 + q kappa P)
    y' = delta y [ x/(s+x) (1+cy)/(1+cy+kappa P) - mu - lam P ]     (lam: lethal predation on raccoons)
    P' = eps P [ 1 - P/KP + beta y ]                  (generalist carnivore; beta: benefit from raccoons)

KP is the carnivores' carrying capacity (habitat, persecution). KP -> 0 is extirpation; raising KP is reintroduction.
With beta = 0, P -> KP and the planar model is recovered with f = kappa KP.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp
from .model import calibrated_mu, Q_PLAYBACK


@dataclass(frozen=True)
class Chain:
    s0: float = 1.0
    mu: float = None
    q: float = Q_PLAYBACK
    c: float = 5.0
    delta: float = 0.05
    kappa: float = 1.0
    lam: float = 0.0
    eps: float = 0.02
    KP: float = 1.0
    beta: float = 0.0

    def __post_init__(self):
        if self.mu is None:
            object.__setattr__(self, 'mu', calibrated_mu(self.s0))

    def with_(self, **kw):
        return replace(self, **kw)


def rhs3(t, z, p: Chain):
    x, y, P = z
    f = p.kappa * P
    s = p.s0 * (1 + p.q * f)
    g = x / (s + x)
    phi = (1 + p.c * y) / (1 + p.c * y + f)
    return np.array([x * (1 - x) - y * g,
                     p.delta * y * (g * phi - p.mu - p.lam * P),
                     p.eps * P * (1 - P / max(p.KP, 1e-9) + p.beta * y)])


def ramp_KP(p: Chain, KP_values, z0, settle=15000.0, immigration=1e-6):
    z = np.array(z0, float); out = []
    for K in KP_values:
        q = p.with_(KP=float(K))
        sol = solve_ivp(lambda t, w: rhs3(t, w, q) + np.array([0, immigration, 1e-9]), (0, settle), z,
                        method='LSODA', rtol=1e-9, atol=1e-12)
        z = sol.y[:, -1]; out.append(z.copy())
    return np.array(out)


def scenario(p: Chain, KP_of_t, z0, t_end, n=3000):
    """Time course under a carnivore-habitat schedule (extirpation, then reintroduction)."""
    def F(t, w):
        return rhs3(t, w, p.with_(KP=float(KP_of_t(t)))) + np.array([0, 1e-6, 1e-9])
    ts = np.linspace(0, t_end, n)
    sol = solve_ivp(F, (0, t_end), z0, t_eval=ts, method='LSODA', rtol=1e-9, atol=1e-12, max_step=50.0)
    return sol.t, sol.y
