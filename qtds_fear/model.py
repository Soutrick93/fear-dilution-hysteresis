# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Fear-diluted mesopredator model (dimensionless form).

    x' = x(1 - x) - x y / (s + x)                                  prey (crabs), scaled by K
    y' = delta * y * [ x/(s+x) * (1 + c y)/(1 + c y + f) - mu ]     mesopredator (raccoons)

    s = s(f) = s0 (1 + q f)   fear suppresses foraging (raises half-saturation)
    (1 + c y)/(1 + c y + f)   fear cost on reproduction, diluted by group size (c)

Dimensional origin (see LOGBOOK, step 3):
    dR/dT = rho R (1 - R/K) - a R M / (1 + q f + a h R)
    dM/dT = M [ e a R / (1 + q f + a h R) / (1 + f/(1 + gamma M)) - m ]
    x = R/K, y = M/(rho h K), t = rho T;  s0 = 1/(a h K), mu = m h / e, delta = e/(h rho), c = gamma rho h K
"""
from dataclasses import dataclass, replace, asdict
import numpy as np

# ---- calibration constants taken from Suraci et al. 2016 ------------------------------------------
FORAGING_KEPT_AT_PLAYBACK = 0.34      # raccoons foraged 66% less under carnivore playback
Q_PLAYBACK = 1.0 / FORAGING_KEPT_AT_PLAYBACK - 1.0   # 1/(1+q) = 0.34 at f = 1  ->  q = 1.941
CRAB_FRACTION_WITH_RACCOONS = 0.10    # raccoons reduce intertidal crabs by 90%
CRAB_GAIN_28_DAYS = 1.97              # +97% intertidal crabs after a 28-day playback


@dataclass(frozen=True)
class Params:
    s0: float = 1.0     # half-saturation without fear
    mu: float = None    # raccoon death / conversion ratio (None -> calibrated)
    q: float = Q_PLAYBACK
    c: float = 5.0      # group-dilution strength of the fear cost
    f: float = 0.0      # fear cue (f = 1 is the playback level)
    delta: float = 0.05 # raccoon / crab time-scale ratio

    def __post_init__(self):
        if self.mu is None:
            object.__setattr__(self, 'mu', calibrated_mu(self.s0))

    @property
    def s(self):
        return self.s0 * (1.0 + self.q * self.f)

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)


def calibrated_mu(s0, x_star=CRAB_FRACTION_WITH_RACCOONS):
    """mu such that, with no fear, crabs sit at x_star of carrying capacity:  x*/(s0+x*) = mu."""
    return x_star / (s0 + x_star)


def rhs(t, z, p: Params):
    x, y = z
    s = p.s
    g = x / (s + x)
    phi = (1.0 + p.c * y) / (1.0 + p.c * y + p.f)
    return np.array([x * (1.0 - x) - y * g, p.delta * y * (g * phi - p.mu)])


def jacobian(x, y, p: Params):
    s = p.s
    g = x / (s + x)
    gx = s / (s + x) ** 2
    phi = (1.0 + p.c * y) / (1.0 + p.c * y + p.f)
    phiy = p.c * p.f / (1.0 + p.c * y + p.f) ** 2
    return np.array([[1.0 - 2.0 * x - y * gx, -g],
                     [p.delta * y * gx * phi, p.delta * (g * phi - p.mu) + p.delta * y * g * phiy]])


def per_capita_growth_y(x, y, p: Params):
    """Raccoon per-capita growth at fixed prey level x: shows the emergent (fear-induced) Allee effect."""
    s = p.s
    g = x / (s + x)
    phi = (1.0 + p.c * y) / (1.0 + p.c * y + p.f)
    return p.delta * (g * phi - p.mu)


def prey_nullcline(x, p: Params):
    """y on the prey nullcline (x' = 0, x > 0): the hump y = (1 - x)(s + x). Also the slow manifold as delta -> 0."""
    return (1.0 - x) * (p.s + x)


def rhs_log(t, w, p: Params):
    """Same model in log variables w = (ln x, ln y): positivity is automatic and orbits that pass very close to
    x = 0 or y = 0 (crab or raccoon near-extinction) stay accurate. Use for long or stiff integrations."""
    x, y = np.exp(w[0]), np.exp(w[1])
    s = p.s
    g = x / (s + x)
    phi = (1.0 + p.c * y) / (1.0 + p.c * y + p.f)
    return np.array([(1.0 - x) - y / (s + x), p.delta * (g * phi - p.mu)])
