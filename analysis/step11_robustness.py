# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step11_robustness.py
# Also run automatically by:  python run_all.py   (job name: step11)
# Output:  results/step11_robustness.json
# Used in the paper for:  Theorem "Elasticity criterion" (remark with four dilution functions), Fig. 8b
# ----------------------------------------------------------------------------------------------------
"""Step 11: robustness of the hysteresis threshold to the form of the fear-dilution function.

General model: y' = delta y [ g(x) Phi(y; f) - mu ],  g = x/(s+x),  Phi(0; f) = 1/(1+f).
First-order expansion of the coexistence branch at E1 = (1,0), f = f_T, gives (derivation in LOGBOOK, step 11):

    branch is backward (hysteresis)  iff   E := Phi_y(0; f_T) / Phi(0; f_T)  >  s_T / (1 + s_T)^2 .

E is the low-density elasticity of the fear-diluted birth factor. Since s/(1+s)^2 <= 1/4, E > 1/4 guarantees
hysteresis for every foraging parameter. For the paper's form Phi = (1+cy)/(1+cy+f), E = c f/(1+f) and the
criterion reproduces c* exactly.

Test: four dilution functions, threshold c found by direct root counting vs the prediction."""
import json, sys, pathlib
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params
from qtds_fear.bifurcation import f_transcritical
ROOT = pathlib.Path(__file__).resolve().parents[1]

FORMS = {
    'rational (paper)  (1+cy)/(1+cy+f)': (lambda y, f, c: (1 + c * y) / (1 + c * y + f), lambda f, c: c * f / (1 + f)),
    'exponential  1/(1+f e^{-cy})':      (lambda y, f, c: 1 / (1 + f * np.exp(-c * y)), lambda f, c: c * f / (1 + f)),
    'squared  1/(1+f/(1+cy)^2)':         (lambda y, f, c: 1 / (1 + f / (1 + c * y) ** 2), lambda f, c: 2 * c * f / (1 + f)),
    'sigmoid  1/(1+f/(1+(cy)^2))':       (lambda y, f, c: 1 / (1 + f / (1 + (c * y) ** 2)), lambda f, c: 0.0 * c),
}


def n_interior(Phi, p, c, n=20000):
    """Count interior equilibria: roots in x of g(x) Phi(y(x)) - mu along the prey nullcline y = (1-x)(s+x)."""
    s = p.s; xs = np.linspace(1e-6, 1 - 1e-9, n)
    ys = (1 - xs) * (s + xs)
    h = xs / (s + xs) * Phi(ys, p.f, c) - p.mu
    return int(np.sum(np.sign(h[:-1]) != np.sign(h[1:])))


res = {}
for s0 in (0.3, 1.0, 3.0):
    base = Params(s0=s0); fT = f_transcritical(base); sT = base.with_(f=fT).s
    target = sT / (1 + sT) ** 2
    for name, (Phi, E) in FORMS.items():
        # predicted threshold: E(f_T, c) = target, E linear in c (or zero)
        e1 = E(fT, 1.0)
        c_pred = target / e1 if e1 > 0 else None
        # numerical threshold: smallest c with 2 interior equilibria just past f_T
        df = 1e-5
        has = lambda c: n_interior(Phi, base.with_(f=fT + df), c) >= 2
        grid = np.geomspace(1e-3, 1e3, 200)
        flags = [has(c) for c in grid]
        if any(flags):
            i = flags.index(True)
            lo, hi = (grid[i - 1] if i > 0 else grid[0] / 2), grid[i]
            for _ in range(50):
                mid = np.sqrt(lo * hi)
                if has(mid): hi = mid
                else: lo = mid
            c_num = hi
        else:
            c_num = None
        # also: does ANY bistability exist (a fold away from f_T) for large c?
        any_bist = any(n_interior(Phi, base.with_(f=f), 50.0) >= 2 for f in np.linspace(0.01, 6, 120))
        res[f's0={s0} | {name}'] = dict(f_T=fT, s_T=sT, criterion=target, c_pred=c_pred, c_num=c_num,
                                         bistability_at_c50=any_bist)
        print(f"s0={s0:<4} {name:<36} c_pred={c_pred if c_pred is None else round(c_pred, 4)}  "
              f"c_num={c_num if c_num is None else round(c_num, 4)}  bistable somewhere at c=50: {any_bist}", flush=True)
(ROOT / 'results' / 'step11_robustness.json').write_text(json.dumps(res, indent=1, default=float))
