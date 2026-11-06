# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step8_global.py
# Also run automatically by:  python run_all.py   (job name: step8)
# Output:  results/step8.json (basin fractions)
# Used in the paper for:  basin fractions of the persisting attractor (supplementary; the canard threshold is computed in step8e)
# ----------------------------------------------------------------------------------------------------
"""Step 8: fraction of initial conditions in the basin of the mesopredator-persisting attractor, for five fear levels
(s0 = 1, c = 5, delta = 0.05). The canard/homoclinic threshold is computed in step8e; the global convergence test is
step8c. Runtime 5-15 min."""
import json, sys, pathlib, time
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, rhs, rhs_log
from qtds_fear.equilibria import interior, raccoon_free_stable, classify
from qtds_fear.bifurcation import f_transcritical, fold_points, hopf_points
ROOT = pathlib.Path(__file__).resolve().parents[1]
res = {}

DRAFT = __import__('os').environ.get('QTDS_DRAFT', '0') == '1'


def fate(p, z0, T=60000.0):
    """Mesopredator density at time T. Only the end point is stored (t_eval=[T]), and the integration runs in
    log variables, so memory stays small even when orbits pass close to x = 0."""
    sol = solve_ivp(lambda t, w: rhs_log(t, w, p), (0, T), np.log(z0), method='LSODA', rtol=1e-9, atol=1e-12,
                    t_eval=[T])
    return float(np.exp(sol.y[1, -1]))


# The earlier homoclinic search by bisection on the long-run fate of single orbits was ill-conditioned near the
# canard threshold and is superseded by the manifold method of step8e; it has been removed from this script.
res = {}
p = Params(s0=1.0, c=5.0, delta=0.05)
n = 6 if DRAFT else 21
xs = np.linspace(0.02, 1.0, n); ys = np.linspace(0.01, 3.0, n)
basin = []
for f in (1.0, 1.5, 2.0, 2.3, 2.45):
    q = p.with_(f=f); n_ok = 0
    for x0 in xs:
        for y0 in ys:
            n_ok += fate(q, [x0, y0], T=6000.0) > 1e-3
    basin.append(dict(f=f, fraction_persist=n_ok / (len(xs) * len(ys))))
    print('basin', basin[-1], flush=True)
res['basin_fraction'] = basin

# (iii) global convergence test: random parameter sets with E1 unstable and a stable coexistence state
(ROOT / 'results' / 'step8.json').write_text(json.dumps(res, indent=1, default=float))
