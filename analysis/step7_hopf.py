# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step7_hopf.py
# Also run automatically by:  python run_all.py   (job name: step7)
# Output:  results/step7.json
# Used in the paper for:  Section "Oscillations and global dynamics": Hopf points, closed form for c = 0
# ----------------------------------------------------------------------------------------------------
"""Step 7: Hopf bifurcation. (a) c = 0: closed form via the Rosenzweig-MacArthur reduction, checked against the
numerical Hopf finder. (b) c > 0: Hopf points and first Lyapunov coefficient over a wide grid (Bautin search)."""
import json, sys, pathlib, time
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params
from qtds_fear.bifurcation import hopf_points
ROOT = pathlib.Path(__file__).resolve().parents[1]
res = {}

# (a) c = 0: Hopf where x* = s mu(1+f)/(1 - mu(1+f)) equals (1 - s)/2
def fH_closed(p):
    g = lambda f: (lambda s, G: s * G / (1 - G) - (1 - s) / 2)(p.s0 * (1 + p.q * f), p.mu * (1 + f))
    fs = np.linspace(0, 10, 4001); v = [g(f) for f in fs]
    for i in range(len(fs) - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and np.sign(v[i]) != np.sign(v[i + 1]):
            return brentq(g, fs[i], fs[i + 1], xtol=1e-14)
    return None
rows = []
for s0 in (0.1, 0.2, 0.4, 0.6, 0.75):
    p = Params(s0=s0, c=0.0, delta=0.1)
    num = hopf_points(p, fmax=5.0, n=2500)
    rows.append(dict(s0=s0, f_H_closed=fH_closed(p), f_H_numeric=num[0]['f'] if num else None,
                     l1=num[0]['l1'] if num else None))
res['c0_check'] = rows
print('c=0 closed vs numeric:', [(r['s0'], round(r['f_H_closed'], 6), round(r['f_H_numeric'], 6), '%.2e' % r['l1']) for r in rows])

# (b) Bautin search: sign of l1 over (s0, c, delta)
t0 = time.time(); grid = []
for dl in (0.01, 0.05, 0.3, 1.0, 3.0):
    for cv in (0.0, 0.5, 2.0, 5.0, 20.0, 50.0):
        for s0 in np.round(np.geomspace(0.02, 0.95, 18), 4):
            p = Params(s0=float(s0), c=cv, delta=dl)
            for h in hopf_points(p, fmax=6.0, n=1500):
                grid.append(dict(delta=dl, c=cv, s0=float(s0), f_H=h['f'], x=h['x'], y=h['y'], l1=h['l1'], omega=h['omega']))
res['hopf_grid'] = grid
l1s = np.array([g['l1'] for g in grid])
res['l1_summary'] = dict(n=len(grid), n_positive=int((l1s > 0).sum()), max_l1=float(l1s.max()), min_l1=float(l1s.min()))
(ROOT / 'results' / 'step7.json').write_text(json.dumps(res, indent=1, default=float))
print('Hopf points:', len(grid), ' l1>0:', int((l1s > 0).sum()), ' max l1 = %.3e' % l1s.max(), ' time %.0fs' % (time.time() - t0))
