# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step8a_bt.py
# Also run automatically by:  python run_all.py   (job name: step8a)
# Output:  results/step8a_bt.json
# Used in the paper for:  Theorem "Bogdanov-Takens curve", Fig. 7b
# ----------------------------------------------------------------------------------------------------
"""Step 8a: Bogdanov-Takens points. Equilibria (and so the fold f_F) do not depend on delta; delta only scales the
second row of J. At the fold det J = 0, and tr J = J11 + delta * j22 with j22 = y g Phi_y > 0 (fear dilution makes
the raccoon equation self-reinforcing). Hence the fold is a BT point exactly at
        delta_BT = -J11 / j22   (when J11 < 0),
a closed-form BT curve. Below it the fold is a saddle-node of a stable node; above it, of an unstable node."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, jacobian
from qtds_fear.equilibria import interior
from qtds_fear.bifurcation import fold_points
ROOT = pathlib.Path(__file__).resolve().parents[1]

def bt_delta(s0, c):
    p = Params(s0=s0, c=c, delta=1.0)
    fl = fold_points(p, fmax=8.0, n=3000)
    if not fl: return None
    fF = fl[0]; eq = interior(p.with_(f=fF - 1e-11))
    e = tuple(np.mean(eq, axis=0)) if len(eq) == 2 else eq[0]
    J = jacobian(*e, p.with_(f=fF))           # delta = 1: J[1,1] = j22 (g*Phi = mu at equilibrium)
    J11, j22 = J[0, 0], J[1, 1]
    return dict(s0=s0, c=c, f_F=fF, x=e[0], y=e[1], J11=J11, j22=j22, delta_BT=(-J11 / j22) if J11 < 0 < j22 else None)

out = {}
for c in (2.0, 5.0, 20.0):
    rows = [bt_delta(float(s0), c) for s0 in np.geomspace(0.005, 3.0, 30)]
    out[str(c)] = [r for r in rows if r]
    print(f'c={c}:', [(round(r['s0'], 3), round(r['delta_BT'], 2) if r['delta_BT'] else None) for r in out[str(c)][::5]])
# check: at delta_BT both eigenvalues vanish
r = [x for x in out['5.0'] if x['delta_BT']][10]
p = Params(s0=r['s0'], c=5.0, delta=r['delta_BT'], f=r['f_F'])
ev = np.linalg.eigvals(jacobian(r['x'], r['y'], p))
out['check'] = dict(s0=r['s0'], delta_BT=r['delta_BT'], f_F=r['f_F'], eigenvalues=[complex(v).__repr__() for v in ev])
print('eigenvalues at a BT point:', ev)
(ROOT / 'results' / 'step8a_bt.json').write_text(json.dumps(out, indent=1, default=float))
