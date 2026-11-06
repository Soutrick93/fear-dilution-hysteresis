# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step5_equilibria.py
# Also run automatically by:  python run_all.py   (job name: step5)
# Output:  results/step5_equilibria.json, results/step5_parity_table.json
# Used in the paper for:  Theorem "Number of coexistence equilibria" (random-parameter test in the remark)
# ----------------------------------------------------------------------------------------------------
"""Step 5: interior equilibria. Checks (i) agreement with the earlier independent solver, (ii) the parity rule
P(1) = -c s [1 - mu(1+f)(1+s)], (iii) whether three admissible interior equilibria can occur (random search),
(iv) whether a root u > 1 can have D < 0 (spurious, x < 0)."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from numpy.polynomial import polynomial as Pn
from qtds_fear.model import Params
from qtds_fear.equilibria import interior, cubic_coeffs, raccoon_free_stable

rng = np.random.default_rng(1)
res = dict(n_samples=0, parity_violations=0, max_interior=0, three_examples=[], spurious_Dneg=0, counts={})
for k in range(200000):
    s0 = 10 ** rng.uniform(-2, 1); c = 10 ** rng.uniform(-2, 2.5); f = 10 ** rng.uniform(-2, 1.2)
    xs = rng.uniform(0.01, 0.9)                      # sets mu through the no-fear crab level
    p = Params(s0=s0, mu=xs / (s0 + xs), c=c, f=f, q=10 ** rng.uniform(-1, 1))
    eq = interior(p)
    n = len(eq); res['counts'][n] = res['counts'].get(n, 0) + 1
    res['max_interior'] = max(res['max_interior'], n)
    if n == 3 and len(res['three_examples']) < 5:
        res['three_examples'].append(dict(s0=s0, mu=p.mu, q=p.q, c=c, f=f, eq=eq))
    # parity: E1 unstable -> odd number of roots in u > 1 ; stable -> even
    roots = [r.real for r in Pn.polyroots(cubic_coeffs(p)) if abs(r.imag) < 1e-9 and r.real > 1 + 1e-10]
    n_roots = len(roots)
    if (n_roots % 2 == 1) == raccoon_free_stable(p):
        res['parity_violations'] += 1
    for u in roots:
        if (1 - p.mu) * u - p.mu * p.f <= 0:
            res['spurious_Dneg'] += 1
    res['n_samples'] += 1
res['counts'] = {str(k): v for k, v in sorted(res['counts'].items())}
out = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'step5_equilibria.json'
out.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: v for k, v in res.items() if k != 'three_examples'}, indent=1), '\nthree examples:', len(res['three_examples']))
