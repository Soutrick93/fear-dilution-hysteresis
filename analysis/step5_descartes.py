# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step5_descartes.py
# Also run automatically by:  python run_all.py   (job name: step5b)
# Output:  printed output
# Used in the paper for:  proof of Theorem "Number of coexistence equilibria" (sign pattern of the cubic coefficients)
# ----------------------------------------------------------------------------------------------------
"""Try to prove 'at most two interior equilibria' via Descartes' rule on P(1+w), w = c y > 0."""
import sympy as sp
w, s, mu, f, c = sp.symbols('w s mu f c', positive=True)
u = 1 + w
D = (1 - mu) * u - mu * f
P = sp.expand((u - 1) * D ** 2 - c * s * u * (D - s * mu * (u + f)))
poly = sp.Poly(P, w)
for k, cc in zip(range(poly.degree(), -1, -1), poly.all_coeffs()):
    print(f"w^{k}:", sp.factor(cc))

# ---- numerical confirmation of the refined theorem:
#      E1 unstable -> exactly one interior equilibrium;  E1 stable -> zero or two
import sys, pathlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params
from qtds_fear.equilibria import interior, raccoon_free_stable
rng = np.random.default_rng(7); tab = {}
for k in range(100000):
    s0 = 10 ** rng.uniform(-2, 1); c = 10 ** rng.uniform(-2, 2.5); f = 10 ** rng.uniform(-2, 1.2); xs = rng.uniform(0.01, 0.9)
    p = Params(s0=s0, mu=xs / (s0 + xs), c=c, f=f, q=10 ** rng.uniform(-1, 1))
    key = f"E1 {'stable' if raccoon_free_stable(p) else 'unstable'}: n_interior={len(interior(p))}"
    tab[key] = tab.get(key, 0) + 1
print(json.dumps(dict(sorted(tab.items())), indent=1))
(pathlib.Path(__file__).resolve().parents[1] / 'results' / 'step5_parity_table.json').write_text(json.dumps(tab, indent=1))
