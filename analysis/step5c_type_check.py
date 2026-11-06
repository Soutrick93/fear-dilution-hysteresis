# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step5c_type_check.py
# Also run automatically by:  python run_all.py   (job name: step5c)
# Output:  results/step5c_type_check.json
# Used in the paper for:  Remark after Proposition "Type of the coexistence equilibria" (sign det J = sign P'(u); lower state is a saddle)
# ----------------------------------------------------------------------------------------------------
"""Step 5c: numerical check of Lemma "det" and Proposition "Type of the coexistence equilibria".
For random parameter sets, at every interior equilibrium compare  sign det J  with  sign P'(u)  (P = the cubic in
u = 1 + c y, Lemma: they agree), and in every case with two interior equilibria check that the lower one (fewer
mesopredators) is a saddle and the upper one has det J > 0. Runtime < 1 min."""
import json, sys, pathlib
import numpy as np
from numpy.polynomial import polynomial as Pn
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, jacobian
from qtds_fear.equilibria import interior, cubic_coeffs

rng = np.random.default_rng(55)
N = 20000
n_eq = agree = n_bist = lower_saddle = upper_detpos = 0
bad = []
for k in range(N):
    s0 = 10 ** rng.uniform(-2, 1); c = 10 ** rng.uniform(-2, 2.5); f = 10 ** rng.uniform(-2, 1.2)
    xs = rng.uniform(0.01, 0.9)
    p = Params(s0=s0, mu=xs / (s0 + xs), c=c, f=f, q=10 ** rng.uniform(-1, 1), delta=10 ** rng.uniform(-2, 0.5))
    eq = interior(p)
    dP = Pn.polyder(cubic_coeffs(p))
    dets = []
    for x, y in eq:
        det = np.linalg.det(jacobian(x, y, p)); dets.append(det)
        u = 1 + c * y; sp = np.sign(Pn.polyval(u, dP))
        n_eq += 1
        if np.sign(det) == sp:
            agree += 1
        elif len(bad) < 10:
            bad.append(dict(s0=s0, c=c, f=f, x=x, y=y, det=det, dP=float(Pn.polyval(u, dP))))
    if len(eq) == 2:
        n_bist += 1; lower_saddle += dets[0] < 0; upper_detpos += dets[1] > 0
res = dict(parameter_sets=N, interior_equilibria=n_eq, sign_agreement=agree, cases_with_two=n_bist,
           lower_is_saddle=int(lower_saddle), upper_det_positive=int(upper_detpos), disagreements=bad)
(pathlib.Path(__file__).resolve().parents[1] / 'results' / 'step5c_type_check.json').write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: v for k, v in res.items() if k != 'disagreements'}, indent=1))
