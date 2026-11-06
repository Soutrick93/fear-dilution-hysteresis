# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step8d_recheck_failures.py
# Also run automatically by:  python run_all.py   (job name: step8d)
# Output:  results/step8d_recheck.json (only if step8c reports failures)
# Used in the paper for:  recheck of flagged cases from step8c with a stiff solver
# ----------------------------------------------------------------------------------------------------
"""Step 8d: recheck global-test 'failures' with a stiff solver (Radau), tight tolerances and longer time, plus the
eigenvalues of the target equilibrium. A true failure would be an orbit that converges to something else."""
import json, sys, pathlib
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, rhs, rhs_log
from qtds_fear.equilibria import interior, classify
ROOT = pathlib.Path(__file__).resolve().parents[1]
src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'step8c_global_test.json'
fails = json.loads(src.read_text())['failures']
if not fails:
    print('step8c reported no failures: nothing to recheck')
out = []
for f in fails:
    p = Params(s0=f['s0'], c=f['c'], f=f['f'], delta=f['delta'])
    e = interior(p)[0]; k = classify(e, p)
    T = 20000.0 / f['delta']
    sol = solve_ivp(lambda t, w: rhs_log(t, w, p), (0, T), np.log(f['z0']), method='LSODA', rtol=1e-11, atol=1e-12)
    sol.y = np.exp(sol.y)          # log variables keep x, y > 0 exactly
    end = sol.y[:, -1]; dist = float(np.hypot(*(end - np.array(e))))
    out.append(dict(case=f, eig=[str(v) for v in k['eig']], kind=k['kind'], end_radau=end.tolist(), dist=dist,
                    min_x=float(sol.y[0].min()), min_y=float(sol.y[1].min()), converged=dist < 1e-6))
    print(f"c={f['c']:.3f} delta={f['delta']:.3f}: {k['kind']}, eig={np.round(k['eig'],5)}, Radau end dist={dist:.2e}, "
          f"min x={sol.y[0].min():.3e}, min y={sol.y[1].min():.3e}")
(ROOT / 'results' / 'step8d_recheck.json').write_text(json.dumps(out, indent=1))
