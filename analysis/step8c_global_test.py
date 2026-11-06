# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step8c_global_test.py
# Also run automatically by:  python run_all.py   (job name: step8c)
# Output:  results/step8c_global_test.json
# Used in the paper for:  global convergence test supporting Conjecture "global stability" (c > 0)
# ----------------------------------------------------------------------------------------------------
"""Step 8c: computer-assisted global convergence test. Random parameter sets with E1 unstable and a unique, locally
stable coexistence state; several random initial conditions each; report any orbit not converging to it."""
import json, sys, pathlib, time
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, rhs, rhs_log
from qtds_fear.equilibria import interior, raccoon_free_stable, classify
ROOT = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(3); tests = 0; failures = []; t0 = time.time()
N_SETS = 6 if __import__('os').environ.get('QTDS_DRAFT', '0') == '1' else 60
while tests < N_SETS and time.time() - t0 < 3000:
    s0 = 10 ** rng.uniform(-1.5, 0.7); c = 10 ** rng.uniform(-1, 1.7); f = rng.uniform(0, 3); dl = 10 ** rng.uniform(-1.3, 0)
    q = Params(s0=s0, c=c, f=f, delta=dl)
    if raccoon_free_stable(q): continue
    eq = interior(q)
    if len(eq) != 1 or not classify(eq[0], q)['stable']: continue
    tests += 1
    for _ in range(5):
        z0 = [rng.uniform(0.01, 1.0), rng.uniform(0.01, 3.0)]
        target = np.array(eq[0])
        # first try: plain variables, moderate tolerance; only the end point is stored (memory-safe)
        sol = solve_ivp(lambda t, w: rhs(t, w, q), (0, 1500.0 / dl), z0, method='LSODA', rtol=1e-7, atol=1e-10,
                        t_eval=[1500.0 / dl])
        end = np.asarray(sol.y, dtype=float)[:, -1] if np.size(sol.y) else np.array([np.nan, np.nan])
        if not np.all(np.isfinite(end)) or np.hypot(*(end - target)) > 1e-3:
            # fallback: log variables, tight tolerances, longer horizon (LSODA can fail when prey pass near x = 0)
            sol = solve_ivp(lambda t, w: rhs_log(t, w, q), (0, 20000.0 / dl), np.log(z0), method='LSODA',
                            rtol=1e-11, atol=1e-12, t_eval=[20000.0 / dl])
            end = np.exp(np.asarray(sol.y, dtype=float)[:, -1]) if np.size(sol.y) else np.array([np.nan, np.nan])
        if not np.all(np.isfinite(end)) or np.hypot(*(end - target)) > 1e-3:
            failures.append(dict(s0=s0, c=c, f=f, delta=dl, z0=z0, end=end.tolist(), eq=list(eq[0]))); break
out = dict(parameter_sets=tests, initial_conditions_each=5, failures=failures, seconds=time.time() - t0)
(ROOT / 'results' / 'step8c_global_test.json').write_text(json.dumps(out, indent=1, default=float))
print(tests, 'sets;', len(failures), 'failures;', round(time.time() - t0), 's')
