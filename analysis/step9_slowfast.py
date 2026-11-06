# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step9_slowfast.py
# Also run automatically by:  python run_all.py   (job name: step9)
# Output:  results/step9.json
# Used in the paper for:  Section "What a 28-day experiment measures" (slow-fast separation)
# ----------------------------------------------------------------------------------------------------
"""Step 9: slow-fast analysis and the 28-day playback.
(a) Crab renewal rate needed for +97% in 28 days with raccoons frozen (fast subsystem), for several s0.
(b) Full-model check: over 28 days raccoons barely move when delta is raccoon-like.
(c) Canard explosion: raccoon cycle amplitude vs f near the Hopf point, small vs moderate delta."""
import json, sys, pathlib
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, rhs, CRAB_GAIN_28_DAYS, CRAB_FRACTION_WITH_RACCOONS
from qtds_fear.equilibria import interior
from qtds_fear.slowfast import time_to_gain, cycle_amplitudes
from qtds_fear.bifurcation import hopf_points
ROOT = pathlib.Path(__file__).resolve().parents[1]
res = {}
rows = []
for s0 in (0.5, 1.0, 2.0):
    p0 = Params(s0=s0, f=0.0); e0 = interior(p0)[-1]
    T = time_to_gain(p0, p0.with_(f=1.0), e0[0], e0[1], CRAB_GAIN_28_DAYS)
    r_day = T / 28.0
    # raccoon-like delta: raccoon population rate ~0.5/yr vs crab rate r_day*365
    dl = 0.5 / (r_day * 365.0)
    p1 = p0.with_(f=1.0, delta=dl)
    sol = solve_ivp(lambda t, z: rhs(t, z, p1), (0, 28 * r_day), list(e0), rtol=1e-10, atol=1e-12)
    rows.append(dict(s0=s0, x0=e0[0], y0=e0[1], fast_time=T, crab_rate_per_day=r_day, crab_rate_per_year=r_day * 365,
                     delta_raccoon_like=dl, full_model_crab_gain_28d=sol.y[0, -1] / e0[0],
                     raccoon_change_28d_percent=100 * (sol.y[1, -1] / e0[1] - 1)))
    print(rows[-1])
res['playback_28d'] = rows

can = {}
for dl in (0.02, 0.2):
    p = Params(s0=0.3, c=5.0, delta=dl)
    H = hopf_points(p, fmax=3, n=2000)[0]
    fs = np.concatenate([np.linspace(0.0, H['f'] - 0.02, 12), np.linspace(H['f'] - 0.02, H['f'] + 0.005, 30)])
    e = interior(p.with_(f=float(fs[0])))[-1]
    A = cycle_amplitudes(p, fs, [e[0] * 1.05, e[1]], t_transient=4000 / dl ** 0.5, t_record=2000 / dl ** 0.5)
    can[str(dl)] = dict(f_H=H['f'], l1=H['l1'], f=fs.tolist(), amp_y=A[:, 0].tolist(), amp_x=A[:, 1].tolist())
    jumps = np.abs(np.diff(A[:, 0])) / np.diff(fs)
    print(f"delta={dl}: f_H={H['f']:.4f}, max d(amp)/df = {jumps.max():.1f}")
res['canard'] = can
(ROOT / 'results' / 'step9.json').write_text(json.dumps(res, indent=1, default=float))
