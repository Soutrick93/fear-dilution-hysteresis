# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step8f_homoclinic_diagnostic.py
# Also run automatically by:  python run_all.py   (job name: step8f)
# Output:  results/step8f_*.json
# Used in the paper for:  diagnostic near f_c: period growth and distance to the saddle (Section "Canard-mediated collapse")
# ----------------------------------------------------------------------------------------------------
"""Step 8f: is the sign flip at f_hom a true homoclinic loop? Diagnostic for the period mismatch in step 8e.

At a true homoclinic loop the stable cycle just above f_hom must pass close to the saddle S, and its period must grow
like -(1/lambda_u) ln(f - f_hom). Step 8e measured periods of ~230-320 with a slope ~20x smaller than predicted,
so this script checks directly:
  (1) minimum distance from the cycle to the saddle as f -> f_hom (should go to 0 roughly like f - f_hom),
  (2) period vs ln(f - f_hom),
  (3) phase portraits: saddle, focus, the saddle's unstable manifold just below and above f_hom, and the cycle.
Default case s0 = 0.02, delta = 0.05, c = 5. Other case:  python analysis/step8f_homoclinic_diagnostic.py 0.03 0.1
Runtime: about 2-5 min. Outputs results/step8f_*.json and results/quicklook/step8f_*.png"""
import json, sys, pathlib, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, rhs_log
from qtds_fear.bifurcation import hopf_points, f_transcritical
from qtds_fear.homoclinic import find_homoclinic, saddle_and_focus
ROOT = pathlib.Path(__file__).resolve().parents[1]
s0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
dl = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
p = Params(s0=s0, c=5.0, delta=dl)
tag = f's0_{s0}_delta_{dl}'
t0 = time.time()
fT = f_transcritical(p); fH = hopf_points(p, fmax=3, n=2000)[0]['f']
roots, _, _ = find_homoclinic(p, fT + 1e-4, fH - 1e-6)
print(f'f_T={fT:.6f} f_H={fH:.6f} sign flips at {roots}', flush=True)


def orbit_from_Wu(q, t_end, n=200000):
    S, E, lu, vu, ls, vs = saddle_and_focus(q)
    if np.dot(vu, E - S) < 0: vu = -vu
    sol = solve_ivp(lambda t, w: rhs_log(t, w, q), (0, t_end), np.log(S + 1e-7 * vu), method='LSODA',
                    rtol=1e-11, atol=1e-12, dense_output=True)
    ts = np.linspace(0, sol.t[-1], n)
    return ts, np.exp(sol.sol(ts)), S, E, lu


rows = []
fh = roots[-1] if roots else None   # the flip closest to the Hopf point
if fh is not None:
    for k in (1, 2, 3, 4, 5, 6):
        df = 10.0 ** (-k) * (fH - fh); q = p.with_(f=fh + df)
        ts, Z, S, E, lu = orbit_from_Wu(q, 60000.0 / dl)
        late = ts > 0.5 * ts[-1]                          # attractor only, transient discarded
        x = Z[0, late]; tt = ts[late]
        dist = np.hypot(Z[0, late] - S[0], Z[1, late] - S[1]).min()
        up = np.where((x[:-1] < E[0]) & (x[1:] >= E[0]))[0]
        T = float(np.mean(np.diff(tt[up][-4:]))) if len(up) >= 5 else np.nan
        amp = float(Z[1, late].max() - Z[1, late].min())
        rows.append(dict(df=df, f=fh + df, min_dist_to_saddle=float(dist), period=T, y_amplitude=amp,
                         minus_1_over_lambda_u=float(-1 / lu)))
        print(f'df={df:.2e}: min distance to saddle={dist:.3e}  period={T:.1f}  y amplitude={amp:.4f}', flush=True)

    good = [r for r in rows if np.isfinite(r['period'])]
    slope = float(np.polyfit(np.log([r['df'] for r in good]), [r['period'] for r in good], 1)[0]) if len(good) > 1 else None
    dslope = float(np.polyfit(np.log([r['df'] for r in rows]), np.log([r['min_dist_to_saddle'] for r in rows]), 1)[0])
    verdict = ('TRUE HOMOCLINIC: the cycle approaches the saddle as f -> f_hom' if dslope > 0.5 else
               'NOT A SIMPLE HOMOCLINIC: the cycle stays away from the saddle; the sign flip has another cause')
    print(f'period slope dT/dln(df) = {slope}  (theory {rows[0]["minus_1_over_lambda_u"]:.0f})')
    print(f'distance scaling d ~ df^{dslope:.2f}   ->  {verdict}')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for side, col in ((-1, 'tab:red'), (1, 'tab:blue')):
        q = p.with_(f=fh + side * 1e-3 * (fH - fh))
        ts, Z, S, E, lu = orbit_from_Wu(q, 20000.0 / dl, 60000)
        ax[0].plot(Z[0], Z[1], color=col, lw=0.7, label=f'W^u at f = f_hom {"-" if side < 0 else "+"} 1e-3 (f_H - f_hom)')
    ax[0].plot(*S, 'kx', ms=9, label='saddle'); ax[0].plot(*E, 'ko', mfc='none', ms=7, label='focus')
    ax[0].set(xlabel='crabs x', ylabel='raccoons y', title=f'unstable manifold either side of the flip ({tag})')
    ax[0].legend(fontsize=7)
    ax[1].loglog([r['df'] for r in rows], [r['min_dist_to_saddle'] for r in rows], 'o-')
    ax[1].set(xlabel='f - f_hom', ylabel='min distance cycle to saddle', title=f'slope {dslope:.2f} (1 = homoclinic)')
    ax[2].semilogx([r['df'] for r in rows], [r['period'] for r in rows], 'o-')
    ax[2].set(xlabel='f - f_hom', ylabel='cycle period', title=f'dT/dln(df) = {slope:.0f}  vs theory {rows[0]["minus_1_over_lambda_u"]:.0f}')
    fig.tight_layout(); (ROOT / 'results' / 'quicklook').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / 'results' / 'quicklook' / f'step8f_{tag}.png', dpi=130)
    out = dict(s0=s0, delta=dl, f_T=fT, f_H=fH, flips=roots, f_hom_used=fh, rows=rows, period_slope=slope,
               distance_exponent=dslope, verdict=verdict, seconds=time.time() - t0)
    (ROOT / 'results' / f'step8f_{tag}.json').write_text(json.dumps(out, indent=1, default=float))
print(f'done in {time.time() - t0:.0f} s')
