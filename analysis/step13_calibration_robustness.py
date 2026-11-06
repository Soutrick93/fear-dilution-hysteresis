# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step13_calibration_robustness.py
# Also run automatically by:  python run_all.py   (job name: step13)
# Output:  results/step13_calibration.json
# Used in the paper for:  Corollary "Robustness to calibration", Fig. 8a
# ----------------------------------------------------------------------------------------------------
"""Step 13: does the headline result depend on the Suraci calibration? (item 2)

The calibration fixes two numbers: crabs at x* = 10% of K with raccoons and no fear (sets mu), and 34% of foraging
kept at playback (sets q). Here both are varied:  x* in [0.02, 0.4],  foraging kept in [0.15, 0.8].
For each pair we compute sup over s0 in [0.01, 10] of c*(s0) (closed form), i.e. the dilution strength above which
fear-driven cascade reversal is hysteretic for EVERY foraging saturation, and check it against the universal
elasticity bound (E > 1/4, i.e. c f_T/(1+f_T) > 1/4).
Runtime < 1 min. Output: results/step13_calibration.json and results/quicklook/step13_calibration.png"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params
from qtds_fear.bifurcation import c_star, f_transcritical
ROOT = pathlib.Path(__file__).resolve().parents[1]

XS = np.round(np.linspace(0.02, 0.4, 20), 4)
KEPT = np.round(np.linspace(0.15, 0.8, 14), 4)
S0 = np.geomspace(0.01, 10, 400)
sup = np.full((len(KEPT), len(XS)), np.nan); arg = np.full_like(sup, np.nan)
for i, kept in enumerate(KEPT):
    q = 1 / kept - 1
    for j, xs in enumerate(XS):
        vals = []
        for s0 in S0:
            p = Params(s0=float(s0), mu=xs / (s0 + xs), q=q)
            cs = c_star(p)
            vals.append(np.nan if cs is None else cs)
        vals = np.array(vals)
        if np.isfinite(vals).any():
            k = int(np.nanargmax(vals)); sup[i, j] = vals[k]; arg[i, j] = S0[k]

i0 = int(np.argmin(np.abs(KEPT - 0.34))); j0 = int(np.argmin(np.abs(XS - 0.1)))
print(f'Suraci calibration (kept 0.34, x* 0.10): sup c* = {sup[i0, j0]:.3f} at s0 = {arg[i0, j0]:.3f}')
print(f'over the whole calibration box: sup c* ranges {np.nanmin(sup):.3f} to {np.nanmax(sup):.3f}')
print('note: c* f_T/(1+f_T) = s_T/(1+s_T)^2 <= 1/4 is an identity, so c f_T/(1+f_T) > 1/4 always gives hysteresis')
fig, ax = plt.subplots(figsize=(6, 4.2))
im = ax.pcolormesh(XS, KEPT, sup, shading='auto', cmap='viridis')
cs_ = ax.contour(XS, KEPT, sup, levels=[0.25, 0.5, 0.75, 1.0], colors='w', linewidths=0.8)
ax.clabel(cs_, fmt='%.2f', fontsize=7)
ax.plot(0.1, 0.34, 'r*', ms=12, label='Suraci calibration')
ax.set(xlabel='crabs with raccoons, no fear (x*)', ylabel='foraging kept at playback',
       title='sup over s0 of c*: dilution needed for hysteresis')
plt.colorbar(im, ax=ax); ax.legend(loc='upper right', fontsize=7); fig.tight_layout()
(ROOT / 'results' / 'quicklook').mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / 'results' / 'quicklook' / 'step13_calibration.png', dpi=130)
(ROOT / 'results' / 'step13_calibration.json').write_text(json.dumps(dict(
    x_star=XS.tolist(), foraging_kept=KEPT.tolist(), sup_c_star=sup.tolist(), argmax_s0=arg.tolist(),
    suraci=dict(sup_c_star=sup[i0, j0], s0=arg[i0, j0]), range=[float(np.nanmin(sup)), float(np.nanmax(sup))]),
    indent=1))
