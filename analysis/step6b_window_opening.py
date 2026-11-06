# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step6b_window_opening.py
# Also run automatically by:  python run_all.py   (job name: step6b)
# Output:  results/step6b_window_opening.json
# Used in the paper for:  Proposition "Opening of the window" (coefficient C vs computed folds) and the threshold shift in the elasticity remark
# ----------------------------------------------------------------------------------------------------
"""Step 6b: check of Proposition "Opening of the window". Compares the closed-form coefficient C in
f_F - f_T ~ C (c - c*)^2 with folds computed from the discriminant of the cubic, for s0 = 0.5, 1, 2, and predicts the
shift sqrt(df / C) of the hysteresis threshold found by root counting at f_T + df (df = 1e-5, as in step 11).
Runtime < 1 min."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params
from qtds_fear.bifurcation import f_transcritical, c_star, fold_by_discriminant, opening_coefficient
ROOT = pathlib.Path(__file__).resolve().parents[1]
out = {}
for s0 in (0.5, 1.0, 2.0):
    P = Params(s0=s0); fT = f_transcritical(P); cs = c_star(P); C = opening_coefficient(P)
    rows = []
    for d in (0.3, 0.1, 0.03, 0.01, 0.003):
        fF = fold_by_discriminant(P.with_(c=cs + d), fmax=10)
        rows.append(dict(c_minus_cstar=d, ratio=(fF - fT) / d ** 2))
    out[str(s0)] = dict(f_T=fT, c_star=cs, C=C, predicted_shift_at_df_1em5=float(np.sqrt(1e-5 / C)), rows=rows)
    print(f's0={s0}: C={C:.4f}; (f_F-f_T)/(c-c*)^2 =', ', '.join(f"{r['ratio']:.3f}" for r in rows),
          f'; predicted counting shift {np.sqrt(1e-5 / C):.5f}')
(ROOT / 'results' / 'step6b_window_opening.json').write_text(json.dumps(out, indent=1, default=float))
