# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step10_chain3d.py
# Also run automatically by:  python run_all.py   (job name: step10)
# Output:  results/step10_chain3d.json
# Used in the paper for:  Appendix "Numerical extension: a living carnivore population", Fig. 8c
# ----------------------------------------------------------------------------------------------------
"""Step 10 (outlook section): does hysteresis survive when fear comes from a living carnivore population?
Quasi-static ramps of carnivore carrying capacity KP up and down for several (beta, lam)."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.chain3d import Chain, ramp_KP
ROOT = pathlib.Path(__file__).resolve().parents[1]
res = {}
KP = np.linspace(0.05, 3.2, 33)
for beta, lam in ((0.0, 0.0), (0.3, 0.0), (0.0, 0.02), (0.3, 0.02)):
    for c in (0.15, 5.0):
        p = Chain(c=c, beta=beta, lam=lam)
        up = ramp_KP(p, KP, [0.1, 0.99, 1e-3])
        down = ramp_KP(p, KP[::-1], up[-1])[::-1]
        key = f'beta={beta},lam={lam},c={c}'
        gap = np.abs(up[:, 1] - down[:, 1])
        res[key] = dict(KP=KP.tolist(), up=up.tolist(), down=down.tolist(), max_gap_y=float(gap.max()),
                        KP_window=[float(KP[i]) for i in np.where(gap > 0.05)[0][[0, -1]]] if (gap > 0.05).any() else None)
        print(key, 'max |y_up - y_down| =', round(float(gap.max()), 3), 'window:', res[key]['KP_window'], flush=True)
(ROOT / 'results' / 'step10_chain3d.json').write_text(json.dumps(res, indent=1))
