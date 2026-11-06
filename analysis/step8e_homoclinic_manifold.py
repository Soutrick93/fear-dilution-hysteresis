# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step8e_homoclinic_manifold.py
# Also run automatically by:  python run_all.py   (job name: step8e)
# Output:  results/step8e_homoclinic.json
# Used in the paper for:  canard threshold f_c by the manifold method (Section "Canard-mediated collapse", Figs 6 and 7)
# ----------------------------------------------------------------------------------------------------
"""Step 8e: homoclinic orbits by the manifold method (qtds_fear/homoclinic.py). Cross-check: the cycle period blows
up logarithmically, T ~ -(1/lambda_u) ln(f - f_hom), as f decreases to f_hom.

Cases run in parallel on all CPU cores; each finished case is printed and saved at once
(results/step8e_homoclinic.json), so partial results survive an interruption.
Usage:  python analysis/step8e_homoclinic_manifold.py           all 14 cases   (about 5-15 min on 4+ cores)
        python analysis/step8e_homoclinic_manifold.py quick     disputed case only (s0 = 0.02, delta = 0.05)
        python analysis/step8e_homoclinic_manifold.py noperiod  skip the period cross-check (faster)"""
import json, sys, pathlib, time, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params
from qtds_fear.bifurcation import f_transcritical, hopf_points
from qtds_fear.homoclinic import find_homoclinic, cycle_period, saddle_and_focus
ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_case(dl, s0, do_period=True):
    t0 = time.time()
    p = Params(s0=s0, c=5.0, delta=dl)
    fT = f_transcritical(p); fH = hopf_points(p, fmax=3, n=2000)[0]['f']
    roots, fs, D = find_homoclinic(p, fT + 1e-4, fH - 1e-6)
    row = dict(delta=dl, s0=s0, f_T=fT, f_H=fH, f_hom=roots,
               scan_sign=[int(np.sign(v)) if np.isfinite(v) else None for v in D])
    if roots and do_period:
        fh = roots[0]; lu = saddle_and_focus(p.with_(f=fh))[2]
        per = []
        for k in (1, 2, 3):
            f = fh + 10.0 ** (-k) * (fH - fh)
            per.append(dict(f=f, df=f - fh, T=cycle_period(p.with_(f=f))))
        good = [r for r in per if np.isfinite(r['T'])]
        row['period_check'] = per; row['predicted_slope'] = float(-1.0 / lu)
        if len(good) >= 2:
            row['fitted_slope'] = float(np.polyfit(np.log([r['df'] for r in good]), [r['T'] for r in good], 1)[0])
    row['seconds'] = time.time() - t0
    return row


def fmt(row):
    s = (f"delta={row['delta']} s0={row['s0']}: f_T={row['f_T']:.5f} "
         f"f_hom={[round(r, 9) for r in row['f_hom']] or 'none (cycle survives to f_T)'} f_H={row['f_H']:.5f}")
    if 'fitted_slope' in row:
        s += f"  period slope {row['fitted_slope']:.0f} vs predicted {row['predicted_slope']:.0f}"
    return s + f"  ({row['seconds']:.0f}s)"


if __name__ == '__main__':
    quick, do_period = 'quick' in sys.argv, 'noperiod' not in sys.argv
    cases = [(0.05, 0.02)] if quick else [(dl, s0) for dl in (0.05, 0.1)
                                          for s0 in (0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1)]
    out = ROOT / 'results' / ('step8e_homoclinic_quick.json' if quick else 'step8e_homoclinic.json')
    workers = max(1, min(len(cases), int(os.environ.get('QTDS_WORKERS', min(4, max(1, (os.cpu_count() or 2) - 1))))))
    print(f'{len(cases)} cases on {workers} cores; each line appears when its case finishes.', flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_case, dl, s0, do_period): (dl, s0) for dl, s0 in cases}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                row = fu.result()
            except Exception as e:
                dl, s0 = futs[fu]; row = dict(delta=dl, s0=s0, error=repr(e))
                print(f'[{k}/{len(cases)}] delta={dl} s0={s0}: ERROR {e!r}', flush=True)
            else:
                print(f'[{k}/{len(cases)}] ' + fmt(row), flush=True)
            rows.append(row)
            out.write_text(json.dumps(sorted(rows, key=lambda r: (r['delta'], r['s0'])), indent=1, default=float))
    print(f'done in {time.time() - t0:.0f} s -> {out}')
