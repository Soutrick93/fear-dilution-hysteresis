# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step12c_bt_normal_form.py
# Also run automatically by:  python run_all.py   (job name: step12c)
# Output:  results/step12c_bt_normal_form.json
# Used in the paper for:  Bogdanov-Takens normal-form coefficients a, b
# ----------------------------------------------------------------------------------------------------
"""Step 12c: Bogdanov-Takens nondegeneracy (item 1 of the proof list).

At each BT point (fold f_F with delta = delta_BT = -J11/j22, from step 8a) compute the quadratic normal-form
coefficients of Kuznetsov (2004), Sec. 8.4:
    A q0 = 0, A q1 = q0,  A^T p1 = 0, A^T p0 = p1,  <p0,q0> = <p1,q1> = 1, <p0,q1> = <p1,q0> = 0,
    a = 1/2 <p1, B(q0,q0)>,   b = <p0, B(q0,q0)> + <p1, B(q0,q1)>.
The BT point is nondegenerate iff a b != 0. Transversality in the parameters (f, delta): the fold is a regular curve
in f (Sotomayor, step 6) and d tr J / d delta = j22 > 0 at fixed f, so the unfolding map is regular.
Sign of a b decides which unfolding (homoclinic on which side). Runtime ~2-4 min."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, jacobian
from qtds_fear.equilibria import interior
from qtds_fear.bifurcation import fold_points, _Bn
ROOT = pathlib.Path(__file__).resolve().parents[1]


def bt_point(s0, c):
    p = Params(s0=s0, c=c, delta=1.0)
    fl = fold_points(p, fmax=8.0, n=3000)
    if not fl:
        return None
    fF = fl[0]; eq = interior(p.with_(f=fF - 1e-11))
    e = tuple(np.mean(eq, axis=0)) if len(eq) == 2 else eq[0]
    J = jacobian(*e, p.with_(f=fF)); J11, j22 = J[0, 0], J[1, 1]
    if not (J11 < 0 < j22):
        return None
    return p.with_(f=fF, delta=-J11 / j22), e


def coefficients(p, e):
    A = jacobian(*e, p)
    B = lambda v, w: np.array(_Bn(e[0], e[1], p.s0, p.q, p.c, p.f, p.mu, p.delta, *v, *w), dtype=float)
    # Jordan chain: q0 spans ker A, q1 solves A q1 = q0 (least squares, then made orthogonal-free below)
    U, S, Vt = np.linalg.svd(A); q0 = Vt[-1]
    if q0[0] < 0: q0 = -q0          # canonical orientation (see qtds_fear/btnf.py)
    q1 = np.linalg.lstsq(A, q0, rcond=None)[0]
    U2, S2, Vt2 = np.linalg.svd(A.T); p1 = Vt2[-1]
    p0 = np.linalg.lstsq(A.T, p1, rcond=None)[0]
    # normalise: <p0,q0>=1 and <p1,q1>=1 ; <p1,q0>=0 holds automatically; fix <p0,q1>=0 by p0 -> p0 - k p1
    sc = p1 @ q1; p1 = p1 / sc; p0 = p0 / sc
    p0 = p0 - (p0 @ q1) * p1
    sc0 = p0 @ q0; p0 = p0 / sc0; p1 = p1 / sc0
    checks = dict(p0q0=p0 @ q0, p1q1=p1 @ q1, p0q1=p0 @ q1, p1q0=p1 @ q0, residual=float(np.abs(A @ q1 - q0).max()),
                  eigenvalues=[complex(v).__repr__() for v in np.linalg.eigvals(A)])
    a = 0.5 * p1 @ B(q0, q0)
    b = p0 @ B(q0, q0) + p1 @ B(q0, q1)
    return float(a), float(b), checks


rows = []
for c in (2.0, 5.0, 20.0):
    for s0 in np.geomspace(0.005, 2.0, 12):
        r = bt_point(float(s0), c)
        if r is None:
            continue
        p, e = r
        a, b, chk = coefficients(p, e)
        rows.append(dict(c=c, s0=float(s0), f_BT=p.f, delta_BT=p.delta, x=e[0], y=e[1], a=a, b=b,
                         ab=a * b, nondegenerate=bool(abs(a) > 1e-8 and abs(b) > 1e-8), checks=chk))
        print(f'c={c:<5} s0={s0:.4f}: f_BT={p.f:.5f} delta_BT={p.delta:.3f}  a={a:+.4e}  b={b:+.4e}  '
              f'normalisation |<p0,q1>|={abs(chk["p0q1"]):.1e}', flush=True)
nd = sum(r['nondegenerate'] for r in rows)
signs = sorted(set(np.sign([r['ab'] for r in rows]).astype(int).tolist()))
print(f'{nd}/{len(rows)} BT points nondegenerate; sign(ab) values: {signs}')
(ROOT / 'results' / 'step12c_bt_normal_form.json').write_text(json.dumps(dict(rows=rows, nondegenerate=nd,
                                                                              n=len(rows), ab_signs=signs), indent=1,
                                                                         default=float))
