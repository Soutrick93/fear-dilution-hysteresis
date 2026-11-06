# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step6_transcritical_fold.py
# Also run automatically by:  python run_all.py   (job name: step6)
# Output:  results/step6.json
# Used in the paper for:  Theorem "Hysteresis threshold", Corollary "Closing of the window", Sotomayor checks at the folds
# ----------------------------------------------------------------------------------------------------
"""Step 6: transcritical (closed form), its direction (c*), and the saddle-node that closes the bistable window.
Outputs results/step6.json: c*(s0), fold curve f_F(c), window widths, Sotomayor checks at folds."""
import json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, jacobian
from qtds_fear.equilibria import interior
from qtds_fear.bifurcation import f_transcritical, c_star, fold_points, transcritical_slope
import sympy as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
res = {}

# (1) c* as a function of s0 (closed form) and check of the slope sign on both sides
s0s = np.round(np.geomspace(0.05, 5, 25), 5)
res['c_star_vs_s0'] = [dict(s0=float(s), f_T=float(f_transcritical(Params(s0=s))), c_star=float(c_star(Params(s0=s))))
                       for s in s0s]
chk = []
for s in (0.25, 0.5, 1.0, 2.0, 4.0):
    cs = c_star(Params(s0=s))
    lo, hi = transcritical_slope(Params(s0=s, c=0.98 * cs)), transcritical_slope(Params(s0=s, c=1.02 * cs))
    chk.append(dict(s0=s, c_star=cs, slope_below=lo, slope_above=hi, ok=bool(lo < 0 < hi)))
res['slope_sign_check'] = chk

# (2) fold curve f_F(c) for s0 = 1 and its limit as c -> c*
curve = []
p1 = Params(s0=1.0)
cs1 = c_star(p1)
for cv in np.concatenate([cs1 * (1 + np.geomspace(1e-3, 1, 12)), np.geomspace(2 * cs1, 100, 22)]):
    fl = fold_points(p1.with_(c=float(cv)), fmax=12.0, n=1500)
    curve.append(dict(c=float(cv), f_F=float(fl[0]) if fl else None))
res['fold_curve_s0_1'] = curve
res['f_T_s0_1'] = float(f_transcritical(p1)); res['c_star_s0_1'] = float(cs1)

# (3) Sotomayor saddle-node conditions at fold points (numerical, with symbolic derivatives)
x, y, s0, q, c, f, mu, dl = sp.symbols('x y s0 q c f mu delta')
s = s0 * (1 + q * f)
F = sp.Matrix([x * (1 - x) - x * y / (s + x), dl * y * (x / (s + x) * (1 + c * y) / (1 + c * y + f) - mu)])
Ff = sp.lambdify((x, y, s0, q, c, f, mu, dl), list(F.diff(f)), 'numpy')
H = [sp.lambdify((x, y, s0, q, c, f, mu, dl), sp.hessian(F[i], (x, y)), 'numpy') for i in range(2)]
sot = []
for cv in (1.0, 5.0, 20.0):
    for s0v in (0.5, 1.0, 2.0):
        p = Params(s0=s0v, c=cv)
        fl = fold_points(p, fmax=12.0, n=1500)
        if not fl:
            continue
        fF = fl[0]
        q_ = p.with_(f=fF - 1e-9)
        eq = interior(q_)
        e = eq[0] if len(eq) == 1 else tuple(np.mean(eq, axis=0))
        J = jacobian(*e, q_)
        ev, V = np.linalg.eig(J); k = int(np.argmin(np.abs(ev))); v = V[:, k].real
        evT, W = np.linalg.eig(J.T); k2 = int(np.argmin(np.abs(evT))); w = W[:, k2].real
        a = (e[0], e[1], s0v, p.q, cv, fF, p.mu, p.delta)
        t1 = float(w @ np.array(Ff(*a), float))
        t2 = float(w @ np.array([v @ np.array(H[i](*a), float) @ v for i in range(2)]))
        sot.append(dict(c=cv, s0=s0v, f_F=fF, min_abs_eig=float(np.min(np.abs(ev))), w_Ff=t1, w_D2F_vv=t2,
                        nondegenerate=bool(abs(t1) > 1e-8 and abs(t2) > 1e-8)))
res['sotomayor_saddle_node'] = sot

# (4) window width f_F - f_T against c, for three s0
ww = {}
for s0v in (0.5, 1.0, 2.0):
    p = Params(s0=s0v); fT = f_transcritical(p); cs = c_star(p); rows = []
    for cv in np.geomspace(cs * 1.01, 100, 30):
        fl = fold_points(p.with_(c=float(cv)), fmax=15.0, n=1500)
        rows.append(dict(c=float(cv), width=float(fl[0] - fT) if fl else None))
    ww[str(s0v)] = rows
res['window_width'] = ww
(ROOT / 'results' / 'step6.json').write_text(json.dumps(res, indent=1))
print('slope checks:', [r['ok'] for r in chk])
print('fold curve near c*:', [(round(r['c'], 4), None if r['f_F'] is None else round(r['f_F'], 4)) for r in curve[:5]], 'f_T =', round(res['f_T_s0_1'], 4))
print('Sotomayor:', [(r['c'], r['s0'], round(r['f_F'], 4), '%.2e' % r['w_Ff'], '%.2e' % r['w_D2F_vv'], r['nondegenerate']) for r in sot])
print('widths s0=1:', [(round(r['c'], 2), None if r['width'] is None else round(r['width'], 3)) for r in ww['1.0'][::5]])
