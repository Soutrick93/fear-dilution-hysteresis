# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step12a_hopf_direction.py
# Also run automatically by:  python run_all.py   (job name: step12a)
# Output:  results/step12a_hopf_direction.json
# Used in the paper for:  Theorem "Supercritical Hopf bifurcation, c = 0" and the Lyapunov-coefficient grid
# ----------------------------------------------------------------------------------------------------
"""Step 12a: direction of the Hopf bifurcation (item 1 of the proof list).

Part A (proof for c = 0). With c = 0 the model is exactly Rosenzweig-MacArthur (RM) with death rate mu(1+f) and time
scale delta/(1+f) (verified in qtds_fear.symbolic). We compute the first Lyapunov coefficient symbolically at the RM
Hopf point x* = (1-s)/2 and show it is negative for every 0 < s < 1: the Hopf bifurcation is supercritical.

Part B (c > 0). l1 depends continuously on c, so supercriticality persists for c < c1(s0, delta). We give
computer-assisted evidence over a wide grid, including uncalibrated mu (not tied to the Suraci numbers), and report
the largest l1 found (all negative = no Bautin point in the grid).
Runtime: Part A ~1 min, Part B ~5-15 min.  Usage: python analysis/step12a_hopf_direction.py [A|B]"""
import json, sys, pathlib, time
import numpy as np
import sympy as sp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]
part = sys.argv[1] if len(sys.argv) > 1 else 'AB'
res = {}

if 'A' in part:
    # RM in our scaling: x' = x(1-x) - x y/(s+x),  y' = d y (x/(s+x) - m),  Hopf at x* = (1-s)/2.
    x, y, s, d, m = sp.symbols('x y s d m', positive=True)
    F = sp.Matrix([x * (1 - x) - x * y / (s + x), d * y * (x / (s + x) - m)])
    X = sp.Matrix([x, y])
    xs = (1 - s) / 2
    ms = xs / (s + xs)                       # death rate that puts the equilibrium at x*
    ys = (1 - xs) * (s + xs)
    sub = {x: xs, y: ys, m: ms}
    J = F.jacobian(X).subs(sub).applyfunc(sp.simplify)
    tr = sp.simplify(J.trace()); det = sp.simplify(J.det())
    omega = sp.sqrt(det)
    # Planar formula (Guckenheimer-Holmes 3.4.11 / Kuznetsov 3.5): bring J to [[0,-w],[w,0]] and evaluate.
    a11, a12, a21, a22 = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    # columns of the transform T so that T^{-1} J T = [[0,-w],[w,0]]
    T = sp.Matrix([[a12, 0], [a22, -omega]]) if a12 != 0 else None
    u, v = sp.symbols('u v')
    Xuv = sp.Matrix([xs, ys]) + T * sp.Matrix([u, v])
    G = (T.inv() * F.subs({x: Xuv[0], y: Xuv[1], m: ms})).applyfunc(lambda e: e)
    f1, g1 = G[0], G[1]
    D = lambda e, *vs: sp.diff(e, *vs).subs({u: 0, v: 0})
    a_GH = sp.Rational(1, 16) * (D(f1, u, u, u) + D(f1, u, v, v) + D(g1, u, u, v) + D(g1, v, v, v)) + \
        1 / (16 * omega) * (D(f1, u, v) * (D(f1, u, u) + D(f1, v, v)) - D(g1, u, v) * (D(g1, u, u) + D(g1, v, v))
                            - D(f1, u, u) * D(g1, u, u) + D(f1, v, v) * D(g1, v, v))
    a_simpl = sp.factor(sp.simplify(a_GH))
    print('trace at x* =', tr, '   det =', sp.factor(det))
    print('first Lyapunov (GH) coefficient a(s, d) =', a_simpl)
    vals = [float(a_simpl.subs({s: sv, d: dv})) for sv in np.linspace(0.01, 0.99, 50) for dv in (0.01, 0.1, 1, 10)]
    print('sign check on 200 points of 0 < s < 1, d in {0.01,0.1,1,10}: max a =', max(vals))
    res['A'] = dict(trace=str(tr), det=str(sp.factor(det)), a=str(a_simpl), max_a_on_grid=max(vals),
                    supercritical_everywhere_on_grid=max(vals) < 0)

if 'B' in part:
    from qtds_fear.model import Params
    from qtds_fear.bifurcation import hopf_points
    t0 = time.time(); rows = []
    for xstar in (0.02, 0.05, 0.1, 0.2, 0.3):                         # uncalibrated mu: crab level without fear
        for dl in (0.01, 0.05, 0.3, 1.0, 3.0):
            for c in (0.0, 0.3, 1.0, 5.0, 20.0, 100.0):
                for s0 in np.geomspace(0.01, 0.95, 14):
                    p = Params(s0=float(s0), mu=xstar / (float(s0) + xstar), c=c, delta=dl)
                    for h in hopf_points(p, fmax=8.0, n=1500):
                        rows.append(dict(xstar=xstar, delta=dl, c=c, s0=float(s0), f_H=h['f'], l1=h['l1']))
        print(f'x* = {xstar}: {len(rows)} Hopf points so far, max l1 = {max(r["l1"] for r in rows):.3e} '
              f'({time.time() - t0:.0f}s)', flush=True)
    l1 = np.array([r['l1'] for r in rows])
    worst = rows[int(np.argmax(l1))]
    print(f'TOTAL {len(rows)} Hopf points; l1 > 0 in {(l1 > 0).sum()}; max l1 = {l1.max():.3e} at {worst}')
    res['B'] = dict(n=len(rows), n_positive=int((l1 > 0).sum()), max_l1=float(l1.max()), worst=worst, rows=rows)

(ROOT / 'results' / 'step12a_hopf_direction.json').write_text(json.dumps(res, indent=1, default=float))
