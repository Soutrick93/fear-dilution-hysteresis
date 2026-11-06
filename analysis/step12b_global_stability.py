# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step12b_global_stability.py
# Also run automatically by:  python run_all.py   (job name: step12b)
# Output:  results/step12b_global_stability.json
# Used in the paper for:  Theorem "Global dynamics, c = 0" and the Dulac-function search
# ----------------------------------------------------------------------------------------------------
"""Step 12b: global dynamics (item 1 of the proof list).

Part A, c = 0 (proof). The model is Rosenzweig-MacArthur. With h(x) = (1-x)(s+x) (prey nullcline) and
    V = integral_{x*}^{x} (g(xi) - g(x*))/g(xi) d xi  +  (1/delta') [y - y* - y* ln(y/y*)],
g(x) = x/(s+x), delta' = delta/(1+f), one gets dV/dt = (g(x) - g(x*))(h(x) - h(x*)) (checked symbolically below),
whose sign is that of -(x - x*)^2 (x + x* - (1 - s)). This is <= 0 on 0 < x < 1 whenever
x* >= 1 - s, so the coexistence state is globally stable there (LaSalle). For (1-s)/2 <= x* < 1-s, global stability
of RM follows from uniqueness of limit cycles (cite Cheng 1981; Kuang & Freedman 1988; check before citing).

Part B, c > 0 (computer-assisted). Dulac function B = x^a y^b on the absorbing box
0 < x < 1, 0 < y < y_max (y_max from the bound V = x + y/delta <= (1+mu delta)^2/(4 mu delta)).
If div(B F) < 0 on the box there are no periodic orbits, so by Poincare-Bendixson every orbit tends to an
equilibrium. Random parameter sets; report the fraction certified, split by regime.
Runtime ~5-10 min.  Usage: python analysis/step12b_global_stability.py [A|B]"""
import json, sys, pathlib, time
import numpy as np
import sympy as sp
from scipy.optimize import minimize
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]
part = sys.argv[1] if len(sys.argv) > 1 else 'AB'
res = {}

if 'A' in part:
    x, y, s, dp, xs, xi = sp.symbols('x y s dp xs xi', positive=True)
    ys = (1 - xs) * (s + xs)
    m = xs / (s + xs)
    F1 = x * (1 - x) - x * y / (s + x)
    F2 = dp * y * (x / (s + x) - m)
    g = x / (s + x); gs = m
    Vx = (g - gs) / g                               # dV/dx
    Vy = (1 - ys / y) / dp                          # dV/dy
    Vdot = sp.simplify(Vx * F1 + Vy * F2)
    target = (g - gs) * ((1 - x) * (s + x) - ys)
    ok = sp.simplify(Vdot - target) == 0
    print('dV/dt = (g(x) - g(x*))(h(x) - h(x*)) :', ok)
    # g increasing, so sign(g - g*) = sign(x - x*); h(x) - h(x*) = (x* - x)(x + x* - (1 - s))
    fac = sp.factor(sp.simplify(target))
    print('factorised dV/dt =', fac)
    res['A'] = dict(identity_verified=bool(ok), factorised=str(fac),
                    condition='dV/dt <= 0 for all 0 < x < 1 iff x* >= 1 - s')

if 'B' in part:
    from qtds_fear.model import Params
    from qtds_fear.equilibria import interior, classify, raccoon_free_stable

    def div_BF(a, b, X, Y, p):
        s, c, f, mu, dl = p.s, p.c, p.f, p.mu, p.delta
        g = X / (s + X); gx = s / (s + X) ** 2
        phi = (1 + c * Y) / (1 + c * Y + f); phiy = c * f / (1 + c * Y + f) ** 2
        F1 = X * (1 - X) - Y * g; F2 = dl * Y * (g * phi - mu)
        dF1 = 1 - 2 * X - Y * gx; dF2 = dl * (g * phi - mu) + dl * Y * g * phiy
        return (a / X) * F1 + dF1 + (b / Y) * F2 + dF2      # div(BF)/B, B > 0

    def certify(p, n=160):
        ymax = p.delta * (1 + p.mu * p.delta) ** 2 / (4 * p.mu * p.delta)
        X, Y = np.meshgrid(np.geomspace(1e-4, 1, n), np.geomspace(1e-5, ymax, n))
        worst = lambda ab: float(np.max(div_BF(ab[0], ab[1], X, Y, p)))
        best = min((minimize(worst, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
                    for x0 in ([-1.0, -1.0], [0.0, -1.0], [-1.0, 0.0], [-2.0, -1.0])), key=lambda r: r.fun)
        return best.fun < 0, best.x.tolist(), best.fun

    rng = np.random.default_rng(11); rows = []; t0 = time.time()
    while len(rows) < 300 and time.time() - t0 < 900:
        s0 = 10 ** rng.uniform(-1.5, 0.7); c = 10 ** rng.uniform(-1, 1.7); f = rng.uniform(0, 3)
        dl = 10 ** rng.uniform(-1.5, 0.3)
        p = Params(s0=s0, c=c, f=f, delta=dl)
        eq = interior(p)
        st = [classify(e, p)['stable'] for e in eq]
        regime = ('coexistence' if len(eq) == 1 and st[0] else 'bistable' if len(eq) == 2 else
                  'raccoon-free' if not eq and raccoon_free_stable(p) else 'unstable focus (cycle)' if eq else 'other')
        if regime == 'unstable focus (cycle)':
            continue                                   # cycles exist there; Dulac cannot (and should not) apply
        ok, ab, val = certify(p)
        rows.append(dict(s0=s0, c=c, f=f, delta=dl, regime=regime, certified=bool(ok), a=ab[0], b=ab[1], max_div=val))
        if len(rows) % 25 == 0:
            print(f'{len(rows)} sets, certified {sum(r["certified"] for r in rows)} ({time.time() - t0:.0f}s)', flush=True)
    summ = {}
    for r in rows:
        k = r['regime']; summ.setdefault(k, [0, 0]); summ[k][0] += 1; summ[k][1] += r['certified']
    for k, (n, ok) in summ.items():
        print(f'{k:<14} {ok}/{n} certified (no periodic orbits)')
    res['B'] = dict(summary={k: dict(n=n, certified=ok) for k, (n, ok) in summ.items()}, rows=rows)

(ROOT / 'results' / 'step12b_global_stability.json').write_text(json.dumps(res, indent=1, default=float))
