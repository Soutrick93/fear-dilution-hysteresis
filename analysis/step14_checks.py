# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step14_checks.py
# Also run automatically by:  python run_all.py   (job name: step14)
# Output:  results/step14_A..E.json
# Used in the paper for:  A: ramp table; B: single hysteresis window; C: Proposition "Stability of the upper state"; D: BT coefficient zeros (Prop. "Nondegeneracy away from isolated points", Fig. 7b); E: Proposition "Parity"
# ----------------------------------------------------------------------------------------------------
"""Step 14: additional checks behind the propositions on stability, parity and BT points. Five independent checks (run all, or name some):

  A  dynamic hysteresis: how the collapse and recovery fear levels of a slow ramp depend on the immigration floor
     (1e-5, 1e-7, 1e-9) and on the ramp duration (100, 300, 1000, 3000 generations).          [s0=1, c=5, delta=0.03]
  B  single window: is the bistable range {f > f_T : two interior equilibria} always ONE interval?  Counts folds
     (sign changes of the cubic's discriminant) right of f_T over a wide parameter scan, incl. uncalibrated mu.
  C  is the upper state E+ stable inside the window?  Fraction of the window where E+ is stable, for delta = 0.03
     and 0.3, and a test of the sufficient condition  h'(x+) < 0 and delta < -h'(x+)/(y+ Phi'(y+)).
  D  Bogdanov-Takens nondegeneracy: a(s0), b(s0) on a fine grid along the BT curve for c = 2, 5, 20, with a fixed
     eigenvector orientation (otherwise a and b flip sign together, an artefact); zeros located by bisection
     (codimension-3 points). Supports: 'nondegenerate except at isolated points'.
  E  parity for general dilution functions: number of interior equilibria is odd when Lambda < 0 and even when
     Lambda > 0, for rational, exponential, squared and sigmoid Phi (random parameter sets).

Usage:  python analysis/step14_checks.py            (all, about 10-20 min on 4+ cores)
        python analysis/step14_checks.py A C        (only A and C)
Outputs: results/step14_<X>.json, results/quicklook/step14_<X>.png, and a summary printed at the end.
"""
import json, sys, pathlib, time, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, rhs_log, jacobian
from qtds_fear.equilibria import interior, classify, raccoon_free_stable
from qtds_fear.bifurcation import f_transcritical, c_star, _discriminant
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'; QL = OUT / 'quicklook'; QL.mkdir(parents=True, exist_ok=True)
DRAFT = os.environ.get('QTDS_DRAFT', '0') == '1'
WORKERS = int(os.environ.get('QTDS_WORKERS', min(4, max(1, (os.cpu_count() or 2) - 1))))
summary = {}


# ------------------------------------------------------------------------------------------------- A
def ramp_run(args):
    imm, gens = args
    P = Params(s0=1.0, c=5.0, delta=0.03); gen = 1 / P.delta; FMAX = 3.2
    t_up = t_down = gens * gen; t_hold = 0.4 * gens * gen; t_end = t_up + t_hold + t_down + 0.1 * gens * gen
    def f_t(t):
        if t < t_up: return FMAX * t / t_up
        if t < t_up + t_hold: return FMAX
        return max(FMAX * (1 - (t - t_up - t_hold) / t_down), 0.0)
    z0 = np.log(interior(P.with_(f=0.0))[-1])
    F = lambda t, w: rhs_log(t, w, P.with_(f=f_t(t))) + np.array([0.0, imm / np.exp(w[1])])
    ts = np.linspace(0, t_end, 20000 if not DRAFT else 3000)
    sol = solve_ivp(F, (0, t_end), z0, t_eval=ts, method='LSODA', rtol=1e-9, atol=1e-11, max_step=gen / 4)
    y = np.exp(sol.y[1]); f = np.array([f_t(t) for t in sol.t]); half = np.argmax(sol.t >= t_up + t_hold)
    y_ref = y[0]
    low = np.where(y < 0.05 * y_ref)[0]
    if len(low) == 0:                       # ramp too fast: raccoons never collapsed (rate-induced escape)
        return dict(immigration=imm, ramp_generations=gens, f_collapse=None, f_recover=None, collapsed=False)
    i_c = int(low[0]); f_collapse = float(f[i_c])
    up = np.where((np.arange(len(y)) > i_c) & (y > 0.5 * y_ref))[0]
    f_recover = float(f[up[0]]) if len(up) else None
    return dict(immigration=imm, ramp_generations=gens, f_collapse=f_collapse, f_recover=f_recover, collapsed=True)


def check_A():
    P = Params(s0=1.0, c=5.0); fT = f_transcritical(P); fF = float(__import__('qtds_fear.bifurcation', fromlist=['x']).fold_points(P)[0])
    jobs = [(imm, g) for imm in (1e-5, 1e-7, 1e-9) for g in ((100, 300, 1000, 3000) if not DRAFT else (300,))]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(ramp_run, jobs))
    for r in rows:
        print(f"  A: immigration {r['immigration']:.0e}, ramp {r['ramp_generations']:>4} gen: "
              f"collapse at f = {r['f_collapse']}, recovery at f = {r['f_recover']}   (static: f_F = {fF:.3f}, f_T = {fT:.3f})")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for g in sorted(set(r['ramp_generations'] for r in rows)):
        rr = [r for r in rows if r['ramp_generations'] == g]
        ax[0].semilogx([r['immigration'] for r in rr], [r['f_collapse'] or np.nan for r in rr], 'o-', label=f'{g} gen')
        ax[1].semilogx([r['immigration'] for r in rr], [r['f_recover'] or np.nan for r in rr], 'o-', label=f'{g} gen')
    ax[0].axhline(fF, ls='--', c='k'); ax[1].axhline(fT, ls='--', c='k')
    ax[0].set(xlabel='immigration floor', ylabel='f at collapse', title='up-ramp (dashed: static fold)')
    ax[1].set(xlabel='immigration floor', ylabel='f at recovery', title='down-ramp (dashed: static f_T)')
    ax[0].legend(); fig.tight_layout(); fig.savefig(QL / 'step14_A.png', dpi=130)
    summary['A'] = dict(f_T=fT, f_F=fF, rows=rows)


# ------------------------------------------------------------------------------------------------- B
def windows(p, fmax=20.0):
    """Number of disjoint f-intervals right of f_T with two interior equilibria (via discriminant sign changes,
    confirmed by root counts)."""
    fT = f_transcritical(p)
    if fT is None:
        return None
    fs = np.concatenate([fT + np.geomspace(1e-9, 1e-2, 60), np.linspace(fT + 1e-2, fmax, 1500 if not DRAFT else 300)])
    n2 = np.array([len(interior(p.with_(f=float(f)))) == 2 for f in fs])
    starts = int(np.sum(n2[1:] & ~n2[:-1])) + int(n2[0])
    return starts


def b_task(args):
    s0, c, xs, kept = args
    p = Params(s0=s0, c=c, mu=xs / (s0 + xs), q=1 / kept - 1)
    return dict(s0=s0, c=c, x_star=xs, kept=kept, backward=bool(c > (c_star(p) or np.inf)), n_windows=windows(p))


def check_B():
    rng = np.random.default_rng(14)
    n = 4000 if not DRAFT else 200
    jobs = [(float(10 ** rng.uniform(-2, 1)), float(10 ** rng.uniform(-1.5, 2)), float(rng.uniform(0.02, 0.4)),
             float(rng.uniform(0.15, 0.8))) for _ in range(n)]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(b_task, jobs, chunksize=20))
    rows = [r for r in rows if r['n_windows'] is not None]
    cnt = {}
    for r in rows:
        k = f"{'backward' if r['backward'] else 'forward'}: {r['n_windows']} window(s)"; cnt[k] = cnt.get(k, 0) + 1
    multi = [r for r in rows if r['n_windows'] > 1]
    print('  B:', dict(sorted(cnt.items())), f'; cases with more than one window: {len(multi)}')
    summary['B'] = dict(counts=cnt, multi_window_examples=multi[:20], n=len(rows))


# ------------------------------------------------------------------------------------------------- C
def c_task(args):
    s0, c = args
    P = Params(s0=s0, c=c); fT = f_transcritical(P)
    from qtds_fear.bifurcation import fold_by_discriminant
    fF = fold_by_discriminant(P, fmax=20.0)
    if fT is None or fF is None:
        return None
    out = dict(s0=s0, c=c, f_T=fT, f_F=fF)
    fs = np.linspace(fT + 1e-6, fF - 1e-6, 40 if not DRAFT else 10)
    for dl in (0.03, 0.3):
        stable, cond, cond_violations = 0, 0, 0
        for f in fs:
            q = P.with_(f=float(f), delta=dl); eq = interior(q)
            if len(eq) != 2: continue
            x, y = eq[1]; k = classify(eq[1], q)
            s = q.s; g = x / (s + x); hp = 1 - 2 * x - s
            Phip = q.c * q.f / (1 + q.c * y + q.f) ** 2
            sufficient = hp < 0 and dl < -hp / (y * Phip)
            stable += k['stable']; cond += sufficient
            cond_violations += (sufficient and not k['stable'])
        out[f'frac_stable_delta_{dl}'] = stable / len(fs)
        out[f'frac_sufficient_delta_{dl}'] = cond / len(fs)
        out[f'violations_delta_{dl}'] = cond_violations
    return out


def check_C():
    s0s = np.geomspace(0.01, 5, 26 if not DRAFT else 6); cs = np.geomspace(0.5, 80, 20 if not DRAFT else 5)
    jobs = [(float(s0), float(c)) for s0 in s0s for c in cs]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = [r for r in ex.map(c_task, jobs, chunksize=4) if r]
    viol = sum(r['violations_delta_0.03'] + r['violations_delta_0.3'] for r in rows)
    for dl in (0.03, 0.3):
        fr = np.array([r[f'frac_stable_delta_{dl}'] for r in rows])
        print(f'  C: delta = {dl}: E+ stable on the whole window in {np.mean(fr > 0.999):.0%} of (s0, c) cells, '
              f'on part of it in {np.mean((fr > 0) & (fr <= 0.999)):.0%}, never in {np.mean(fr == 0):.0%}')
    print(f'  C: sufficient condition violated (condition true but E+ unstable): {viol} times (must be 0)')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a_, dl in zip(ax, (0.03, 0.3)):
        Z = np.full((len(cs), len(s0s)), np.nan)
        for r in rows:
            i = int(np.argmin(np.abs(cs - r['c']))); j = int(np.argmin(np.abs(s0s - r['s0'])))
            Z[i, j] = r[f'frac_stable_delta_{dl}']
        im = a_.pcolormesh(s0s, cs, Z, shading='auto', vmin=0, vmax=1, cmap='viridis')
        a_.set(xscale='log', yscale='log', xlabel='s0', ylabel='c', title=f'fraction of window with E+ stable, delta={dl}')
        plt.colorbar(im, ax=a_)
    fig.tight_layout(); fig.savefig(QL / 'step14_C.png', dpi=130)
    summary['C'] = dict(rows=rows, sufficient_condition_violations=viol)


# ------------------------------------------------------------------------------------------------- D
def d_task(args):
    c, s0 = args
    from qtds_fear.btnf import bt_point, coefficients
    r = bt_point(s0, c)
    if r is None: return None
    p, e = r; a, b, _ = coefficients(p, e)
    return dict(c=c, s0=s0, delta_BT=p.delta, f_BT=p.f, a=a, b=b)


def check_D():
    from qtds_fear.btnf import bt_point, coefficients
    grid = np.geomspace(0.003, 3.0, 120 if not DRAFT else 20)
    jobs = [(c, float(s0)) for c in (2.0, 5.0, 20.0) for s0 in grid]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = [r for r in ex.map(d_task, jobs, chunksize=4) if r]
    zeros = []
    for c in (2.0, 5.0, 20.0):
        rr = [r for r in rows if r['c'] == c]
        for key in ('a', 'b'):
            for r1, r2 in zip(rr[:-1], rr[1:]):
                if np.sign(r1[key]) != np.sign(r2[key]):
                    def fz(s0):
                        q = bt_point(s0, c); return coefficients(*q)[0 if key == 'a' else 1]
                    try:
                        s_z = brentq(fz, r1['s0'], r2['s0'], xtol=1e-10)
                        zeros.append(dict(c=c, coefficient=key, s0=s_z, delta_BT=bt_point(s_z, c)[0].delta))
                    except Exception as err:
                        zeros.append(dict(c=c, coefficient=key, bracket=[r1['s0'], r2['s0']], error=str(err)))
    min_abs = {f'c={c}': dict(min_abs_a=float(min(abs(r['a']) for r in rows if r['c'] == c)),
                              min_abs_b=float(min(abs(r['b']) for r in rows if r['c'] == c))) for c in (2.0, 5.0, 20.0)}
    for z in zeros:
        print(f"  D: {z['coefficient']} = 0 at c = {z['c']}, s0 = {z.get('s0')}, delta_BT = {z.get('delta_BT')}")
    print(f'  D: {len(rows)} BT points; isolated zeros found: {len(zeros)}')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for c in (2.0, 5.0, 20.0):
        rr = [r for r in rows if r['c'] == c]
        ax[0].semilogx([r['s0'] for r in rr], [r['a'] for r in rr], label=f'c={c}')
        ax[1].semilogx([r['s0'] for r in rr], [r['b'] for r in rr], label=f'c={c}')
    for a_, k in zip(ax, 'ab'):
        a_.axhline(0, c='k', lw=0.6); a_.set(xlabel='s0', ylabel=k, title=f'BT coefficient {k} along the BT curve'); a_.legend()
    fig.tight_layout(); fig.savefig(QL / 'step14_D.png', dpi=130)
    summary['D'] = dict(n_points=len(rows), zeros=zeros, min_abs=min_abs, rows=rows)


# ------------------------------------------------------------------------------------------------- E
PHI = {'rational': lambda y, f, c: (1 + c * y) / (1 + c * y + f),
       'exponential': lambda y, f, c: 1 / (1 + f * np.exp(-c * y)),
       'squared': lambda y, f, c: 1 / (1 + f / (1 + c * y) ** 2),
       'sigmoid': lambda y, f, c: 1 / (1 + f / (1 + (c * y) ** 2))}


def e_count(Phi, p, n=40000):
    s = p.s; xs = np.linspace(1e-7, 1 - 1e-9, n); ys = (1 - xs) * (s + xs)
    H = xs / (s + xs) * Phi(ys, p.f, p.c) - p.mu
    return int(np.sum(np.sign(H[:-1]) != np.sign(H[1:])))


def check_E():
    rng = np.random.default_rng(15); res = {k: dict(ok=0, bad=0, max_count=0) for k in PHI}
    n = 3000 if not DRAFT else 200
    for _ in range(n):
        s0 = 10 ** rng.uniform(-2, 1); xs = rng.uniform(0.02, 0.6)
        p = Params(s0=s0, mu=xs / (s0 + xs), c=10 ** rng.uniform(-1, 2), f=rng.uniform(0, 6), q=10 ** rng.uniform(-1, 1))
        for k, Phi in PHI.items():
            Lam = Phi(0.0, p.f, p.c) / (1 + p.s) - p.mu          # > 0 : E1 unstable (mesopredator invades)
            if abs(Lam) < 1e-9: continue
            m = e_count(Phi, p)
            ok = (m % 2 == 1) if Lam > 0 else (m % 2 == 0)
            res[k]['ok' if ok else 'bad'] += 1; res[k]['max_count'] = max(res[k]['max_count'], m)
    for k, v in res.items():
        print(f"  E: {k:<12} parity rule holds in {v['ok']}/{v['ok'] + v['bad']}; max number of equilibria {v['max_count']}")
    summary['E'] = res


if __name__ == '__main__':
    sel = [a.upper() for a in sys.argv[1:]] or list('ABCDE')
    t0 = time.time()
    for k in sel:
        t1 = time.time(); print(f'[{time.strftime("%H:%M:%S")}] check {k} ...', flush=True)
        {'A': check_A, 'B': check_B, 'C': check_C, 'D': check_D, 'E': check_E}[k]()
        (OUT / f'step14_{k}.json').write_text(json.dumps(summary.get(k), indent=1, default=float))
        print(f'    done in {time.time() - t1:.0f} s', flush=True)
    print(f'all done in {time.time() - t0:.0f} s; outputs: results/step14_*.json, results/quicklook/step14_*.png')
