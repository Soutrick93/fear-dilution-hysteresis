# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/step15_dynamics.py
# Also run automatically by:  python run_all.py   (job name: step15)
# Output:  results/step15_A..F.json
# Used in the paper for:  A: ramp-rate scaling and entry-exit test; B: calibration sensitivity of the dynamic results; C: critical boost duration; D: extended Hopf scan; E: wide delta_BT scan; F: error bar on f_c (Fig. 9)
# ----------------------------------------------------------------------------------------------------
"""Step 15: dynamic hysteresis, calibration sensitivity and accuracy checks (run all, or name some parts).

  A  Ramp-rate scaling of dynamic hysteresis. Collapse and recovery fear levels for ramp durations from 30 to 10 000
     mesopredator generations and three immigration floors; loop width versus ramp rate; and a test of the
     entry-exit (way-in/way-out) prediction for the recovery point:
         delta * int_{t_T}^{t_out} lambda(f(t)) dt = ln(y_th / y(t_T)),   lambda(f) = 1/((1+f)(1+s(f))) - mu,
     where t_T is when the down-ramp crosses f_T and y(t_T) is read from the simulation.        [s0=1, c=5]
  B  Sensitivity of the dynamic predictions (Figs 4 and 6) to the calibration: no-fear prey level x* in
     {0.05, 0.1, 0.2}, retained foraging 1/(1+q) in {0.2, 0.34, 0.5}, delta in {0.01, 0.03, 0.1}.
     Reports f_T, c*, f_F, ramp collapse/recovery, lock-in by a temporary boost, and whether the canard regime
     (Hopf point inside the bistable window at s0 = 0.02) exists.
  C  Critical boost duration D_c(f_b): shortest boost to f_boost that removes the mesopredator permanently when fear
     then returns to the background level f_b inside the window. Tests D_c -> infinity as f_b -> f_T.
  D  First Lyapunov coefficient on an extended grid s0 in [0.01, 3], c in {0..100}, delta in [0.01, 3],
     x* in {0.02, 0.1, 0.3}; counts near-degenerate Hopf points.
  E  delta_BT on a wide grid (s0 in [0.003, 10], c in [0.3, 300], x* and q varied): minimum and distribution,
     to decide whether "delta_BT > 4" can be stated beyond c = 2, 5, 20.
  F  Accuracy of the canard threshold f_c (s0 = 0.02, c = 5, delta = 0.05): f_c recomputed with integration
     tolerances 1e-9..1e-12, initial displacements 1e-6..1e-8 and section positions 0.02..0.1. The spread is the
     error estimate reported in the paper.

Usage:  python analysis/step15_dynamics.py            (all; about 1-3 h with 4 workers, mostly D and F)
        set QTDS_WORKERS=2 (Windows cmd) or $env:QTDS_WORKERS=2 (PowerShell) to use fewer workers / less memory
        python analysis/step15_dynamics.py A C        (only A and C)
        QTDS_DRAFT=1 python analysis/step15_dynamics.py   (tiny grids, a few minutes: checks that everything runs)
Outputs: results/step15_<X>.json, results/quicklook/step15_<X>.png, summary printed at the end.
"""
import json, sys, pathlib, time, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qtds_fear.model import Params, rhs_log, jacobian
from qtds_fear.equilibria import interior, classify
from qtds_fear.bifurcation import f_transcritical, c_star, fold_by_discriminant, hopf_points
from qtds_fear.homoclinic import saddle_and_focus
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'; QL = OUT / 'quicklook'; QL.mkdir(parents=True, exist_ok=True)
DRAFT = os.environ.get('QTDS_DRAFT', '0') == '1'
WORKERS = int(os.environ.get('QTDS_WORKERS', min(4, max(1, (os.cpu_count() or 2) - 1))))


def robust_map(fn, jobs, label=''):
    """Like Pool.map, but survives a worker that dies (out of memory, or a solver crash on one parameter set).
    On a broken pool, the unfinished jobs are retried one at a time in fresh single-worker pools; a job that kills
    its worker again is recorded as None and listed in results/step15_crashed_<label>.json."""
    from concurrent.futures.process import BrokenProcessPool
    out = [None] * len(jobs); todo = list(range(len(jobs))); done = 0
    try:
        with ProcessPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(fn, jobs[i]): i for i in todo}
            for fu in __import__('concurrent.futures').futures.as_completed(futs):
                i = futs[fu]
                try:
                    out[i] = fu.result(); todo.remove(i); done += 1
                    if done % max(1, len(jobs) // 20) == 0:
                        print(f'    [{label}] {done}/{len(jobs)} done', flush=True)
                except BrokenProcessPool:
                    raise
                except Exception as e:
                    print(f'    [{label}] job {jobs[i]} raised {e!r}; skipped', flush=True); todo.remove(i)
    except BrokenProcessPool:
        print(f'    [{label}] a worker died; retrying {len(todo)} unfinished jobs one by one', flush=True)
    crashed = []
    for i in list(todo):
        try:
            with ProcessPoolExecutor(1) as ex:
                out[i] = ex.submit(fn, jobs[i]).result()
        except BrokenProcessPool:
            print(f'    [{label}] job {jobs[i]} kills its worker; skipped', flush=True); crashed.append(jobs[i])
        except Exception as e:
            print(f'    [{label}] job {jobs[i]} raised {e!r}; skipped', flush=True)
    if crashed:
        (OUT / f'step15_crashed_{label}.json').write_text(json.dumps(crashed, default=float))
    return out


def P_of(s0=1.0, c=5.0, x_star=0.1, kept=0.34, delta=0.03, f=0.0):
    return Params(s0=s0, c=c, mu=x_star / (s0 + x_star), q=1.0 / kept - 1.0, delta=delta, f=f)


def save(name, obj):
    (OUT / f'step15_{name}.json').write_text(json.dumps(obj, indent=1, default=float))


# ------------------------------------------------------------------------------------------------- ramps (A, B)
def ramp(P, gens, imm, fmax):
    """Up-ramp 0 -> fmax, hold, down-ramp -> 0 (each ramp lasts `gens` generations). Returns collapse/recovery levels
    and the entry-exit prediction of the recovery point."""
    gen = 1 / P.delta
    t_up = t_down = gens * gen; t_hold = 0.4 * gens * gen; t_end = t_up + t_hold + t_down + 0.2 * gens * gen
    def f_t(t):
        if t < t_up: return fmax * t / t_up
        if t < t_up + t_hold: return fmax
        return max(fmax * (1 - (t - t_up - t_hold) / t_down), 0.0)
    eq0 = interior(P.with_(f=0.0))
    if not eq0:
        return None
    z0 = np.log(eq0[-1])
    F = lambda t, w: rhs_log(t, w, P.with_(f=f_t(t))) + np.array([0.0, imm / np.exp(w[1])])
    ts = np.linspace(0, t_end, 4000 if DRAFT else 30000)
    sol = solve_ivp(F, (0, t_end), z0, t_eval=ts, method='LSODA', rtol=1e-9, atol=1e-11, max_step=gen / 4)
    y = np.exp(sol.y[1]); f = np.array([f_t(t) for t in sol.t]); y_ref = y[0]
    out = dict(gens=gens, immigration=imm, collapsed=False, f_collapse=None, f_recover=None, f_exit_sim=None,
               f_exit_pred=None, y_at_fT=None, y_min=float(y.min()))
    low = np.where(y < 0.05 * y_ref)[0]
    if len(low) == 0:
        return out
    i_c = int(low[0]); out.update(collapsed=True, f_collapse=float(f[i_c]))
    after = np.arange(len(y)) > i_c
    up50 = np.where(after & (y > 0.5 * y_ref))[0]; up5 = np.where(after & (y > 0.05 * y_ref))[0]
    out['f_recover'] = float(f[up50[0]]) if len(up50) else None
    out['f_exit_sim'] = float(f[up5[0]]) if len(up5) else None
    # entry-exit prediction: start where the down-ramp crosses f_T
    fT = f_transcritical(P)
    t_T = t_up + t_hold + (1 - fT / fmax) * t_down
    iT = int(np.searchsorted(sol.t, t_T)); yT = y[min(iT, len(y) - 1)]; out['y_at_fT'] = float(yT)
    lam = lambda ff: 1.0 / ((1 + ff) * (1 + P.s0 * (1 + P.q * ff))) - P.mu
    target = np.log(0.05 * y_ref / yT)
    if target > 0:
        # f decreases at rate fmax/t_down; int lambda dt = (t_down/fmax) int_{f_out}^{f_T} lambda df
        G = lambda fo: P.delta * (t_down / fmax) * quad(lam, fo, fT)[0] - target
        if G(0.0) > 0:
            out['f_exit_pred'] = float(brentq(G, 0.0, fT))
    else:
        out['f_exit_pred'] = float(fT)   # mesopredator never fell below the exit threshold: recovers at f_T
    return out


def a_task(args):
    gens, imm = args
    return ramp(P_of(), gens, imm, 3.2)


def check_A():
    P = P_of(); fT = f_transcritical(P); fF = fold_by_discriminant(P, fmax=10)
    gens = np.geomspace(30, 10000, 5 if DRAFT else 14)
    jobs = [(float(g), imm) for imm in (1e-5, 1e-7, 1e-9) for g in gens]
    rows = [r for r in robust_map(a_task, jobs, 'A') if r]
    lam_p = (lambda ff, h=1e-6: ((1 / ((1 + ff + h) * (1 + P.s0 * (1 + P.q * (ff + h)))) -
                                  1 / ((1 + ff - h) * (1 + P.s0 * (1 + P.q * (ff - h))))) / (2 * h)))(fT)
    fits = {}
    for imm in (1e-5, 1e-7, 1e-9):
        rr = [r for r in rows if r['immigration'] == imm and r['f_recover'] is not None and fT - r['f_recover'] > 0.02]
        if len(rr) >= 3:
            sl = np.polyfit(np.log([r['gens'] for r in rr]), np.log([fT - r['f_recover'] for r in rr]), 1)[0]
            fits[f'{imm:.0e}'] = dict(slope_log_delay_vs_log_duration=float(sl), n=len(rr))
    err = [abs(r['f_exit_pred'] - r['f_exit_sim']) for r in rows if r['f_exit_pred'] is not None and r['f_exit_sim'] is not None]
    for r in rows:
        print(f"  A: imm {r['immigration']:.0e} ramp {r['gens']:7.0f} gen: collapse {r['f_collapse']}, recovery {r['f_recover']}, "
              f"exit sim {r['f_exit_sim']} vs entry-exit pred {r['f_exit_pred']}", flush=True)
    print(f'  A: static f_T={fT:.4f}, f_F={fF:.4f}, lambda\'(f_T)={lam_p:.4f}; delay ~ duration^slope: {fits}')
    if err:
        print(f'  A: entry-exit prediction error |f_pred - f_sim|: median {np.median(err):.4f}, max {np.max(err):.4f}')
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for imm, col in zip((1e-5, 1e-7, 1e-9), ('C0', 'C1', 'C2')):
        rr = [r for r in rows if r['immigration'] == imm]
        g = [r['gens'] for r in rr]
        ax[0].semilogx(g, [r['f_collapse'] or np.nan for r in rr], 'o-', c=col, label=f'imm {imm:.0e}')
        ax[1].semilogx(g, [r['f_recover'] or np.nan for r in rr], 'o-', c=col)
        ax[1].semilogx(g, [r['f_exit_pred'] or np.nan for r in rr], 'x--', c=col)
        ax[2].loglog(g, [(r['f_collapse'] - r['f_recover']) / (fF - fT) if r['f_recover'] else np.nan for r in rr], 'o-', c=col)
    ax[0].axhline(fF, c='k', ls='--'); ax[1].axhline(fT, c='k', ls='--'); ax[2].axhline(1, c='k', ls='--')
    ax[0].set(xlabel='ramp duration (generations)', ylabel='f at collapse'); ax[0].legend()
    ax[1].set(xlabel='ramp duration', ylabel='f at recovery (o sim, x entry-exit)')
    ax[2].set(xlabel='ramp duration', ylabel='loop width / static window')
    fig.tight_layout(); fig.savefig(QL / 'step15_A.png', dpi=130)
    save('A', dict(f_T=fT, f_F=fF, lambda_prime_fT=lam_p, fits=fits, rows=rows,
                   entry_exit_error=dict(median=float(np.median(err)) if err else None, max=float(np.max(err)) if err else None)))


# ------------------------------------------------------------------------------------------------- boost (B, C)
def removed_after_boost(P, fb, D, fboost, t_settle_gens=4000):
    """Start at E+ at background fear fb, boost to fboost for D generations, return to fb. True if the mesopredator
    is then excluded. Decided by events: y below 1e-3 of the saddle level (removed) or back above half of E+ (kept)."""
    q = P.with_(f=fb); eq = interior(q)
    if len(eq) != 2:
        return None
    S, Ep = eq[0], eq[1]
    gen = 1 / P.delta
    w = np.log(Ep)
    if D > 0:
        sol = solve_ivp(lambda t, ww: rhs_log(t, ww, P.with_(f=fboost)), (0, D * gen), w, method='LSODA',
                        rtol=1e-10, atol=1e-12)
        w = sol.y[:, -1]
    lo = np.log(1e-3 * S[1]); hi = np.log(0.5 * Ep[1])
    if w[1] <= lo:                       # already far below the saddle level (prey near K after the boost)
        return True
    if w[1] >= hi:
        return False
    e1 = lambda t, ww: ww[1] - lo; e1.terminal = True
    e2 = lambda t, ww: ww[1] - hi; e2.terminal = True; e2.direction = 1.0
    sol = solve_ivp(lambda t, ww: rhs_log(t, ww, q), (0, t_settle_gens * gen), w, method='LSODA', rtol=1e-10,
                    atol=1e-12, events=(e1, e2))
    if sol.t_events[0].size: return True
    if sol.t_events[1].size: return False
    return None


def critical_duration(P, fb, fboost, Dmax=2000.0):
    if not removed_after_boost(P, fb, Dmax, fboost):
        return np.inf
    lo, hi = 0.0, Dmax
    if removed_after_boost(P, fb, 0.0, fboost):
        return 0.0
    for _ in range(14 if DRAFT else 30):
        mid = 0.5 * (lo + hi)
        r = removed_after_boost(P, fb, mid, fboost)
        if r: hi = mid
        else: lo = mid
        if hi - lo < 1e-4 * hi: break
    return hi


def b_task(args):
    xs, kept, dl = args
    P = P_of(x_star=xs, kept=kept, delta=dl)
    fT = f_transcritical(P); cs = c_star(P); fF = fold_by_discriminant(P, fmax=30)
    row = dict(x_star=xs, kept=kept, delta=dl, f_T=fT, c_star=cs, f_F=fF)
    if fF is None:
        return row
    fmax = max(3.2, 1.3 * fF)
    r = ramp(P, 1000, 1e-7, fmax)
    row.update(ramp_fmax=fmax, f_collapse=r and r['f_collapse'], f_recover=r and r['f_recover'])
    fb = 0.5 * (fT + fF)
    eq = interior(P.with_(f=fb))
    row['Eplus_stable_mid'] = bool(len(eq) == 2 and classify(eq[1], P.with_(f=fb))['stable'])
    row['D_c_mid_window'] = critical_duration(P, fb, fmax)
    # canard regime at s0 = 0.02: is there a Hopf point of the upper branch inside the window?
    Q = P_of(s0=0.02, x_star=xs, kept=kept, delta=dl)
    qT = f_transcritical(Q); qF = fold_by_discriminant(Q, fmax=30)
    hp = hopf_points(Q, fmax=(qF or 5) * 1.05, n=300 if DRAFT else 1500)
    row['canard_s0_0.02'] = dict(f_T=qT, f_F=qF, hopf=[h['f'] for h in hp],
                                 hopf_in_window=bool(qF and any(qT < h['f'] < qF for h in hp)))
    return row


def check_B():
    vals = dict(x_star=(0.05, 0.1, 0.2), kept=(0.2, 0.34, 0.5), delta=(0.01, 0.03, 0.1))
    if DRAFT:
        jobs = [(0.1, 0.34, 0.03), (0.05, 0.34, 0.03), (0.1, 0.2, 0.1)]
    else:
        jobs = [(a, b, c) for a in vals['x_star'] for b in vals['kept'] for c in vals['delta']]
    rows = [r for r in robust_map(b_task, jobs, 'B') if r]
    for r in rows:
        print(f"  B: x*={r['x_star']}, kept={r['kept']}, delta={r['delta']}: f_T={r['f_T']:.3f} c*={r['c_star']:.3f} "
              f"f_F={r.get('f_F')} collapse={r.get('f_collapse')} recover={r.get('f_recover')} "
              f"E+stable={r.get('Eplus_stable_mid')} D_c={r.get('D_c_mid_window')} "
              f"canard regime={r.get('canard_s0_0.02', {}).get('hopf_in_window')}", flush=True)
    keep = dict(
        all_backward=all(5.0 > r['c_star'] for r in rows),
        loop_wider_than_window=sum(1 for r in rows if r.get('f_recover') is not None and r.get('f_collapse') is not None
                                   and r['f_collapse'] - r['f_recover'] > r['f_F'] - r['f_T']),
        recover_below_fT=sum(1 for r in rows if r.get('f_recover') is not None and r['f_recover'] < r['f_T']),
        lock_in_finite=sum(1 for r in rows if np.isfinite(r.get('D_c_mid_window', np.inf))),
        canard_regime=sum(1 for r in rows if r.get('canard_s0_0.02', {}).get('hopf_in_window')), n=len(rows))
    print('  B summary:', keep)
    save('B', dict(summary=keep, rows=rows))


def c_task(fb):
    P = P_of(); fT = f_transcritical(P); fF = fold_by_discriminant(P, fmax=10)
    return dict(f_b=fb, d=fb - fT, D_c=critical_duration(P, fb, 3.2, Dmax=5000.0))


def check_C():
    P = P_of(); fT = f_transcritical(P); fF = fold_by_discriminant(P, fmax=10)
    ds = np.geomspace(1e-3, 0.95 * (fF - fT), 4 if DRAFT else 16)
    rows = [r for r in robust_map(c_task, [float(fT + d) for d in ds], 'C') if r]
    fin = [r for r in rows if np.isfinite(r['D_c']) and r['D_c'] > 0]
    fit = {}
    near = [r for r in fin if r['d'] < 0.1 * (fF - fT)]
    if len(near) >= 3:
        fit['power_slope_logDc_vs_logd'] = float(np.polyfit(np.log([r['d'] for r in near]), np.log([r['D_c'] for r in near]), 1)[0])
        fit['log_slope_Dc_vs_ln1_over_d'] = float(np.polyfit(np.log(1 / np.array([r['d'] for r in near])), [r['D_c'] for r in near], 1)[0])
    for r in rows:
        print(f"  C: f_b - f_T = {r['d']:.2e}: D_c = {r['D_c']:.3f} generations", flush=True)
    print('  C: near f_T,', fit)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.loglog([r['d'] for r in fin], [r['D_c'] for r in fin], 'o-')
    ax.set(xlabel='f_b - f_T', ylabel='critical boost duration D_c (generations)')
    fig.tight_layout(); fig.savefig(QL / 'step15_C.png', dpi=130)
    save('C', dict(f_T=fT, f_F=fF, fit=fit, rows=rows))


# ------------------------------------------------------------------------------------------------- D Hopf
def d_task(args):
    s0, c, dl, xs = args
    try:
        P = P_of(s0=s0, c=c, delta=dl, x_star=xs)
        hp = hopf_points(P, fmax=8.0, n=250 if DRAFT else 800)
        return [dict(s0=s0, c=c, delta=dl, x_star=xs, f=h['f'], l1=h['l1'], omega=h['omega'],
                     l1_over_omega=h['l1'] / h['omega']) for h in hp]
    except Exception as e:
        return [dict(s0=s0, c=c, delta=dl, x_star=xs, error=repr(e))]


def check_D():
    s0s = np.geomspace(0.01, 3.0, 5 if DRAFT else 30)
    cs = (0.0, 5.0) if DRAFT else (0.0, 0.5, 2.0, 5.0, 20.0, 100.0)
    dls = (0.05, 1.0) if DRAFT else (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    xss = (0.1,) if DRAFT else (0.02, 0.1, 0.3)
    jobs = [(float(a), b, c, d) for a in s0s for b in cs for c in dls for d in xss]
    res = [r for rr in robust_map(d_task, jobs, 'D') if rr for r in rr]
    errs = [r for r in res if 'error' in r]; pts = [r for r in res if 'error' not in r]
    l1 = np.array([r['l1'] for r in pts])
    big = [r for r in pts if r['s0'] > 0.95]
    summ = dict(n_parameter_sets=len(jobs), n_hopf=len(pts), n_errors=len(errs),
                n_positive=int(np.sum(l1 > 0)) if len(l1) else 0,
                max_l1=float(l1.max()) if len(l1) else None,
                n_l1_above_minus_1em2=int(np.sum(l1 > -1e-2)) if len(l1) else 0,
                n_l1_above_minus_1em3=int(np.sum(l1 > -1e-3)) if len(l1) else 0,
                n_hopf_s0_above_095=len(big), max_l1_s0_above_095=max((r['l1'] for r in big), default=None),
                s0_max_with_hopf=max((r['s0'] for r in pts), default=None))
    print('  D:', summ)
    if summ['n_positive']:
        print('  D: POSITIVE l1 at', [r for r in pts if r['l1'] > 0][:10])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    if pts:
        sc = ax.scatter([r['s0'] for r in pts], [-r['l1'] for r in pts], c=np.log10([r['c'] + 0.1 for r in pts]), s=6)
        ax.set(xscale='log', yscale='log', xlabel='s0', ylabel='-l1  (all > 0 means supercritical)')
        plt.colorbar(sc, label='log10(c+0.1)')
    fig.tight_layout(); fig.savefig(QL / 'step15_D.png', dpi=130)
    save('D', dict(summary=summ, errors=errs[:20], points=pts))


# ------------------------------------------------------------------------------------------------- E delta_BT
def e_task(args):
    s0, c, xs, kept = args
    P = P_of(s0=s0, c=c, x_star=xs, kept=kept, delta=1.0)
    cs = c_star(P)
    if cs is None or c <= cs:
        return None
    fF = fold_by_discriminant(P, fmax=40)
    if fF is None:
        return None
    eq = interior(P.with_(f=fF - 1e-10))
    if not eq:
        return None
    e = tuple(np.mean(eq, axis=0)) if len(eq) == 2 else eq[0]
    J = jacobian(*e, P.with_(f=fF)); J11, j22 = J[0, 0], J[1, 1]
    if not (J11 < 0 < j22):
        return dict(s0=s0, c=c, x_star=xs, kept=kept, f_F=fF, delta_BT=None, J11=float(J11))
    return dict(s0=s0, c=c, x_star=xs, kept=kept, f_F=fF, delta_BT=float(-J11 / j22), x=e[0], y=e[1])


def check_E():
    s0s = np.geomspace(0.003, 10, 6 if DRAFT else 40); cs = np.geomspace(0.3, 300, 5 if DRAFT else 25)
    xss = (0.1,) if DRAFT else (0.02, 0.1, 0.3); kepts = (0.34,) if DRAFT else (0.2, 0.34, 0.6)
    jobs = [(float(a), float(b), x, k) for a in s0s for b in cs for x in xss for k in kepts]
    rows = [r for r in robust_map(e_task, jobs, 'E') if r]
    bt = [r for r in rows if r['delta_BT'] is not None]
    d = np.array([r['delta_BT'] for r in bt])
    summ = dict(n_backward=len(rows), n_bt=len(bt), n_no_bt_J11_nonneg=len(rows) - len(bt),
                min_delta_BT=float(d.min()) if len(d) else None,
                argmin=min(bt, key=lambda r: r['delta_BT']) if bt else None,
                quantiles=dict(zip(('1%', '10%', '50%'), np.percentile(d, [1, 10, 50]).tolist())) if len(d) else None,
                n_below_1=int(np.sum(d < 1)), n_below_4=int(np.sum(d < 4)))
    print('  E:', {k: v for k, v in summ.items() if k != 'argmin'}); print('  E: smallest at', summ['argmin'])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    if bt:
        sc = ax.scatter([r['s0'] for r in bt], d, c=np.log10([r['c'] for r in bt]), s=5)
        ax.set(xscale='log', yscale='log', xlabel='s0', ylabel='delta_BT'); ax.axhline(4, c='k', ls='--')
        plt.colorbar(sc, label='log10 c')
    fig.tight_layout(); fig.savefig(QL / 'step15_E.png', dpi=130)
    save('E', dict(summary=summ, rows=rows))


# ------------------------------------------------------------------------------------------------- F canard threshold
def splitting_tol(p, eps, a0_frac, rtol):
    """Same as qtds_fear.homoclinic.splitting, with the integration tolerance as a parameter."""
    out = saddle_and_focus(p)
    if out is None:
        return np.nan
    S, E, lu, vu, ls, vs = out
    toE = E - S
    if np.dot(vu, toE) < 0: vu = -vu
    if np.dot(vs, toE) < 0: vs = -vs
    M = np.column_stack([vs, vu]); Minv = np.linalg.inv(M)
    a0 = a0_frac * np.linalg.norm(toE)
    t_max = 60.0 / max(abs(lu), 1e-4) + 6000.0 / p.delta
    def ev(t, w):
        a, b = Minv @ (np.exp(w) - S)
        return a - a0
    ev.direction = -1.0; ev.terminal = True
    sol = solve_ivp(lambda t, w: rhs_log(t, w, p), (0, t_max), np.log(S + eps * vu), method='LSODA',
                    rtol=rtol, atol=rtol * 1e-2, events=ev)
    if sol.t_events[0].size == 0:
        xe, ye = np.exp(sol.y[:, -1])
        return -1.0 if (xe > 0.9 and ye < 0.5 * S[1]) or ye < 1e-3 * S[1] else 1.0
    a, b = Minv @ (np.exp(sol.y_events[0][0]) - S)
    return float(b) / a0


def f_task(args):
    rtol, eps, a0 = args
    P = Params(s0=0.02, c=5.0, delta=0.05)
    lo, hi = 0.29530, 0.29545
    slo, shi = np.sign(splitting_tol(P.with_(f=lo), eps, a0, rtol)), np.sign(splitting_tol(P.with_(f=hi), eps, a0, rtol))
    if slo == shi or slo == 0:
        return dict(rtol=rtol, eps=eps, a0_frac=a0, f_c=None, note=f'no sign change on [{lo},{hi}]: {slo},{shi}')
    while hi - lo > (1e-8 if DRAFT else 1e-11):
        mid = 0.5 * (lo + hi)
        if np.sign(splitting_tol(P.with_(f=mid), eps, a0, rtol)) == slo: lo = mid
        else: hi = mid
    return dict(rtol=rtol, eps=eps, a0_frac=a0, f_c=0.5 * (lo + hi))


def check_F():
    if DRAFT:
        jobs = [(1e-10, 1e-7, 0.05)]
    else:
        jobs = [(r, e, a) for r in (1e-9, 1e-10, 1e-11, 1e-12) for e in (1e-6, 1e-7, 1e-8) for a in (0.02, 0.05, 0.1)]
    rows = [r for r in robust_map(f_task, jobs, 'F') if r]
    fc = np.array([r['f_c'] for r in rows if r['f_c'] is not None])
    for r in rows:
        print(f"  F: rtol {r['rtol']:.0e} eps {r['eps']:.0e} a0 {r['a0_frac']}: f_c = {r['f_c']}", flush=True)
    summ = dict(n=len(rows), n_ok=len(fc), f_c_min=float(fc.min()) if len(fc) else None,
                f_c_max=float(fc.max()) if len(fc) else None, f_c_median=float(np.median(fc)) if len(fc) else None,
                spread=float(fc.max() - fc.min()) if len(fc) else None)
    best = [r['f_c'] for r in rows if r['f_c'] is not None and r['rtol'] <= 1e-11]
    if best:
        summ.update(f_c_min_tight=float(min(best)), f_c_max_tight=float(max(best)), spread_tight=float(max(best) - min(best)))
    print('  F:', summ)
    save('F', dict(summary=summ, rows=rows))


CHECKS = dict(A=check_A, B=check_B, C=check_C, D=check_D, E=check_E, F=check_F)
if __name__ == '__main__':
    sel = [a.upper() for a in sys.argv[1:]] or list(CHECKS)
    for k in sel:
        t0 = time.time(); print(f'--- step15 {k} ---', flush=True)
        try:
            CHECKS[k]()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'--- step15 {k} FAILED ({e!r}); continuing with the next part ---', flush=True)
        print(f'--- step15 {k} done in {time.time() - t0:.0f} s ---', flush=True)
