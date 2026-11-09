# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig9_cycles_canard.py
# Also run automatically by:  python run_all.py   (job name: fig9)
# Output:  figures/pdf/Fig9_cycles_canard.pdf and figures/png/
# Used in the paper for:  Figure 9
# ----------------------------------------------------------------------------------------------------
"""Figure 9. Boom-bust cycles and canard-induced collapse (s0 = 0.02, c = 5, delta = 0.05).
a  phase plane: the family of stable cycles from the Hopf point f_H down to the canard threshold f_c, coloured by
   fear, on top of the critical manifold y = (1-x)(s+x) (attracting part solid, repelling part dashed).
b  canard explosion: cycle amplitude in crabs vs fear, on a log scale of the distance to f_c (inset: linear scale).
c  time series just above f_c (canard cycle persists) and just below (crab crash, raccoons starve to extinction).
Uses f_c from results/step8e_homoclinic.json if present, else recomputes it.
Runtime ~5-15 min (draft ~1-2 min)."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from qtds_fear import style as S
from qtds_fear import figtools as T
from qtds_fear.model import Params, rhs_log
from qtds_fear.equilibria import interior
from qtds_fear.bifurcation import hopf_points, f_transcritical
from qtds_fear.homoclinic import saddle_and_focus, find_homoclinic

P = Params(s0=0.02, c=5.0, delta=0.05)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def f_canard():
    fpath = ROOT / 'results' / 'step8e_homoclinic.json'
    if fpath.exists():
        for r in json.loads(fpath.read_text()):
            if r.get('s0') == P.s0 and r.get('delta') == P.delta and r.get('f_hom'):
                return max(r['f_hom'])
    fT = f_transcritical(P); fH = hopf_points(P, fmax=3, n=2000)[0]['f']
    return max(find_homoclinic(P, fT + 1e-4, fH - 1e-6)[0])


def cycle(q, t_settle, t_rec):
    S_, E, lu, vu, ls, vs = saddle_and_focus(q)
    if np.dot(vu, E - S_) < 0: vu = -vu
    z0 = np.log(S_ + 1e-7 * vu)
    s1 = solve_ivp(lambda t, w: rhs_log(t, w, q), (0, t_settle), z0, method='LSODA', rtol=1e-11, atol=1e-12)
    s2 = solve_ivp(lambda t, w: rhs_log(t, w, q), (0, t_rec), s1.y[:, -1], method='LSODA', rtol=1e-11, atol=1e-12,
                   t_eval=np.linspace(0, t_rec, 6000))
    return np.exp(s2.y)


def main():
    S.apply()
    fH = hopf_points(P, fmax=3, n=2000)[0]['f']; fc = f_canard()
    gen = 1 / P.delta
    fig = plt.figure(figsize=(S.DOUBLE, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.25], wspace=0.38)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); c = fig.add_subplot(gs[2])
    # a: cycles on the critical manifold
    q0 = P.with_(f=fc); s = q0.s; xh = (1 - s) / 2
    xx = np.linspace(0, 1, 500); yy = (1 - xx) * (s + xx)
    a.plot(xx[xx >= xh], yy[xx >= xh], color=S.PREY, lw=2.0)
    a.plot(xx[xx < xh], yy[xx < xh], color=S.PREY, lw=1.6, ls=(0, (3, 2)))
    exps = np.linspace(-0.5, -5, T.res(12, 5))      # f_c is known to ~1e-9: stay above that
    dfs = 10.0 ** exps * (fH - fc)
    cm = T.cmap('fear'); amps = []
    for k, df in enumerate(dfs):
        Z = cycle(P.with_(f=fc + df), T.res(30000, 8000) * 1.0, 3 * gen * 10)
        amps.append(Z[0].max() - Z[0].min())
        a.plot(Z[0], Z[1], color=cm(0.3 + 0.7 * k / max(len(dfs) - 1, 1)), lw=1.0)
    e = interior(q0)
    a.plot(*e[0], 'o', ms=5, mfc='white', mec=S.INK, mew=1.1); a.plot(*e[1], 'o', ms=4, mfc='white', mec=S.FEAR, mew=1.1)
    a.set_xlim(0, 1); a.set_ylim(0, 0.3); a.set_xlabel('crabs  x'); a.set_ylabel('raccoons  y')
    S.panel_label(a, 'a', x=-0.22)
    # b: explosion
    b.semilogx(dfs, amps, 'o-', ms=3.5, color=S.FEAR, lw=1.2)
    b.set_xlabel('f - f$_c$'); b.set_ylabel('crab amplitude')
    ins = b.inset_axes([0.12, 0.1, 0.4, 0.36])
    ins.plot(fc + dfs, amps, 'o-', ms=2, color=S.FEAR, lw=0.9); ins.axvline(fc, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ins.axvline(fH, color=S.INK2, lw=0.6); ins.tick_params(labelsize=7)
    ins.set_xticks([fc, fH]); ins.set_xticklabels(['$f_c$', '$f_H$']); ins.set_yticks([0.1, 0.2])
    S.panel_label(b, 'b', x=-0.25)
    # c: time series either side
    for sign, col in ((+1, S.BLUE), (-1, S.RED)):
        q = P.with_(f=fc + sign * 1e-3 * (fH - fc))
        E = np.array(interior(q)[-1])                 # unstable focus: orbits spiral out from it
        tend = T.res(160, 100) * gen
        sol = solve_ivp(lambda t, w: rhs_log(t, w, q), (0, tend), np.log(E * np.array([1.02, 1.0])), method='LSODA',
                        rtol=1e-10, atol=1e-12, t_eval=np.linspace(0, tend, 8000))
        Z = np.exp(sol.y)
        c.plot(sol.t / gen, Z[1], color=col, lw=1.3)
        c.plot(sol.t / gen, Z[0] * 0.3, color=col, lw=0.8, alpha=0.5)
    c.set_xlabel('time (raccoon generations)'); c.set_ylabel('raccoons  y  (thin: crabs x 0.3)')
    S.panel_label(c, 'c', x=-0.2)
    return S.save(fig, 'Fig9_cycles_canard')


if __name__ == '__main__':
    print(main())
