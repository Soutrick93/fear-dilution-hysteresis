# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig7_playback_experiment.py
# Also run automatically by:  python run_all.py   (job name: fig7)
# Output:  figures/pdf/Fig7_playback_experiment.pdf and figures/png/
# Used in the paper for:  Figure 7
# ----------------------------------------------------------------------------------------------------
"""Figure 7. In-silico playback experiments (s0 = 1, c = 5, raccoon-like time scale delta = 0.03).
a  fear schedule: a slow ramp up, a plateau, a slow ramp down (stacked above b)
b  crab and raccoon time series under that schedule: collapse on the way up, recovery only much later on the way down
c  the same run as a loop in the (fear, raccoons) plane over the equilibrium branches (stable solid, saddle dashed),
   coloured by time: the dynamic hysteresis loop
d  'can a temporary boost lock in the cascade reversal?': background fear f_b (carnivores at some density), plus a
   temporary boost to f = 3.2 lasting D raccoon generations (1/delta). Colour = final raccoon density after the boost
   ends. Inside the hysteresis window a long enough boost removes raccoons for good; outside it the effect fades.
   Star: the Suraci experiment (no carnivores, 28 days).
Runtime ~5-12 min (draft ~1 min)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from qtds_fear import style as S
from qtds_fear import figtools as T
from qtds_fear.model import Params, rhs_log
from qtds_fear.equilibria import interior, classify
from qtds_fear.bifurcation import f_transcritical, fold_points

P = Params(s0=1.0, c=5.0, delta=0.03)
FMAX = 3.2


def schedule(t, t_up, t_hold, t_down):
    if t < t_up:
        return FMAX * t / t_up
    if t < t_up + t_hold:
        return FMAX
    return max(FMAX * (1 - (t - t_up - t_hold) / t_down), 0.0)


def run_ramp():
    gen = 1 / P.delta
    t_up, t_hold, t_down = 300 * gen, 120 * gen, 300 * gen
    t_end = t_up + t_hold + t_down + 40 * gen
    z0 = interior(P.with_(f=0.0))[-1]
    f_t = lambda t: schedule(t, t_up, t_hold, t_down)
    sol = solve_ivp(lambda t, w: rhs_log(t, w, P.with_(f=f_t(t))) + np.array([0.0, 1e-7 / np.exp(w[1])]),
                    (0, t_end), np.log(z0), t_eval=np.linspace(0, t_end, T.res(6000, 1500)),
                    method='LSODA', rtol=1e-8, atol=1e-10, max_step=gen / 4)
    return sol.t / gen, np.exp(sol.y), np.array([f_t(t) for t in sol.t])


def branches():
    fs = np.linspace(0, FMAX, T.res(500, 150)); st, un = [], []
    for f in fs:
        q = P.with_(f=float(f))
        for e in interior(q):
            (st if classify(e, q)['stable'] else un).append((f, e[1]))
    return np.array(st), np.array(un)


F_PULSE = 3.2


def pulse_map():
    """Background fear f_b (carnivores present at some level) plus a temporary playback boost to F_PULSE lasting D
    generations. Start at the raccoon equilibrium for f_b; after the boost, fear returns to f_b; record final raccoons."""
    gen = 1 / P.delta
    fbs = np.linspace(0.0, 3.0, T.res(80, 16)); Ds = np.geomspace(0.02, 300, T.res(70, 14))
    out = np.zeros((len(Ds), len(fbs)))
    imm = lambda w: np.array([0.0, 1e-7 / np.exp(w[1])])
    for j, fb in enumerate(fbs):
        eq = interior(P.with_(f=float(fb)))
        if not eq:
            continue                                         # raccoon-free already
        z0 = np.log(eq[-1])
        for i, D in enumerate(Ds):
            s1 = solve_ivp(lambda t, w: rhs_log(t, w, P.with_(f=F_PULSE)) + imm(w), (0, D * gen), z0,
                           method='LSODA', rtol=1e-8, atol=1e-10)
            s2 = solve_ivp(lambda t, w: rhs_log(t, w, P.with_(f=float(fb))) + imm(w), (0, 400 * gen), s1.y[:, -1],
                           method='LSODA', rtol=1e-8, atol=1e-10)
            out[i, j] = np.exp(s2.y[1, -1])
    return fbs, Ds, out


def main():
    S.apply()
    fig = plt.figure(figsize=(S.DOUBLE * 1.1, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[0.33, 1], hspace=0.28, wspace=0.95)
    t, Z, f = run_ramp()
    fT, fF = f_transcritical(P), fold_points(P)[0]
    # a: schedule
    a = fig.add_subplot(gs[0, 0])
    a.plot(t, f, color=S.FEAR, lw=1.3)
    for lev in (fT, fF):
        a.axhline(lev, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    a.set_xlim(t[0], t[-1]); a.set_ylim(0, FMAX * 1.05); a.tick_params(labelbottom=False); a.set_ylabel('fear  f')
    S.panel_label(a, 'a', x=-0.13)
    # b: time series
    b = fig.add_subplot(gs[1, 0], sharex=a)
    b.plot(t, Z[0], color=S.PREY, lw=1.5)
    b2 = b.twinx(); b2.spines['right'].set_visible(True)
    b2.plot(t, Z[1], color=S.MESO, lw=1.5)
    i_up = np.argmax(f >= fF); i_dn = len(f) // 2 + np.argmax(f[len(f) // 2:] <= fT)
    for i in (i_up, i_dn):
        b.axvline(t[i], color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    b.set_ylim(0, 1.05); b2.set_ylim(0, 2.8); b.set_xlabel('time (raccoon generations, 1/δ)')
    b.set_ylabel('crabs  x'); b2.set_ylabel('raccoons  y')
    S.panel_label(b, 'b', x=-0.13, y=1.0)
    # c: dynamic hysteresis loop
    c = fig.add_subplot(gs[:, 1])
    st, un = branches()
    c.plot(st[:, 0], st[:, 1], '.', ms=1.4, color=S.INK)
    if len(un): c.plot(un[:, 0], un[:, 1], '.', ms=0.7, color=S.INK2)
    c.plot([0, fT], [0, 0], color=S.INK2, lw=1.0, ls=(0, (3, 2))); c.plot([fT, FMAX], [0, 0], color=S.INK, lw=1.6)
    c.axvspan(fT, fF, color=S.TINT_BISTABLE, zorder=0, lw=0)
    pts = np.column_stack([f, Z[1]]).reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=T.cmap('time'), linewidths=2.2)
    lc.set_array(t[:-1]); c.add_collection(lc)
    for k in (len(t) // 6, len(t) // 3, 2 * len(t) // 3, 5 * len(t) // 6):
        S.anno(c, '', xy=(f[k + 5], Z[1][k + 5]), xytext=(f[k], Z[1][k]),
                   arrowprops=dict(arrowstyle='-|>', color=S.INK, lw=0.8, mutation_scale=9))
    c.set_xlim(0, FMAX); c.set_ylim(-0.05, 2.7); c.set_xlabel('fear  f'); c.set_ylabel('raccoons  y')
    S.panel_label(c, 'c', x=-0.22, y=1.01)
    # d: pulse strength x duration map
    d = fig.add_subplot(gs[:, 2])
    fbs, Ds, out = pulse_map()
    im = d.pcolormesh(fbs, Ds, out, cmap=T.cmap('meso'), shading='auto', rasterized=True, vmin=0)
    d.contour(fbs, Ds, out, levels=[0.05], colors='white', linewidths=1.6)
    d.set_yscale('log')
    for lev in (fT, fF):
        d.axvline(lev, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    d.plot(0.0, 28 / 365 * 0.5, '*', ms=9, color=S.FEAR, mec='white', mew=0.6, clip_on=False, zorder=5)
    d.set_xlabel('background fear  f$_b$'); d.set_ylabel('boost duration  D (generations)')
    cb = fig.colorbar(im, ax=d, fraction=0.06, pad=0.03); cb.set_label('final raccoons  y'); cb.outline.set_visible(False)
    S.panel_label(d, 'd', x=-0.3, y=1.01)
    return S.save(fig, 'Fig7_playback_experiment')


if __name__ == '__main__':
    print(main())
