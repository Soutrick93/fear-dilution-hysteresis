# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig3_theorem_map.py
# Also run automatically by:  python run_all.py   (job name: fig3)
# Output:  figures/pdf/Fig3_hysteresis_theorem.pdf and figures/png/
# Used in the paper for:  Figure 3
# ----------------------------------------------------------------------------------------------------
"""Figure 3. The headline theorem: fear-driven cascade reversal is hysteretic exactly when c > c*.
a  regime map in the (c, f) plane (s0 = 1): number of attractors, f_T, fold curve, the codimension-2 point (c*, f_T)
b  c < c*: forward transcritical, reversible        c  c > c*: backward transcritical, hysteresis loop
d  closed-form c*(s0), validated by direct root counting
e  window width grows like (c - c*)^2 near the codimension-2 point
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from qtds_fear import style as S
from qtds_fear.model import Params
from qtds_fear.equilibria import interior, raccoon_free_stable, classify
from qtds_fear.bifurcation import opening_coefficient, f_transcritical, c_star, fold_points


def regime(p):
    n = len(interior(p)); e1 = raccoon_free_stable(p)
    if not e1:
        return 1               # raccoons persist (single coexistence state)
    return 2 if n == 2 else 0  # bistable or raccoon-free


def panel_a(ax, s0=1.0):
    p = Params(s0=s0)
    cs, fT = c_star(p), f_transcritical(p)
    C = np.geomspace(0.05, 60, 240); F = np.linspace(0, 4.0, 240)
    Z = np.array([[regime(p.with_(c=float(cv), f=float(fv))) for cv in C] for fv in F])
    cmap = ListedColormap([S.TINT_FREE, S.TINT_COEX, S.TINT_BISTABLE]); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    ax.pcolormesh(C, F, Z, cmap=cmap, norm=norm, shading='auto', rasterized=True, linewidth=0, antialiased=True)  # raster: no seam lines
    cc = np.geomspace(cs * 1.01, 60, 60)
    ff = [fold_points(p.with_(c=float(v)), fmax=8, n=1200)[0] for v in cc]
    ax.plot(np.r_[cs, cc], np.r_[fT, ff], color=S.BLUE, lw=1.5)
    ax.plot([0.05, 60], [fT, fT], color=S.MESO, lw=1.3)
    ax.plot([cs], [fT], 'o', ms=5, mfc='white', mec=S.INK, mew=1.2, zorder=6)
    ax.axhline(1.0, color=S.FEAR, lw=0.9, ls=(0, (4, 2)))
    ax.set_xscale('log'); ax.set_xlim(0.05, 60); ax.set_ylim(0, 4.0)
    ax.set_xlabel('group dilution of the fear cost  c'); ax.set_ylabel('fear cue  f')
    S.note(ax, 0.07, 0.45, 'raccoons persist\n(one stable state)', fontsize=8.2, color=S.INK)
    S.note(ax, 0.07, 3.1, 'raccoon-free\ncrabs recover fully', fontsize=8.2, color=S.INK)
    S.note(ax, 6, 2.05, 'bistable:\nhysteresis', fontsize=8.2, color=S.INK, ha='center')
    S.note(ax, 0.055, fT + 0.06, r'$f_T$: raccoon-free state turns stable', fontsize=7.5, color=S.MESO)
    S.note(ax, 20, 3.55, 'fold $f_F(c)$', fontsize=7.5, color=S.BLUE, ha='center')
    S.anno(ax, f'c* = {cs:.3f}\n(degenerate\ntranscritical)', xy=(cs, fT), xytext=(0.6, 0.35), fontsize=7.6,
                arrowprops=dict(arrowstyle='-', lw=0.6, color=S.INK2))
    S.note(ax, 0.055, 1.06, 'Suraci playback level', fontsize=7.5, color=S.FEAR)
    S.title(ax, f'Regimes of the fear cascade (s$_0$ = {s0:g})')
    return cs, fT


def branches(ax, c, s0=1.0, title='', show_loop=False):
    p = Params(s0=s0, c=c); fT = f_transcritical(p)
    fs = np.linspace(0, 3.2, 700)
    st, un = [], []
    for f in fs:
        q = p.with_(f=float(f))
        for e in interior(q):
            (st if classify(e, q)['stable'] else un).append((f, e[0]))
    st, un = np.array(st), np.array(un)
    ax.plot(st[:, 0], st[:, 1], '.', ms=1.2, color=S.PREY)
    if len(un):
        ax.plot(un[:, 0], un[:, 1], '.', ms=0.8, color=S.INK2)
    ax.plot([0, fT], [1, 1], color=S.INK2, lw=1.0, ls=(0, (3, 2)))
    ax.plot([fT, 3.2], [1, 1], color=S.PREY, lw=2.0)
    ax.axvline(1.0, color=S.FEAR, lw=0.7, ls=(0, (4, 2)))
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 1.08)
    ax.set_xlabel('fear cue  f'); ax.set_ylabel('crabs at equilibrium  x*')
    S.title(ax, title)
    if show_loop:
        fF = fold_points(p, fmax=6)[0]
        up = [e for e in st if e[0] < fF]
        xF = st[np.argmin(np.abs(st[:, 0] - fF)), 1]
        S.anno(ax, '', xy=(fF, 0.98), xytext=(fF, xF + 0.02), arrowprops=dict(arrowstyle='-|>', color=S.RED, lw=1.1))
        S.anno(ax, '', xy=(fT, 0.5), xytext=(fT, 0.97), arrowprops=dict(arrowstyle='-|>', color=S.RED, lw=1.1))
        S.note(ax, fF + 0.05, 0.62, 'collapse of\nraccoons', fontsize=7.4, color=S.RED)
        S.note(ax, fT - 0.06, 0.58, 'recovery\nonly here', fontsize=7.4, color=S.RED, ha='right')
        ax.axvspan(fT, fF, color=S.TINT_BISTABLE, zorder=0)
    S.note(ax, 3.15, 0.04, 'solid: stable   dots: saddle', fontsize=7.2, color=S.INK2, ha='right')


def panel_d(ax):
    s0s = np.geomspace(0.05, 5, 200)
    cs = np.array([c_star(Params(s0=s)) for s0 in [None] for s in s0s])
    ax.fill_between(s0s, cs, 10, color=S.TINT_BISTABLE, lw=0)
    ax.plot(s0s, cs, color=S.INK, lw=1.4)
    chk = []
    for s in (0.1, 0.3, 1.0, 3.0):
        p = Params(s0=s); c0 = c_star(p); fT = f_transcritical(p)
        for fac, mk in ((0.9, 'x'), (1.1, 'o')):
            n = len(interior(p.with_(c=c0 * fac, f=fT + 1e-7)))
            ax.plot(s, c0 * fac, mk, ms=3.5, color=S.BLUE if n == 2 else S.MESO, mfc='none' if mk == 'o' else None, mew=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.05, 5); ax.set_ylim(0.05, 10)
    ax.set_xlabel('half-saturation  s$_0$'); ax.set_ylabel('c*')
    S.note(ax, 0.07, 3.0, 'hysteresis possible', fontsize=7.8)
    S.note(ax, 0.5, 0.075, 'always reversible', fontsize=7.8)
    S.note(ax, 0.06, 0.12, r'$c^*=\dfrac{s_T(1+f_T)}{f_T(1+s_T)^2}$', fontsize=8.4)
    S.title(ax, 'Closed-form threshold')
    S.note(ax, 4.8, 4.5, 'symbols: root count at\n0.9c* (x) and 1.1c* (o)', fontsize=7, color=S.INK2, ha='right', va='top')


def panel_e(ax):
    for s0, col in ((0.5, S.MESO), (1.0, S.BLUE), (2.0, S.FEAR)):
        p = Params(s0=s0); cs = c_star(p); fT = f_transcritical(p)
        d = np.geomspace(3e-3, 3, 16) * cs
        w = [fold_points(p.with_(c=float(cs + dd)), fmax=10, n=1500)[0] - fT for dd in d]
        ax.loglog(d, w, 'o', ms=2.8, color=col, label=f's$_0$ = {s0:g}')
        C = opening_coefficient(p)            # exact leading term, Prop. 'Opening of the window'
        # the leading term is an asymptote as c -> c*: draw it only where it applies (c - c* <= 0.05)
        xx = np.geomspace(d[0], 0.05, 30); ax.loglog(xx, C * xx ** 2, color=col, lw=1.0, ls=(0, (3, 2)))
    S.note(ax, 4e-3, 4e-6, 'dashed: C (c-c*)$^2$', fontsize=7.6, color=S.INK2)
    ax.set_xlabel('c - c*'); ax.set_ylabel('hysteresis window  $f_F - f_T$')
    ax.legend(loc='lower right'); S.title(ax, 'Window opens quadratically')


def main():
    S.apply()
    fig = plt.figure(figsize=(S.DOUBLE, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], hspace=0.62, wspace=0.48)
    a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, 1])
    d = fig.add_subplot(gs[0, 2]); e = fig.add_subplot(gs[1, 2])
    cs, fT = panel_a(a)
    branches(b, 0.15, title=f'c = 0.15 < c*: reversible')
    branches(c, 5.0, title=f'c = 5 > c*: hysteresis', show_loop=True)
    panel_d(d); panel_e(e)
    for ax, L in ((a, 'a'), (b, 'b'), (c, 'c'), (d, 'd'), (e, 'e')):
        S.panel_label(ax, L, x=-0.2 if ax is not a else -0.14, y=1.03 if ax is not a else 1.01)
    return S.save(fig, 'Fig3_hysteresis_theorem')


if __name__ == '__main__':
    print(main())
