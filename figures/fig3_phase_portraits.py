# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig3_phase_portraits.py
# Also run automatically by:  python run_all.py   (job name: fig3)
# Output:  figures/pdf/Fig3_phase_portraits*.pdf and figures/png/
# Used in the paper for:  Figure 3
# ----------------------------------------------------------------------------------------------------
"""Figure 3. Phase portraits across the fear gradient (s0 = 1, c = 5).
Three fear levels: below f_T (raccoons persist), inside the hysteresis window (bistable), above the fold (raccoon-free).
Each panel: basins of attraction (tint), flow lines shaded by speed, both nullclines, the saddle's stable manifold
(the tipping boundary), equilibria (filled = stable, open = saddle) and example trajectories.
A fourth panel stacks the three basin boundaries to show the raccoon basin shrinking as fear rises.
Runtime: ~3-8 min (draft: QTDS_DRAFT=1, ~30 s)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from qtds_fear import style as S
from qtds_fear import figtools as T
from qtds_fear.model import Params
from qtds_fear.equilibria import interior, classify

P0 = Params(s0=1.0, c=5.0, delta=0.3)
FEARS = (1.0, 2.0, 2.8)
XMAX, YMAX = 1.0, 3.0


def portrait(ax, p, n_basin, show_traj=True):
    xs = np.linspace(0.004, XMAX, n_basin); ys = np.linspace(0.004, YMAX, n_basin)
    eq = interior(p); e1_stable = p.mu * (1 + p.f) * (1 + p.s) > 1
    if not eq:                                       # raccoon-free everywhere
        persist = np.zeros((len(ys), len(xs)), bool)
    elif not e1_stable:                              # raccoons persist from every positive start
        persist = np.ones((len(ys), len(xs)), bool)
    else:                                            # bistable: integrate the grid
        persist, _, _ = T.basin_grid(p, xs, ys, t_end=T.res(1500.0, 600.0), dt=0.05)
    ax.pcolormesh(xs, ys, persist.astype(float), cmap=ListedColormap([S.TINT_FREE, S.TINT_COEX]), vmin=0, vmax=1,
                  shading='auto', rasterized=True, zorder=0)
    xg = np.linspace(0.002, XMAX, T.res(70, 35)); yg = np.linspace(0.002, YMAX, T.res(70, 35))
    X, Y, U, V = T.field(p, xg, yg)
    sp = np.hypot(U, V / 1.0) + 1e-9
    ax.streamplot(X, Y, U, V, color=np.log10(sp), cmap=T.cmap('speed'), density=T.res(1.6, 0.9), linewidth=0.45,
                  arrowsize=0.55, zorder=1)
    xx = np.linspace(0.0, 1.0, 400)
    ax.plot(xx, (1 - xx) * (p.s + xx), color=S.PREY, lw=1.8, zorder=3, solid_capstyle='round')     # prey nullcline
    Xn, Yn, Z = T.predator_nullcline_y(p, np.linspace(0.002, XMAX, 400), np.linspace(0.002, YMAX, 400))
    ax.contour(Xn, Yn, Z, levels=[0], colors=[S.MESO], linewidths=1.8, zorder=3)                  # raccoon nullcline
    ax.plot([0, XMAX], [0, 0], color=S.MESO, lw=1.8, zorder=3, clip_on=False)
    Ws, Wu = T.saddle_manifolds(p)
    for br in Ws:
        ax.plot(br[0], br[1], color=S.INK, lw=1.3, ls=(0, (4, 2)), zorder=4)
    if show_traj:
        for z0 in ((0.9, 2.7), (0.15, 2.9), (0.6, 0.08), (0.95, 0.6)):
            t, Z0 = T.orbit(p, z0, 2500.0, n=3000)
            ax.plot(Z0[0], Z0[1], color=S.FEAR, lw=0.9, alpha=0.9, zorder=5)
            ax.plot(*z0, 'o', ms=2.5, color=S.FEAR, zorder=5)
    for e in [(1.0, 0.0)] + interior(p):
        k = classify(e, p) if e[1] > 0 else None
        stable = (e[1] == 0 and p.mu * (1 + p.f) * (1 + p.s) > 1) or (k is not None and k['stable'])
        ax.plot(*e, 'o', ms=6, mfc=S.INK if stable else 'white', mec=S.INK, mew=1.2, zorder=6, clip_on=False)
    ax.plot(0, 0, 'o', ms=6, mfc='white', mec=S.INK, mew=1.2, zorder=6, clip_on=False)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
    ax.set_xlabel('crabs  x'); ax.set_ylabel('raccoons  y')
    return persist, xs, ys


def main():
    S.apply()
    fig = plt.figure(figsize=(S.DOUBLE * 1.1, 2.6))
    gs = fig.add_gridspec(1, 4, wspace=0.45)
    n = T.res(260, 70)
    basins = []
    for i, f in enumerate(FEARS):
        ax = fig.add_subplot(gs[0, i])
        basins.append(portrait(ax, P0.with_(f=f), n))
        if i: ax.set_ylabel('')
        S.panel_label(ax, 'abc'[i], x=-0.2)
        if S.ANNOTATE: S.title(ax, f'f = {f:g}')
    ax = fig.add_subplot(gs[0, 3])
    # tipping boundaries (saddle stable manifolds) across the whole hysteresis window, coloured by fear
    from qtds_fear.bifurcation import f_transcritical, fold_points
    fT, fF = f_transcritical(P0), fold_points(P0)[0]
    fs = np.linspace(fT + 0.02, fF - 0.005, T.res(12, 5))
    cm = T.cmap('fear'); sad = []
    for k, f in enumerate(fs):
        q = P0.with_(f=float(f)); col = cm(0.25 + 0.75 * k / max(len(fs) - 1, 1))
        Ws, _ = T.saddle_manifolds(q)
        for br in Ws:
            ax.plot(br[0], br[1], color=col, lw=1.2)
        sd = interior(q)[0]; sad.append(sd)
        ax.plot(*sd, 'o', ms=3.5, mfc='white', mec=col, mew=1.0, zorder=5)
        up = interior(q)[1]; ax.plot(*up, 'o', ms=3.5, color=col, zorder=5)
    ax.plot(1, 0, 'o', ms=6, color=S.INK, clip_on=False, zorder=6)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX); ax.set_xlabel('crabs  x')
    S.panel_label(ax, 'd', x=-0.2)
    return S.save(fig, 'Fig3_phase_portraits')


if __name__ == '__main__':
    print(main())
