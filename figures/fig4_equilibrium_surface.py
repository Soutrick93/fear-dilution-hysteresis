# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig4_equilibrium_surface.py
# Also run automatically by:  python run_all.py   (job name: fig4)
# Output:  figures/pdf/Fig4_equilibrium_surface.pdf and figures/png/
# Used in the paper for:  Figure 4
# ----------------------------------------------------------------------------------------------------
"""Figure 4. The equilibrium surface over (fear f, dilution c) at s0 = 1: the geometry behind the c* theorem.
a  3-D surface of equilibrium raccoon density y*: stable sheet (shaded by height), saddle sheet (translucent grey),
   raccoon-free floor (y = 0, stable part tinted), the fold curve and the transcritical line, which meet at the
   degenerate transcritical point (c*, f_T) (black dot). For c > c* the surface folds back: hysteresis.
b  the same surface seen from above as a map of equilibrium crab density x* on the stable sheet, with the bistable
   region hatched.
Runtime ~1-3 min (draft ~15 s)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from qtds_fear import style as S
from qtds_fear import figtools as T
from qtds_fear.model import Params
from qtds_fear.equilibria import interior, classify, raccoon_free_stable
from qtds_fear.bifurcation import f_transcritical, c_star, fold_points

P = Params(s0=1.0)
FMAX = 3.6


def sheets(nf, nc):
    fs = np.linspace(0.0, FMAX, nf); lc = np.linspace(np.log10(0.05), np.log10(60), nc)
    Yst = np.full((nc, nf), np.nan); Ysd = np.full((nc, nf), np.nan); Xst = np.full((nc, nf), np.nan)
    bist = np.zeros((nc, nf), bool)
    for i, l in enumerate(lc):
        for j, f in enumerate(fs):
            q = P.with_(c=float(10 ** l), f=float(f))
            for e in interior(q):
                if classify(e, q)['stable']:
                    Yst[i, j] = e[1]; Xst[i, j] = e[0]
                else:
                    Ysd[i, j] = e[1]
            bist[i, j] = raccoon_free_stable(q) and np.isfinite(Yst[i, j])
    return fs, lc, Yst, Ysd, Xst, bist


def main():
    S.apply()
    fs, lc, Yst, Ysd, Xst, bist = sheets(T.res(220, 60), T.res(160, 45))
    F, L = np.meshgrid(fs, lc)
    cs, fT = c_star(P), f_transcritical(P)
    cc = np.geomspace(cs * 1.01, 60, T.res(60, 15)); ff = [fold_points(P.with_(c=float(c)), fmax=8)[0] for c in cc]
    fig = plt.figure(figsize=(S.DOUBLE, 3.4))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12)
    a = fig.add_subplot(gsp[0], projection='3d', computed_zorder=False)
    cm = T.cmap('meso')
    col = cm(np.clip(np.nan_to_num(Yst) / np.nanmax(Yst), 0, 1)); col[..., 3] = 0.95
    a.plot_surface(F, L, np.where(np.isfinite(Yst), Yst, np.nan), facecolors=col, rstride=1, cstride=1, linewidth=0,
                   antialiased=False, shade=False, rasterized=True)
    a.plot_surface(F, L, np.where(np.isfinite(Ysd), Ysd, np.nan), color='#9a9993', alpha=0.35, linewidth=0,
                   antialiased=False, shade=False, rasterized=True)
    floor = np.where(F >= fT, 0.0, np.nan)
    a.plot_surface(F, L, floor, color=S.PREY, alpha=0.25, linewidth=0, shade=False, rasterized=True)
    a.plot([fT, fT], [lc[0], lc[-1]], [0, 0], color=S.INK, lw=1.4)
    yF = []
    for c, f in zip(cc, ff):
        eq = interior(P.with_(c=float(c), f=f - 1e-9)); yF.append(np.mean([e[1] for e in eq]) if eq else 0)
    a.plot(ff, np.log10(cc), yF, color=S.BLUE, lw=2.0)
    a.scatter([fT], [np.log10(cs)], [0], s=28, color=S.INK, depthshade=False, zorder=10)
    a.set_xlabel('fear  f'); a.set_ylabel('log$_{10}$ c'); a.set_zlabel('')
    # z label drawn as 2-D text: 3-D axis labels can fall outside the saved bounding box with some matplotlib versions
    a.text2D(-0.17, 0.55, 'raccoons  y*', rotation=90, transform=a.transAxes, va='center', ha='center', fontsize=9.5)
    a.view_init(elev=24, azim=-128); a.set_box_aspect((1.2, 1.0, 0.7))
    for ax_ in (a.xaxis, a.yaxis, a.zaxis):
        ax_.pane.set_facecolor((1, 1, 1, 0)); ax_.pane.set_edgecolor(S.GRID)
    a.grid(False)
    a.text2D(0.02, 0.95, 'a', transform=a.transAxes, fontsize=10.6, fontweight='bold', color=S.INK)
    b = fig.add_subplot(gsp[1])
    Xmap = np.where(np.isfinite(Xst), Xst, 1.0)
    im = b.pcolormesh(fs, 10 ** lc, Xmap, cmap=T.cmap('prey'), shading='auto', rasterized=True, vmin=0, vmax=1)
    b.contourf(fs, 10 ** lc, bist.astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
    b.contour(fs, 10 ** lc, bist.astype(float), levels=[0.5], colors=[S.BLUE], linewidths=1.2)
    b.plot([fT, fT], [0.05, 60], color=S.INK, lw=1.2)
    b.plot(ff, cc, color=S.BLUE, lw=1.6); b.plot(fT, cs, 'o', ms=5, color=S.INK, zorder=5)
    b.axvline(1.0, color=S.FEAR, lw=0.9, ls=(0, (3, 2)))
    b.set_yscale('log'); b.set_xlim(0, FMAX); b.set_ylim(0.05, 60); b.set_xlabel('fear  f'); b.set_ylabel('dilution  c')
    cb = fig.colorbar(im, ax=b, fraction=0.05, pad=0.03); cb.set_label('crabs  x* (stable state)'); cb.outline.set_visible(False)
    S.panel_label(b, 'b', x=-0.18)
    return S.save(fig, 'Fig4_equilibrium_surface')


if __name__ == '__main__':
    print(main())
