# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig1_concept.py
# Also run automatically by:  python run_all.py   (job name: fig1)
# Output:  figures/pdf/Fig1_concept_mechanism_data.pdf and figures/png/
# Used in the paper for:  Figure 1
# ----------------------------------------------------------------------------------------------------
"""Figure 1. The question, the mechanism and the data behind the model.
a  food-chain schematic with the two fear channels
b  emergent Allee effect: raccoon per-capita growth rises with raccoon density only when fear is diluted (c > 0)
c  the Suraci et al. 2016 numbers that calibrate the model
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from qtds_fear import style as S
from qtds_fear.model import Params, per_capita_growth_y


def box(ax, xy, w, h, text, sub, color):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.03',
                                fc=color, ec='none', alpha=0.16, transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.03',
                                fc='none', ec=color, lw=1.1, transform=ax.transAxes))
    ax.text(x + w / 2, y + h * 0.62, text, ha='center', va='center', fontsize=7.8, fontweight='bold',
            transform=ax.transAxes)
    ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=6.8, color=S.INK2, transform=ax.transAxes)


def arrow(ax, p0, p1, color, ls='-', lw=1.2, rad=0.0, text=None, toff=(0, 0), ha='center'):
    ax.add_patch(FancyArrowPatch(p0, p1, transform=ax.transAxes, arrowstyle='-|>', mutation_scale=8,
                                 color=color, lw=lw, ls=ls, connectionstyle=f'arc3,rad={rad}'))
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1]
        ax.text(mx, my, text, transform=ax.transAxes, fontsize=6.8, color=color, ha=ha, va='center')


def panel_a(ax):
    ax.set_axis_off()
    W, H = 0.17, 0.22
    box(ax, (0.02, 0.72), W, H, 'Large carnivores', 'or their calls (playback)', S.FEAR)
    box(ax, (0.02, 0.39), W, H, 'Raccoons  y', 'mesopredator, slow (years)', S.MESO)
    box(ax, (0.02, 0.06), W, H, 'Intertidal crabs  x', 'prey, fast (weeks)', S.PREY)
    arrow(ax, (0.105, 0.705), (0.105, 0.625), S.FEAR, ls=(0, (3, 2)), text='fear cue  f  (no killing)',
          toff=(0.01, 0), ha='left')
    arrow(ax, (0.105, 0.375), (0.105, 0.295), S.INK2, text='eaten by raccoons', toff=(0.01, 0), ha='left')
    for r in (0.02, 0.034, 0.048):
        ax.add_patch(Arc((0.198, 0.83), r, r * 4.2, theta1=-60, theta2=60, transform=ax.transAxes, color=S.FEAR, lw=0.8))
    X1 = 0.30
    ax.text(X1, 0.92, 'Two fear channels', fontsize=8, fontweight='bold', transform=ax.transAxes)
    ax.text(X1, 0.78, '1  Raccoons forage less', fontsize=7.5, fontweight='bold', color=S.FEAR, transform=ax.transAxes)
    ax.text(X1, 0.69, r'half-saturation  $s = s_0\,(1 + q f)$', fontsize=7.2, transform=ax.transAxes)
    ax.text(X1, 0.615, '66% less foraging at playback fixes q = 1.94', fontsize=6.7, color=S.INK2, transform=ax.transAxes)
    ax.text(X1, 0.46, '2  Fear costs reproduction, less in groups', fontsize=7.5, fontweight='bold', color=S.FEAR,
            transform=ax.transAxes)
    ax.text(X1, 0.33, r'birth factor  $\dfrac{1 + c\,y}{1 + c\,y + f}$', fontsize=7.4, transform=ax.transAxes)
    ax.text(X1, 0.20, 'many eyes: the cost fades as raccoons become\ncommon (c = dilution strength)', fontsize=6.7,
            color=S.INK2, transform=ax.transAxes, va='top')
    X2 = 0.685
    ax.add_patch(FancyBboxPatch((X2 - 0.012, 0.06), 0.325, 0.88, boxstyle='round,pad=0.01,rounding_size=0.02',
                                fc='#f6f6f4', ec='none', transform=ax.transAxes))
    ax.text(X2, 0.86, 'Model (dimensionless)', fontsize=8, fontweight='bold', transform=ax.transAxes)
    ax.text(X2, 0.70, r"$x' = x(1-x) - \dfrac{x\,y}{s(f)+x}$", fontsize=7.8, transform=ax.transAxes)
    ax.text(X2, 0.50, r"$y' = \delta\, y\left[\dfrac{x}{s(f)+x}\;\dfrac{1+cy}{1+cy+f} - \mu\right]$", fontsize=7.8,
            transform=ax.transAxes)
    ax.text(X2, 0.31, 'x crabs (fraction of K)   y raccoons\nf fear cue (f = 1: playback level)\n'
            r'$\mu$ raccoon death   $\delta \approx 0.03$ raccoon/crab rate', fontsize=6.7, color=S.INK2,
            transform=ax.transAxes, va='top', linespacing=1.5)


def panel_b(ax):
    ys = np.linspace(0, 3, 400)
    x_fixed = 0.5
    for f, col, lab in ((0.0, S.INK2, 'no fear'), (1.0, S.BLUE, 'f = 1 (playback)'), (1.5, S.FEAR, 'f = 1.5')):
        p = Params(s0=1.0, c=5.0, f=f, delta=1.0)
        g = [per_capita_growth_y(x_fixed, yy, p) for yy in ys]
        ax.plot(ys, g, color=col, lw=1.4, label=lab)
        if f > 0:
            p0 = Params(s0=1.0, c=0.0, f=f, delta=1.0)
            g0 = [per_capita_growth_y(x_fixed, yy, p0) for yy in ys]
            ax.plot(ys, g0, color=col, lw=1.0, ls=(0, (3, 2)))
        if f > 0:
            g = np.array(g); k = np.where(np.diff(np.sign(g)))[0]
            if len(k):
                yA = ys[k[0]]
                ax.plot([yA], [0], 'o', ms=4, mfc='white', mec=col, mew=1.1, zorder=5)
    ax.axhline(0, color=S.INK2, lw=0.5)
    ax.set_xlabel('raccoon density  y'); ax.set_ylabel(r'raccoon per-capita growth  $(\dot{y}/y)/\delta$')
    ax.set_xlim(0, 3); ax.set_ylim(-0.1, 0.28)
    ax.legend(loc='center right', bbox_to_anchor=(1.02, 0.6), handlelength=1.6, fontsize=7.5)
    ax.text(2.95, -0.098, 'solid: fear cost shared (c = 5);  dashed: not shared (c = 0)', fontsize=6.5,
            color=S.INK2, ha='right', va='bottom')
    ax.text(0.15, 0.135, 'open circles: Allee thresholds;\nbelow them raccoons decline', fontsize=6.5, color=S.FEAR,
            va='bottom')
    ax.text(0.05, 0.255, 'crabs held at x = 0.5', fontsize=6.5, color=S.INK2, va='bottom')
    S.title(ax, 'Fear creates an Allee effect')


def panel_c(ax):
    items = [('raccoon foraging\nunder playback', -66, S.MESO, 'sets q = 1.94'),
             ('intertidal crabs', 97, S.PREY, 'sets crab rate'),
             ('intertidal fish', 81, S.PREY, ''), ('polychaete worms', 59, S.PREY, ''),
             ('red rock crabs', 61, S.PREY, ''),
             ('crabs with vs without\nraccoons (earlier study)', -90, S.MESO, 'sets μ')]
    yy = np.arange(len(items))[::-1]
    for (lab, v, col, note), yv in zip(items, yy):
        ax.barh(yv, v, color=col, height=0.62, alpha=0.9)
        ax.text(v + (4 if v > 0 else -4), yv, f'{v:+d}%', va='center', ha='left' if v > 0 else 'right', fontsize=6.8)
        if note:
            ax.text(150, yv, note, va='center', ha='left', fontsize=6.6, color=S.INK2, style='italic')
    ax.set_yticks(yy); ax.set_yticklabels([i[0] for i in items], fontsize=6.6)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_xlim(-128, 215); ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel('change under carnivore playback (%)')
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    S.title(ax, 'Field data used (Suraci et al. 2016)')


def main():
    S.apply()
    fig = plt.figure(figsize=(S.DOUBLE * 1.08, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.62, width_ratios=[1, 1.05])
    a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])
    panel_a(a); panel_b(b); panel_c(c)
    S.panel_label(a, 'a', x=-0.02, y=0.98); S.panel_label(b, 'b', x=-0.2); S.panel_label(c, 'c', x=-0.6)
    return S.save(fig, 'Fig1_concept_mechanism_data')


if __name__ == '__main__':
    print(main())
