# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig8_robustness.py
# Also run automatically by:  python run_all.py   (job name: fig8)
# Output:  figures/pdf/Fig8_robustness*.pdf and figures/png/
# Used in the paper for:  Figure 8
# ----------------------------------------------------------------------------------------------------
"""Figure 8. Robustness of the hysteresis result.
a  dilution needed for hysteresis at every s0 (sup over s0 of c*) across calibrations: crab level without fear x*
   and foraging kept under playback (Suraci calibration = star).
b  the elasticity criterion: predicted vs measured hysteresis thresholds for three dilution functions and three s0
   (from results/step11_robustness.json if present, else recomputed); all on the identity line.
c  living carnivores (3-D extension): raccoon density as carnivore capacity KP is ramped up (solid) and down
   (dashed), for four ecological variants (from results/step10_chain3d.json; run step10 first).
Runtime < 2 min if the results files exist."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from qtds_fear import style as S
from qtds_fear import figtools as T
from qtds_fear.model import Params
from qtds_fear.bifurcation import c_star

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    S.apply()
    fig = plt.figure(figsize=(S.DOUBLE, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.9, 1.1], wspace=1.0)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); c = fig.add_subplot(gs[2])
    # a
    fp = ROOT / 'results' / 'step13_calibration.json'
    if fp.exists():
        d = json.loads(fp.read_text()); XS, KEPT, sup = np.array(d['x_star']), np.array(d['foraging_kept']), np.array(d['sup_c_star'])
    else:
        XS = np.linspace(0.02, 0.4, T.res(20, 8)); KEPT = np.linspace(0.15, 0.8, T.res(14, 6)); S0 = np.geomspace(0.01, 10, T.res(300, 60))
        sup = np.array([[np.nanmax([c_star(Params(s0=float(s0), mu=xs / (s0 + xs), q=1 / k - 1)) or np.nan for s0 in S0])
                         for xs in XS] for k in KEPT])
    im = a.pcolormesh(XS, KEPT, sup, cmap=T.cmap('fear'), shading='gouraud', rasterized=True)
    cs = a.contour(XS, KEPT, sup, levels=[0.4, 0.6, 0.8, 1.2, 1.6], colors='white', linewidths=0.7)
    a.clabel(cs, fmt='%.1f', fontsize=7.1)
    a.plot(0.1, 0.34, '*', ms=10, color=S.MESO, mec='white', mew=0.6)
    a.set_xlabel('crabs with raccoons, no fear  x*'); a.set_ylabel('foraging kept at playback')
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03); cb.outline.set_visible(False); cb.set_label('max$_{s_0}$ c*')
    S.panel_label(a, 'a', x=-0.25)
    # b
    fp = ROOT / 'results' / 'step11_robustness.json'
    lim = [0.04, 2.0]; b.plot(lim, lim, color=S.INK2, lw=0.8, ls=(0, (3, 2)))
    if fp.exists():
        d = json.loads(fp.read_text())
        mk = {'rational': ('o', S.BLUE), 'exponential': ('s', S.AQUA), 'squared': ('^', S.FEAR)}
        ms = {'rational': 3.0, 'exponential': 6.5, 'squared': 5.0}
        for k, v in d.items():
            name = k.split('|')[1].strip().split()[0]
            if v.get('c_pred') and v.get('c_num') and name in mk:
                m, col = mk[name]; b.plot(v['c_pred'], v['c_num'], m, ms=ms[name], mfc=col if name == 'rational' else 'white', mec=col, mew=1.3, zorder=5 if name == 'rational' else 4)
    b.set_xscale('log'); b.set_yscale('log'); b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel('predicted threshold'); b.set_ylabel('measured threshold')
    S.panel_label(b, 'b', x=-0.3)
    # c
    fp = ROOT / 'results' / 'step10_chain3d.json'
    if fp.exists():
        d = json.loads(fp.read_text())
        variants = [('beta=0.0,lam=0.0,c=5.0', S.BLUE), ('beta=0.3,lam=0.0,c=5.0', S.AQUA),
                    ('beta=0.0,lam=0.02,c=5.0', S.FEAR), ('beta=0.3,lam=0.02,c=5.0', S.MESO)]
        for key, col in variants:
            if key not in d: continue
            v = d[key]; KP = np.array(v['KP']); up = np.array(v['up']); dn = np.array(v['down'])
            c.plot(KP, up[:, 1], color=col, lw=1.4); c.plot(KP, dn[:, 1], color=col, lw=1.2, ls=(0, (3, 2)))
            gap = np.abs(up[:, 1] - dn[:, 1]) > 0.05
            if gap.any(): c.fill_between(KP, up[:, 1], dn[:, 1], where=gap, color=col, alpha=0.12, lw=0)
    c.set_xlabel('carnivore capacity  K$_P$'); c.set_ylabel('raccoons  y')
    S.panel_label(c, 'c', x=-0.22)
    return S.save(fig, 'Fig8_robustness')


if __name__ == '__main__':
    print(main())
