# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig7_regime_atlas.py
# Also run automatically by:  python run_all.py   (job name: fig7)
# Output:  figures/pdf/Fig7_regime_atlas*.pdf and figures/png/
# Used in the paper for:  Figure 7
# ----------------------------------------------------------------------------------------------------
"""Figure 7. Regime atlas in the (half-saturation s0, fear f) plane, c = 5, with the codimension-2/3 structure.
a  regimes coloured: raccoon-free, stable coexistence, boom-bust cycles, bistable (raccoon-free + stable state),
   bistable with cycles; curves: transcritical f_T, fold f_F, Hopf f_H, canard thresholds f_c (points, step 8e).
   The Suraci playback level f = 1 is dashed.
b  Bogdanov-Takens curve delta_BT(s0) for c = 2, 5, 20, with the codimension-3 points where a or b changes sign
   (exact zeros of b from results/step14_D.json, if present). Shaded: raccoon-like delta.
Runtime ~2-5 min (draft ~20 s)."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from qtds_fear import style as S
from qtds_fear import figtools as T
from qtds_fear.model import Params, jacobian
from qtds_fear.equilibria import interior
from qtds_fear.bifurcation import f_transcritical, fold_points, hopf_points

ROOT = pathlib.Path(__file__).resolve().parents[1]
C, DL = 5.0, 0.05


def main():
    S.apply()
    fig = plt.figure(figsize=(S.DOUBLE, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.32)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    s0s = np.geomspace(0.01, 3, T.res(240, 50)); fs = np.linspace(0, 4, T.res(240, 50))
    R = np.array([[T.regime(Params(s0=float(s0), c=C, delta=DL, f=float(f))) for s0 in s0s] for f in fs])
    cols = [S.TINT_FREE, S.TINT_COEX, S.TINT_BISTABLE, S.TINT_CYCLE, '#e9d9f2']
    a.pcolormesh(s0s, fs, R, cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(-0.5, 5.5), 5), shading='auto',
                 rasterized=True)
    s_line = np.geomspace(0.01, 3, T.res(120, 30))
    a.plot(s_line, [f_transcritical(Params(s0=float(s), c=C)) for s in s_line], color=S.MESO, lw=1.4)
    fF = [(s, (fold_points(Params(s0=float(s), c=C), fmax=8) or [np.nan])[0]) for s in s_line]
    a.plot(*zip(*fF), color=S.BLUE, lw=1.5)
    hh = [(s, h['f']) for s in s_line for h in hopf_points(Params(s0=float(s), c=C, delta=DL), fmax=4, n=800)]
    if hh: a.plot(*zip(*sorted(hh)), color=S.YELLOW, lw=1.5, ls=(0, (4, 1.5)))
    fp = ROOT / 'results' / 'step8e_homoclinic.json'
    if fp.exists():
        pts = [(r['s0'], max(r['f_hom'])) for r in json.loads(fp.read_text()) if r.get('delta') == DL and r.get('f_hom')
               and r['s0'] <= 0.04]
        if pts: a.plot(*zip(*pts), 'D', ms=4, color=S.RED, mec='white', mew=0.5, zorder=5)
    a.axhline(1.0, color=S.FEAR, lw=0.9, ls=(0, (3, 2)))
    a.set_xscale('log'); a.set_xlim(0.01, 3); a.set_ylim(0, 4)
    a.set_xlabel('half-saturation without fear  s$_0$'); a.set_ylabel('fear  f')
    S.panel_label(a, 'a', x=-0.12)
    # b: BT curves
    fp = ROOT / 'results' / 'step12c_bt_normal_form.json'
    b.axhspan(0.01, 0.1, color=S.TINT_COEX, lw=0)
    for c, col in ((2.0, S.AQUA), (5.0, S.BLUE), (20.0, S.FEAR)):
        rows = []
        for s0 in np.geomspace(0.005, 2.0, T.res(40, 10)):
            p = Params(s0=float(s0), c=c, delta=1.0); fl = fold_points(p, fmax=8.0, n=2000)
            if not fl: continue
            eq = interior(p.with_(f=fl[0] - 1e-11)); e = tuple(np.mean(eq, axis=0)) if len(eq) == 2 else eq[0]
            J = jacobian(*e, p.with_(f=fl[0]))
            if J[0, 0] < 0 < J[1, 1]: rows.append((s0, -J[0, 0] / J[1, 1]))
        if rows: b.loglog(*zip(*rows), color=col, lw=1.6)
        fz = ROOT / 'results' / 'step14_D.json'           # exact zeros of b (codimension-3 points)
        if fz.exists():
            for z in json.loads(fz.read_text())['zeros']:
                if z.get('c') == c and 's0' in z:
                    b.plot(z['s0'], z['delta_BT'], '^', ms=6, mfc='white', mec=col, mew=1.4, zorder=5)
    b.set_xlabel('s$_0$'); b.set_ylabel('δ at Bogdanov-Takens'); b.set_ylim(0.01, 300)
    S.panel_label(b, 'b', x=-0.2)
    return S.save(fig, 'Fig7_regime_atlas')


if __name__ == '__main__':
    print(main())
