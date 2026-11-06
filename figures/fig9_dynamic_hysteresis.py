# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python figures/fig9_dynamic_hysteresis.py
# Also run automatically by:  python run_all.py   (job name: fig9)
# Output:  figures/pdf/Fig9_dynamic_hysteresis*.pdf and figures/png/
# Used in the paper for:  Figure 9
# ----------------------------------------------------------------------------------------------------
"""Figure 9. Dynamic hysteresis and its robustness (data: results/step15_A.json, step15_B.json, step15_C.json).
a  hysteresis loops in the (f, y) plane for three ramp durations (immigration 1e-7), over the static branches
   (E+ solid, E- dashed); shaded: static bistable window. Computed here (about 1 min).
b  recovery delay f_T - f_rec versus ramp duration for three immigration floors: simulation (filled) and the
   entry-exit prediction (open); dashed guide: slope -1/2.
c  critical boost duration D_c versus distance of the background fear from f_T; dashed: logarithmic fit.
d  27 calibrations (x*, retained foraging, delta): dynamic loop width (1000-generation ramp) versus static window
   width; all points above the identity line.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from qtds_fear import style as S
from qtds_fear.model import Params, rhs_log
from qtds_fear.equilibria import interior
from qtds_fear.bifurcation import f_transcritical, fold_by_discriminant

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


def load(name):
    fp = RES / f'step15_{name}.json'
    if not fp.exists():
        sys.exit(f'{fp} missing: run  python analysis/step15_dynamics.py {name}  first')
    return json.loads(fp.read_text())


def loop(P, gens, imm=1e-7, fmax=3.2):
    gen = 1 / P.delta; t_up = t_down = gens * gen; t_hold = 0.4 * gens * gen; t_end = t_up + t_hold + t_down
    def f_t(t):
        if t < t_up: return fmax * t / t_up
        if t < t_up + t_hold: return fmax
        return max(fmax * (1 - (t - t_up - t_hold) / t_down), 0.0)
    F = lambda t, w: rhs_log(t, w, P.with_(f=f_t(t))) + np.array([0.0, imm / np.exp(w[1])])
    ts = np.linspace(0, t_end, 6000)
    sol = solve_ivp(F, (0, t_end), np.log(interior(P.with_(f=0.0))[-1]), t_eval=ts, method='LSODA',
                    rtol=1e-9, atol=1e-11, max_step=gen / 4)
    return np.array([f_t(t) for t in sol.t]), np.exp(sol.y[1])


def panel_a(ax):
    P = Params(s0=1.0, c=5.0, delta=0.03); fT = f_transcritical(P); fF = fold_by_discriminant(P, fmax=10)
    ax.axvspan(fT, fF, color=S.TINT_BISTABLE, lw=0, zorder=0)
    fs = np.linspace(0.0, 3.2, 400); up, lo = [], []
    for f in fs:
        eq = interior(P.with_(f=float(f)))
        up.append(eq[-1][1] if eq else np.nan); lo.append(eq[0][1] if len(eq) == 2 else np.nan)
    ax.plot(fs, up, color=S.INK, lw=1.4, zorder=3); ax.plot(fs, lo, color=S.INK2, lw=1.0, ls=(0, (3, 2)), zorder=3)
    cols = (S.YELLOW, S.MESO, S.MAGENTA)
    for g, col in zip((300, 1000, 10000), cols):
        f, y = loop(P, g)
        ax.plot(f, y, color=col, lw=1.3, alpha=0.95, label=f'{g} gen', zorder=2)
    ax.set_yscale('log'); ax.set_ylim(1e-6, 4); ax.set_xlim(0, 3.25)
    ax.set_xlabel('fear  f'); ax.set_ylabel('mesopredator  y'); ax.legend(loc='lower left')


def panel_b(ax, A):
    """Delay to the exit point (mesopredator back at 5% of its no-fear density): simulation (filled) and the
    entry-exit prediction (open)."""
    fT = A['f_T']
    for imm, col in zip((1e-5, 1e-7, 1e-9), (S.AQUA, S.BLUE, S.VIOLET)):
        rr = [r for r in A['rows'] if r['immigration'] == imm and r['f_exit_sim'] is not None
              and fT - r['f_exit_sim'] > 1e-3]
        g = [r['gens'] for r in rr]
        ax.loglog(g, [fT - r['f_exit_sim'] for r in rr], 'o-', ms=3.4, lw=1.1, color=col,
                  label=f'{imm:.0e}'.replace('e-0', 'e-'))
        pe = [(r['gens'], fT - r['f_exit_pred']) for r in rr if r['f_exit_pred'] is not None and fT - r['f_exit_pred'] > 1e-3]
        ax.loglog([p[0] for p in pe], [p[1] for p in pe], 'o', ms=6, mfc='none', mec=col, mew=0.9)
    xx = np.array([400, 9000]); ax.loglog(xx, 0.75 * (xx / 400) ** -0.5, color=S.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel('ramp duration (generations)'); ax.set_ylabel('recovery delay  $f_T - f_{exit}$')
    ax.legend(loc='lower left', title='immigration', title_fontsize=7.5)


def panel_c(ax, C):
    rows = [r for r in C['rows'] if np.isfinite(r['D_c'])]
    d = np.array([r['d'] for r in rows]); D = np.array([r['D_c'] for r in rows])
    ax.semilogx(d, D, 'o-', ms=3.2, lw=1.2, color=S.FEAR)
    near = d < 0.1 * (C['f_F'] - C['f_T'])
    if near.sum() >= 3:
        k, b = np.polyfit(np.log(1 / d[near]), D[near], 1)
        xx = np.geomspace(d.min(), d.max(), 50); ax.semilogx(xx, k * np.log(1 / xx) + b, color=S.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel('$f_b - f_T$'); ax.set_ylabel('$D_c$ (generations)')


def panel_d(ax, B):
    """Colour: retained foraging; marker: no-fear prey level x*. The three values of delta lie on top of each other
    (delta barely changes the loop measured in generations)."""
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    mk = {0.05: 'o', 0.1: 's', 0.2: '^'}; col = {0.2: S.AQUA, 0.34: S.BLUE, 0.5: S.VIOLET}
    for r in B['rows']:
        if r.get('f_recover') is None or r['delta'] != 0.03:
            continue
        ax.plot(r['f_F'] - r['f_T'], r['f_collapse'] - r['f_recover'], mk[r['x_star']], ms=5.5, mfc=col[r['kept']],
                mec='white', mew=0.6)
    lim = [0.1, 8]; ax.plot(lim, lim, color=S.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(FixedLocator([0.1, 0.3, 1, 3])); a.set_minor_locator(NullLocator())
        a.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlabel('static window  $f_F - f_T$'); ax.set_ylabel('dynamic loop width')
    for k, c_ in col.items():
        ax.plot([], [], 's', color=c_, ms=4.5, label=f'foraging kept {k:g}')
    for xs, m in mk.items():
        ax.plot([], [], m, color=S.INK2, mfc='white', ms=4.5, label=f'x* = {xs:g}')
    ax.legend(loc='lower right', fontsize=7, handletextpad=0.2, labelspacing=0.3)


def main():
    S.apply()
    A, B, C = load('A'), load('B'), load('C')
    fig = plt.figure(figsize=(S.DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.45)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_a(axs[0]); panel_b(axs[1], A); panel_c(axs[2], C); panel_d(axs[3], B)
    for ax, l in zip(axs, 'abcd'):
        S.panel_label(ax, l, x=-0.26)
    print(S.save(fig, 'Fig9_dynamic_hysteresis'))


if __name__ == '__main__':
    main()
