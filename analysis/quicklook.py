# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/quicklook.py
# Also run automatically by:  python run_all.py   (job name: quicklook)
# Output:  results/quicklook/*.png
# Used in the paper for:  diagnostic plots only (not in the paper)
# ----------------------------------------------------------------------------------------------------
"""Quick-look diagnostic plots from results/*.json (not publication figures). Writes results/quicklook/*.png.
Each panel is skipped (with a note) if its results file is missing."""
import json, pathlib, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / 'results'; OUT = R / 'quicklook'; OUT.mkdir(parents=True, exist_ok=True)
load = lambda n: json.loads((R / n).read_text())
notes = []


def step6():
    d = load('step6.json')
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    cs = d['c_star_vs_s0']; ax[0].loglog([r['s0'] for r in cs], [r['c_star'] for r in cs], 'o-')
    ax[0].set(xlabel='s0', ylabel='c*', title='c*(s0), closed form')
    fc = [r for r in d['fold_curve_s0_1'] if r['f_F'] is not None]
    ax[1].semilogx([r['c'] for r in fc], [r['f_F'] for r in fc], 'o-', label='fold f_F')
    ax[1].axhline(d['f_T_s0_1'], color='k', ls='--', label='f_T'); ax[1].axvline(d['c_star_s0_1'], color='r', ls=':', label='c*')
    ax[1].set(xlabel='c', ylabel='f', title='fold curve, s0 = 1'); ax[1].legend()
    for s0, rows in d['window_width'].items():
        rr = [r for r in rows if r['width']]; ax[2].loglog([r['c'] for r in rr], [r['width'] for r in rr], 'o-', label=f's0={s0}')
    ax[2].set(xlabel='c', ylabel='f_F - f_T', title='hysteresis window'); ax[2].legend()
    fig.tight_layout(); fig.savefig(OUT / 'step6.png', dpi=130)


def step7():
    d = load('step7.json'); g = d['hopf_grid']
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for dl in sorted(set(r['delta'] for r in g)):
        rr = [r for r in g if r['delta'] == dl]
        ax[0].semilogy([r['s0'] for r in rr], [-r['l1'] for r in rr], '.', label=f'delta={dl}')
    ax[0].set(xlabel='s0', ylabel='-l1 (all positive = supercritical)', title=f"{d['l1_summary']['n']} Hopf points, "
              f"{d['l1_summary']['n_positive']} with l1 > 0"); ax[0].legend(fontsize=7)
    c0 = d['c0_check']; ax[1].plot([r['s0'] for r in c0], [r['f_H_closed'] for r in c0], 'o', label='closed form (c=0)')
    ax[1].plot([r['s0'] for r in c0], [r['f_H_numeric'] for r in c0], 'x', ms=10, label='numerical')
    ax[1].set(xlabel='s0', ylabel='f_H', title='Hopf point, c = 0'); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / 'step7.png', dpi=130)


def step8a():
    d = load('step8a_bt.json')
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for c, rows in d.items():
        if c == 'check': continue
        rr = [r for r in rows if r['delta_BT']]
        ax.loglog([r['s0'] for r in rr], [r['delta_BT'] for r in rr], 'o-', label=f'c={c}')
    ax.axhline(0.03, color='k', ls='--'); ax.text(0.006, 0.04, 'raccoon-like delta = 0.03')
    ax.set(xlabel='s0', ylabel='delta_BT', title='Bogdanov-Takens curve'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'step8a.png', dpi=130)


def step8():
    d = load('step8.json')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for dl in sorted(set(r['delta'] for r in d['homoclinic'])):
        rr = [r for r in d['homoclinic'] if r['delta'] == dl]
        ax[0].plot([r['s0'] for r in rr], [r['f_H'] for r in rr], 'o-', label=f'f_H, delta={dl}')
        hh = [r for r in rr if r['f_hom']]
        ax[0].plot([r['s0'] for r in hh], [r['f_hom'] for r in hh], 'x', ms=9, label=f'f_hom, delta={dl}')
    rr = d['homoclinic']; ax[0].plot([r['s0'] for r in rr], [r['f_T'] for r in rr], 'k--', label='f_T')
    ax[0].set(xlabel='s0', ylabel='f', title='Hopf, homoclinic, transcritical (c = 5)'); ax[0].legend(fontsize=7)
    b = d['basin_fraction']; ax[1].plot([r['f'] for r in b], [r['fraction_persist'] for r in b], 'o-')
    ax[1].set(xlabel='f', ylabel='fraction of starts where raccoons persist', title='basin shrinks before the fold (2.49)')
    fig.tight_layout(); fig.savefig(OUT / 'step8.png', dpi=130)
    if 'global_test' in d:
        g = d['global_test']; notes.append(f"step8c global test: {g['parameter_sets']} parameter sets x {g['initial_conditions_each']} starts, "
                                           f"{len(g['failures'])} failures")


def step9():
    d = load('step9.json')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for dl, v in d['canard'].items():
        ax[0].plot(v['f'], v['amp_y'], 'o-', ms=3, label=f'delta={dl}'); ax[0].axvline(v['f_H'], ls=':', color='gray')
    ax[0].set(xlabel='f', ylabel='raccoon cycle amplitude', title='cycles shrink with fear (s0=0.3, c=5)'); ax[0].legend()
    pb = d['playback_28d']
    ax[1].bar([str(r['s0']) for r in pb], [100 * (r['full_model_crab_gain_28d'] - 1) for r in pb], label='crabs')
    ax[1].bar([str(r['s0']) for r in pb], [r['raccoon_change_28d_percent'] for r in pb], label='raccoons')
    ax[1].axhline(97, color='k', ls='--'); ax[1].set(xlabel='s0', ylabel='% change after 28 days', title='playback: model vs +97% observed')
    ax[1].legend(); fig.tight_layout(); fig.savefig(OUT / 'step9.png', dpi=130)


def step10():
    d = load('step10_chain3d.json')
    keys = list(d); fig, ax = plt.subplots(2, len(keys) // 2, figsize=(4 * len(keys) // 2, 6), squeeze=False)
    for k, a in zip(keys, ax.flat):
        v = d[k]; up, dn = np.array(v['up']), np.array(v['down'])
        a.plot(v['KP'], up[:, 0], '-', color='tab:green', label='crabs, KP up'); a.plot(v['KP'], dn[:, 0], '--', color='tab:green', label='crabs, KP down')
        a.plot(v['KP'], up[:, 1], '-', color='tab:orange', label='raccoons, KP up'); a.plot(v['KP'], dn[:, 1], '--', color='tab:orange', label='raccoons, KP down')
        a.set(title=k + f"\nmax gap {v['max_gap_y']:.2f}", xlabel='carnivore capacity KP'); a.title.set_fontsize(8)
    ax.flat[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(OUT / 'step10.png', dpi=130)


def step11():
    d = load('step11_robustness.json')
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, v in d.items():
        if v['c_pred'] and v['c_num']:
            ax.loglog(v['c_pred'], v['c_num'], 'o', label=k.split('|')[1].strip()[:12] + ' ' + k.split('|')[0])
    lim = [0.03, 2]; ax.plot(lim, lim, 'k--'); ax.set(xlabel='predicted threshold (criterion)', ylabel='numerical threshold',
                                                     title='general hysteresis criterion'); ax.legend(fontsize=5)
    fig.tight_layout(); fig.savefig(OUT / 'step11.png', dpi=130)


for fn in (step6, step7, step8a, step8, step9, step10, step11):
    try:
        fn(); print('ok  ', fn.__name__)
    except FileNotFoundError as e:
        print('skip', fn.__name__, '(missing', pathlib.Path(e.filename).name + ')')
    except Exception:
        print('FAIL', fn.__name__); traceback.print_exc()
(OUT / 'notes.txt').write_text('\n'.join(notes))
print('\n'.join(notes))
