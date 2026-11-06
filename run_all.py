"""Run every analysis step and then every figure, in order, logging runtimes and outputs.

Usage (from the repository folder, with the virtual environment active):
    python run_all.py                  # everything: all analysis steps, then Figures 1-9 (about 4-6 h)
    python run_all.py --resume         # after a crash or Ctrl-C: skip jobs that already finished OK
    python run_all.py step6 fig2       # only the named jobs
Parallel workers: 4 by default; set QTDS_WORKERS=2 if memory is short (BrokenProcessPool error).
Outputs go to results/, results/logs/, figures/pdf/ and figures/png/.
"""
import json, subprocess, sys, time, pathlib, zipfile

ROOT = pathlib.Path(__file__).resolve().parent
JOBS = [  # name, command, rough run time on a 4-core laptop
    ('symbolic', [sys.executable, '-m', 'qtds_fear.symbolic'], '1-3 min'),
    ('step5', [sys.executable, 'analysis/step5_equilibria.py'], '2-4 min'),
    ('step5b', [sys.executable, 'analysis/step5_descartes.py'], '1-2 min'),
    ('step6', [sys.executable, 'analysis/step6_transcritical_fold.py'], '3-6 min'),
    ('step7', [sys.executable, 'analysis/step7_hopf.py'], '1-2 min'),
    ('step8a', [sys.executable, 'analysis/step8a_bt.py'], '1-2 min'),
    ('step8', [sys.executable, 'analysis/step8_global.py'], '15-30 min (homoclinic + basins)'),
    ('step8c', [sys.executable, 'analysis/step8c_global_test.py'], 'up to 25 min (global convergence test)'),
    ('step8d', [sys.executable, 'analysis/step8d_recheck_failures.py'], '< 1 min (recheck flagged cases)'),
    ('step8e', [sys.executable, 'analysis/step8e_homoclinic_manifold.py'], '5-15 min on 4+ cores (homoclinic by manifold method)'),
    ('step8f', [sys.executable, 'analysis/step8f_homoclinic_diagnostic.py'], '2-5 min (is the flip a true homoclinic?)'),
    ('step9', [sys.executable, 'analysis/step9_slowfast.py'], '3-6 min'),
    ('step10', [sys.executable, 'analysis/step10_chain3d.py'], '15-30 min (3-D ramps)'),
    ('step11', [sys.executable, 'analysis/step11_robustness.py'], '2-4 min (other dilution functions)'),
    ('step12a', [sys.executable, 'analysis/step12a_hopf_direction.py'], '5-15 min (Hopf direction: proof c=0 + grid)'),
    ('step12b', [sys.executable, 'analysis/step12b_global_stability.py'], '5-15 min (Lyapunov c=0 + Dulac c>0)'),
    ('step12c', [sys.executable, 'analysis/step12c_bt_normal_form.py'], '2-4 min (BT nondegeneracy)'),
    ('step13', [sys.executable, 'analysis/step13_calibration_robustness.py'], '< 2 min (calibration robustness)'),
    ('step14', [sys.executable, 'analysis/step14_checks.py'], '10-20 min (checks A-E)'),
    ('step15', [sys.executable, 'analysis/step15_dynamics.py'], '1-2 h (dynamic hysteresis, sensitivity, accuracy: A-F)'),
    ('quicklook', [sys.executable, 'analysis/quicklook.py'], '< 1 min (diagnostic plots)'),
    ('fig1', [sys.executable, 'figures/fig1_concept.py'], '< 1 min'),
    ('fig2', [sys.executable, 'figures/fig2_theorem_map.py'], '3-8 min'),
    ('fig3', [sys.executable, 'figures/fig3_phase_portraits.py'], '3-10 min'),
    ('fig4', [sys.executable, 'figures/fig4_playback_experiment.py'], '5-15 min'),
    ('fig5', [sys.executable, 'figures/fig5_equilibrium_surface.py'], '2-5 min'),
    ('fig6', [sys.executable, 'figures/fig6_cycles_canard.py'], '5-15 min'),
    ('fig7', [sys.executable, 'figures/fig7_regime_atlas.py'], '5-15 min'),
    ('fig8', [sys.executable, 'figures/fig8_robustness.py'], '< 2 min'),
    ('fig9', [sys.executable, 'figures/fig9_dynamic_hysteresis.py'], '1-2 min'),
]


def main(selected):
    import os
    os.environ.setdefault('QTDS_WORKERS', '4')        # parallel workers for the heavy steps; lower to 2 if memory is short
    os.environ.setdefault('QTDS_DRAFT', '0')          # full-resolution unless QTDS_DRAFT=1 is set
    resume = '--resume' in selected; selected = {x for x in selected if x != '--resume'}
    prev = {}
    if resume and (ROOT / 'results' / 'run_log.json').exists():
        prev = {r['job']: r['ok'] for r in json.loads((ROOT / 'results' / 'run_log.json').read_text())}
    logs = ROOT / 'results' / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, cmd, est in JOBS:
        if selected and name not in selected:
            continue
        if resume and prev.get(name):
            print(f'    {name}: already OK in the previous run, skipped (--resume)'); summary.append(dict(job=name, ok=True, seconds=0, returncode=0)); continue
        print(f'[{time.strftime("%H:%M:%S")}] {name}  (expected {est}) ...', flush=True)
        t0 = time.time()
        # stream the job's own progress lines to the screen while also saving them
        with open(logs / f'{name}.out.txt', 'w') as fo, open(logs / f'{name}.err.txt', 'w') as fe:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=fe, text=True, bufsize=1,
                                    env={**__import__('os').environ, 'PYTHONUNBUFFERED': '1'})
            for line in proc.stdout:
                print('      ' + line.rstrip(), flush=True); fo.write(line)
            proc.wait()
        dt = time.time() - t0
        ok = proc.returncode == 0
        summary.append(dict(job=name, ok=ok, seconds=round(dt, 1), returncode=proc.returncode))
        print(f'    {"OK" if ok else "FAILED"} in {dt:.0f} s' + ('' if ok else f'  -> see results/logs/{name}.err.txt'))
        (ROOT / 'results' / 'run_log.json').write_text(json.dumps(summary, indent=1))   # saved after every job
    (ROOT / 'results' / 'run_log.json').write_text(json.dumps(summary, indent=1))
    bad = [r['job'] for r in summary if not r['ok']]
    print('\nDone.' + (f' FAILED jobs: {bad}  (see results/logs/; rerun with: python run_all.py --resume)' if bad else ' All jobs OK.'))
    print('Results: results/   Figures: figures/pdf/ (editable) and figures/png/   Logs: results/logs/')


if __name__ == '__main__':
    main(set(sys.argv[1:]))
