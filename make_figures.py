"""Render all publication figures at full resolution.

    python make_figures.py              all figures
    python make_figures.py 3 4          only Figures 3 and 4
    QTDS_DRAFT=1 python make_figures.py quick low-resolution check (about 1 min in total)
    QTDS_ANNOTATE=1 python make_figures.py   add explanatory notes on panels (default: text-light, for Illustrator)

Output: figures/pdf/*.pdf (vector, text editable in Illustrator: Type 42 fonts; dense colour fields are embedded as
high-resolution images inside the vector file) and figures/png/*.png (300 dpi previews).
Figures 6-8 read results/*.json from the analysis steps if present (step8e, step10, step11, step12c, step13)."""
import subprocess, sys, time, pathlib, os
ROOT = pathlib.Path(__file__).resolve().parent
FIGS = {1: 'fig1_concept.py', 2: 'fig2_theorem_map.py', 3: 'fig3_phase_portraits.py', 4: 'fig4_playback_experiment.py',
        5: 'fig5_equilibrium_surface.py', 6: 'fig6_cycles_canard.py', 7: 'fig7_regime_atlas.py', 8: 'fig8_robustness.py',
        9: 'fig9_dynamic_hysteresis.py'}
EST = {1: '<1 min', 2: '1-3 min', 3: '3-8 min', 4: '5-15 min', 5: '1-3 min', 6: '5-15 min', 7: '2-5 min', 8: '<2 min', 9: '1-2 min'}
sel = [int(a) for a in sys.argv[1:]] or list(FIGS)
for k in sel:
    print(f'[{time.strftime("%H:%M:%S")}] Figure {k} ({FIGS[k]}, expected {EST[k]}) ...', flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / 'figures' / FIGS[k])], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONWARNINGS': 'ignore'})
    ok = r.returncode == 0
    print(f'    {"OK" if ok else "FAILED"} in {time.time() - t0:.0f} s  ' + (r.stdout.strip().splitlines() or [''])[-1])
    if not ok:
        print(r.stderr[-2000:])
print('PDFs in figures/pdf/, previews in figures/png/')
