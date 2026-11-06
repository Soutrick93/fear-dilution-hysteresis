# fear-dilution-hysteresis

Code for the paper

> S. Das, *Fear diluted in groups: an emergent Allee effect, a sharp threshold for hysteresis, and the reversal of a
> fear-driven trophic cascade*, submitted to Qualitative Theory of Dynamical Systems (2026).

Every number and figure in the paper is produced by a script in this repository. `LOGBOOK.md` is the dated record of
every computation, result and correction.

## The model

Dimensionless mesopredator (y) and prey (x) dynamics under an external fear cue f:

    x' = x (1 - x) - x y / (s + x)
    y' = delta y [ x / (s + x) * (1 + c y) / (1 + c y + f) - mu ],     s = s0 (1 + q f)

Fear reduces foraging (through s) and imposes a reproductive cost that is shared among conspecifics (dilution strength
c). The main result: the fear-driven exclusion of the mesopredator is hysteretic exactly when c > c*, with c* in closed
form.

## Requirements

- Python 3.10 to 3.13 (Windows, macOS or Linux)
- numpy, scipy, sympy, matplotlib (installed in step 2 below)
- About 4-6 hours for the full run on a 4-core laptop; 8 GB RAM is enough with the default 4 workers

## Install (once)

1. Download or clone this repository and open a terminal in its folder.
2. Create a virtual environment and install the packages:

   Windows (PowerShell):

       python -m venv venv
       venv\Scripts\activate
       pip install -r requirements.txt

   macOS / Linux:

       python3 -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt

   If PowerShell refuses to run `activate`, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Run everything

    python run_all.py

This runs every analysis step in order, then draws Figures 1-9. Progress and expected times are printed. Outputs:

| Folder | Content |
| --- | --- |
| `results/` | one JSON file per step (all numbers quoted in the paper) |
| `results/logs/` | screen output and errors of every job; `results/run_log.json` lists which jobs succeeded |
| `results/quicklook/` | diagnostic plots (not in the paper) |
| `figures/pdf/` | publication figures, vector PDF with editable text (Type 42 fonts, opens in Illustrator) |
| `figures/png/` | 300 dpi previews |

Other ways to run:

    python run_all.py --resume          # after a crash or Ctrl-C: skips jobs that already finished OK
    python run_all.py step6 fig2        # only the named jobs
    python make_figures.py              # only the figures (needs the results of the analysis steps)
    python make_figures.py 4 9          # only Figures 4 and 9
    python analysis/step15_dynamics.py A C    # one script, only parts A and C

Every script starts with a "HOW TO RUN" comment block giving its command, its output file and what it is used for in
the paper.

### Settings (environment variables)

| Variable | Default | Effect |
| --- | --- | --- |
| `QTDS_WORKERS` | 4 | parallel processes for the heavy steps; use 2 if memory is short |
| `QTDS_DRAFT` | 0 | `1` = tiny grids and low-resolution figures, for a quick test (minutes) |
| `QTDS_ANNOTATE` | 0 | `1` = add explanatory text on figure panels |

Windows PowerShell: `$env:QTDS_WORKERS=2`; Windows cmd: `set QTDS_WORKERS=2`; macOS/Linux: `export QTDS_WORKERS=2`.

Quick test that everything works (about 10 minutes):

    $env:QTDS_DRAFT=1; python run_all.py; $env:QTDS_DRAFT=0      # PowerShell
    QTDS_DRAFT=1 python run_all.py                                # macOS / Linux

## Jobs and where they are used

| Job | Script | Used in the paper for | Time |
| --- | --- | --- | --- |
| symbolic | `qtds_fear/symbolic.py` | symbolic verification of scaling, boundedness, the cubic, c* | 1-3 min |
| step5, step5b | `analysis/step5_*.py` | number of coexistence equilibria (theorem and random test) | 3-6 min |
| step6 | `analysis/step6_transcritical_fold.py` | threshold c*, fold curve, Sotomayor checks | 3-6 min |
| step7 | `analysis/step7_hopf.py` | Hopf points | 1-2 min |
| step8a | `analysis/step8a_bt.py` | Bogdanov-Takens curve | 1-2 min |
| step8 | `analysis/step8_global.py` | basins (its fate-bisection homoclinic search is superseded by step8e) | 15-30 min |
| step8c, step8d | `analysis/step8c_*.py`, `step8d_*.py` | global convergence test (support for the conjecture) | up to 25 min |
| step8e, step8f | `analysis/step8e_*.py`, `step8f_*.py` | canard threshold f_c and its diagnostics | 10-20 min |
| step9 | `analysis/step9_slowfast.py` | what a 28-day experiment measures (slow-fast) | 3-6 min |
| step10 | `analysis/step10_chain3d.py` | appendix: living carnivore population | 15-30 min |
| step11 | `analysis/step11_robustness.py` | elasticity criterion for four dilution functions | 2-4 min |
| step12a/b/c | `analysis/step12*_*.py` | Hopf direction, global dynamics (c = 0), BT normal form | 15-35 min |
| step13 | `analysis/step13_calibration_robustness.py` | robustness to calibration | < 2 min |
| step14 | `analysis/step14_checks.py` | ramps, single window, stability of the upper state, BT zeros, parity | 10-20 min |
| step15 | `analysis/step15_dynamics.py` | ramp-rate scaling, sensitivity, critical boost, Hopf and BT scans, error on f_c | 1-2 h |
| quicklook | `analysis/quicklook.py` | diagnostic plots | < 1 min |
| fig1-fig9 | `figures/fig*_*.py` | Figures 1-9 | 30-70 min |

## Repository layout

    qtds_fear/        the model and the numerical tools (library; imported by the scripts)
      model.py          parameters, calibration constants, right-hand side, Jacobian, log-variable form
      equilibria.py     boundary and interior equilibria (exact roots of the cubic)
      bifurcation.py    f_T, c*, folds, Hopf points, first Lyapunov coefficient, window-opening coefficient
      btnf.py           Bogdanov-Takens points and normal-form coefficients
      homoclinic.py     saddle manifolds and the canard/homoclinic threshold (manifold method)
      simulate.py, slowfast.py, chain3d.py   time integration, slow-fast tools, 3-D extension
      symbolic.py       SymPy derivations (writes results/symbolic_report.md)
      style.py, figtools.py   figure style and shared plotting tools
    analysis/         one script per analysis step
    figures/          one script per figure
    run_all.py        runs everything in order
    make_figures.py   runs the figure scripts only
    LOGBOOK.md        dated record of every computation

## Numerical notes

- Interior equilibria are computed as roots of a cubic with an admissibility test, so no continuation software is used.
- Long and stiff integrations use logarithmic variables (ln x, ln y), which keeps densities positive near x = 0.
- Slow ramps include a small mesopredator immigration term (1e-7 unless stated), so that recovery times are finite.
- Results do not depend on the number of workers; only the run time does.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `BrokenProcessPool` | memory ran out: set `QTDS_WORKERS=2` and run `python run_all.py --resume` |
| a job FAILED | read `results/logs/<job>.err.txt`, then `python run_all.py --resume` |
| `ModuleNotFoundError` | the virtual environment is not active: run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` |
| fonts look different | install Arial; otherwise Liberation Sans or DejaVu Sans is used (text stays editable) |

## Citation and licence

Please cite the paper (and this repository, see `CITATION.cff`). Code released under the MIT licence (`LICENSE`).
