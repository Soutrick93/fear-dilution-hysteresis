# Logbook: fear-diluted mesopredator model (QTDS paper)

Project: when can fear alone reverse a trophic cascade, and when is the reversal hysteretic?
Model, symbols and steps follow the plan document "Fear, Allee effects and trophic cascades: QTDS paper plan".
Every number below is produced by a script in this folder; the script name is given with each entry.

Conventions: dimensionless model in `qtds_fear/model.py`
x' = x(1-x) - xy/(s+x),  y' = delta y [ x/(s+x) (1+cy)/(1+cy+f) - mu ],  s = s0(1+qf).
Calibration: q = 1/0.34 - 1 = 1.941 (66% less foraging at playback f = 1), mu = 0.1/(s0+0.1) (crabs at 10% of K without fear).

---

## 2026-11-04 | Step 0: feasibility (earlier today, scripts in ../ prior to this package)
- FearAllee3 (August model) has folds, supercritical Hopf, cusp and a BT point at (a,k) = (0.0915, 1.552), but its bottom
  species is never eaten, so it cannot describe a cascade. Decision: build a new planar model anchored on Suraci 2016.
- First pass of the new model: hysteresis appears only for c > c*, closed form found and checked at three s0 values.

## 2026-11-04 | Step 1: package set-up
- Created `qtds_fear/` (model, equilibria, bifurcation, symbolic, simulate, slowfast, chain3d, style),
  `analysis/` (one script per step, results to `results/`), `figures/` (one module per figure, PDF + PNG).
- Figures: matplotlib with `pdf.fonttype = 42` so text stays editable in Illustrator; font Liberation Sans
  (metric clone of Arial; swap to Arial in Illustrator with Type > Find Font if wanted).

## 2026-11-04 | Step 3: model and nondimensionalisation (`python -m qtds_fear.symbolic`)
- VERIFIED symbolically that the dimensional model maps to the dimensionless one.
- CORRECTION to the plan document: the predator scale is y = M/(rho h K), not M/(rho h), and therefore
  c = gamma rho h K. s0 = 1/(ahK), mu = mh/e, delta = e/(h rho) are unchanged. (Fix section 5.3 of the plan doc.)

## 2026-11-04 | Step 4: well-posedness and boundary equilibria (`python -m qtds_fear.symbolic`)
- VERIFIED: for V = x + y/delta,
  dV/dt = x(1-x) - mu y - f x y/((s+x)(1+cy+f)) <= x(1-x) + mu delta x - mu delta V,
  hence limsup V <= (1 + mu delta)^2/(4 mu delta). Positivity: both axes are invariant. x(t) <= max(x(0), 1).
- VERIFIED: E0 = (0,0) has eigenvalues 1, -delta mu (saddle).
  E1 = (1,0) has eigenvalues -1 and delta[1/((1+s)(1+f)) - mu]; stable iff mu(1+f)(1+s) > 1.

## 2026-11-04 | Step 5: interior equilibria (`analysis/step5_equilibria.py`, `analysis/step5_descartes.py`)
- VERIFIED: interior equilibria are the roots u > 1 of a cubic P(u) (u = 1 + cy). Shifted to w = cy > 0:
  a3 = (1-mu)^2 > 0
  a2 = c s (A - 1) + 2(1-mu)(1-B)
  a1 = c s (fA + 2A - 2) + (1-B)^2
  a0 = c s Lambda,
  where A = mu(1+s), B = mu(1+f), Lambda = mu(1+f)(1+s) - 1 (Lambda > 0 iff E1 stable).
- NEW THEOREM (proof by Descartes' rule of signs):
  (i) If E1 is unstable (Lambda < 0) there is EXACTLY ONE coexistence equilibrium.
      Proof sketch: Lambda < 0 gives A < 1, B < 1 and fA < 1 - A. If a2 < 0 then c s (1-A) > 2(1-mu)(1-B), so
      a1 < -c s (1-A) + (1-B)^2 < (1-B)(2mu - 1 - B) < 0. The sign pattern is (+,+,?,-) or (+,-,-,-): one change.
  (ii) If E1 is stable (Lambda > 0) there are ZERO OR TWO coexistence equilibria (pattern (+,?,?,+)).
  Hence never three, although P is cubic.
- Numerical confirmation, 100,000 random parameter sets (s0, c, f, q, mu over wide log ranges):
  E1 unstable -> 1 interior equilibrium in 36,677/36,677; E1 stable -> 0 in 57,418 and 2 in 5,905; never 3.
  Independent 200,000-sample run: parity rule never violated, no spurious root with D < 0, max count 2.
- Admissibility (proved): if D <= 0 at some u > 1 then (u-1)D^2 >= 0 and -c s u [D - s mu (u+f)] > 0, so P(u) > 0;
  hence every root u > 1 has D > 0, so x > 0, and 1 - x = y/(s+x) > 0 gives x < 1. Every root u > 1 is admissible.

## 2026-11-04 | Step 2: novelty search (web; queries listed so the search can be repeated)
Queries: fear + "group size"/"herd size" + reproduction cost; fear + mesopredator + apex + playback; fear + Allee + vigilance.
- A 2025 review of predator-prey model adaptations (arXiv 2501.16860) lists every fear term it covers; all depend on
  PREDATOR density (1/(1+sy) and variants). None depends on the prey's own density or group size.
- Yadav et al. 2023 (arXiv 2302.04594): fear 1/(1+k alpha y) on predator density; group defence enters predation only.
- Zhang & Qi 2023, QTDS 22:135: fear eta + alpha(1-eta)/(alpha+v), predator-dependent; Hopf theorem + numerics that
  "verify" it. Useful as a QTDS template for how theorems and simulations are presented.
- Antwi-Fordjour & Takyi 2026 (arXiv 2606.17586): fear and Allee are separate terms; Allee threshold set by hand.
- Sasmal & Takeuchi 2021: Allee effect from fear cost + carry-over, no group dilution.
Conclusion: the dilution of the fear cost by the frightened species' own density, and the emergent Allee effect and
hysteresis threshold it produces, were not found in any paper checked. Still recommended: one zbMATH search by the
group before submission (keywords: fear AND (vigilance OR dilution OR "group size")).

## 2026-11-04 | Step 6: transcritical, c*, saddle-node (`analysis/step6_transcritical_fold.py`, `python -m qtds_fear.symbolic`)
- f_T in closed form: the positive root of mu s0 q f^2 + mu(1 + s0 + s0 q) f + mu(1 + s0) - 1 = 0
  (exists iff mu(1+s0) < 1, i.e. raccoons persist without fear).
- Sotomayor transcritical conditions at E1, VERIFIED symbolically: w.F_f = 0 (E1 persists for all f), transversality
  w.DF_f v = delta(...)/(...) > 0, and w.D^2F(v,v) = 2 delta [c f (1+s)^2 - s(1+f)]/[(1+f)^2 (1+s)^3].
  The last vanishes exactly at c = s(1+f)/(f(1+s)^2) = c*: an independent derivation of c* (degenerate transcritical).
- Slope sign of the coexistence branch checked on both sides of c* at s0 = 0.25, 0.5, 1, 2, 4: correct in all 5.
- c*(s0) is NOT monotone (corrected after the workstation run, see below): it rises from 0.267 at s0 = 0.05 to a
  maximum of 0.486 at s0 = 0.25, then falls (0.292 at s0 = 1, 0.078 at s0 = 5).
- Saddle-node (fold) closing the window: Sotomayor nondegeneracy checked numerically at 9 fold points
  (c = 1, 5, 20 x s0 = 0.5, 1, 2): w.F_f and w.D^2F(v,v) both nonzero (order 1e-3 to 1e-4) in all 9.
- Window width f_F - f_T grows like (c - c*)^2 near c* (fitted exponent 1.98), as expected for a degenerate
  transcritical; at s0 = 1 the width is 0.15 at c = 0.81, 0.61 at c = 2.2, 1.25 at c = 6, 2.35 at c = 45.

## 2026-11-04 | Step 7: Hopf (`analysis/step7_hopf.py`)
- c = 0 is exactly Rosenzweig-MacArthur (step 3), so the Hopf point is where x* = s mu(1+f)/(1 - mu(1+f)) = (1 - s)/2.
  Closed form agrees with the numerical Hopf finder to 6 decimals at s0 = 0.1, 0.2, 0.4, 0.6, 0.75.
  A Hopf exists only if s0 < 0.8 (with the calibration x* = 0.1 at f = 0): fear always ends the cycles.
- Bautin search over delta in (0.01, 0.05, 0.3, 1, 3) x c in (0, 0.5, 2, 5, 20, 50) x 18 values of s0:
  509 Hopf points, l1 < 0 in all (0 positive; max l1 = -0.099). Conjecture: the Hopf is always
  supercritical (soft onset). Proof route: perturb the RM result for small c; computer-assisted sign check otherwise.

## 2026-11-04 | Step 8a: Bogdanov-Takens (`analysis/step8a_bt.py`)
- Key observation: equilibria and f_F do not depend on delta; delta scales only the raccoon row of J. At the fold,
  det J = 0 and tr J = J11 + delta j22 with j22 = y g Phi_y > 0 (the dilution term makes raccoon growth
  self-reinforcing). So the fold is a BT point exactly at delta_BT = -J11/j22 (closed-form BT curve).
- delta_BT is 4 to 100 across s0 in (0.005, 3) and c in (2, 5, 20); minimum about 4.1 to 4.6.
  Check at one BT point: both eigenvalues ~1e-6. Interpretation: no BT for slow mesopredators such as raccoons
  (delta about 0.03); BT, and the homoclinic orbits it implies, for fast-breeding mesopredators (delta > ~4).

## 2026-11-04 | Step 8: global dynamics (`analysis/step8_global.py`, results/step8.json)
- Homoclinic bifurcation (c = 5): when the Hopf point lies inside the bistable window (small s0), the stable cycle born
  at f_H grows as fear is lowered and is destroyed on the saddle at f_hom, just below f_H:
  delta = 0.05: s0 = 0.02 -> f_hom = 0.29537 (f_H = 0.2958); 0.03 -> 0.40136 (0.4024); 0.04 -> 0.48499 (0.4870);
  0.05 -> 0.54973 (0.5532). For s0 >= 0.065 the cycle survives down to f_T (no homoclinic in the window).
  delta = 0.1: homoclinic found only for s0 = 0.02 and 0.03. Biological reading: below f_hom, boom-bust cycles carry
  raccoons below their fear-induced Allee threshold, and they collapse.
- Basin of raccoon persistence (s0 = 1, c = 5, delta = 0.05; box x in [0.02, 1], y in [0.01, 3], 21 x 21 starts):
  100% at f = 1.0, 95% at 1.5, 90% at 2.0, 81% at 2.3, 71% at 2.45 (fold at 2.49). The basin shrinks steadily before
  the collapse: a measurable early warning of the tipping point.
- Global convergence test (step8c): interrupted here; to be run on a workstation (run_all.py job "step8").

## 2026-11-04 | Step 9: slow-fast and the 28-day experiment (`analysis/step9_slowfast.py`, results/step9.json)
- With raccoons frozen, +97% crabs in 28 days needs crab plots renewing at 0.045 per day (16.3 to 17.0 per year,
  s0 = 0.5, 1, 2). With a raccoon-like rate (about 0.5 per year) this gives delta = 0.029 to 0.031.
- Full model with that delta: crabs +97.1% after 28 days, raccoons change by only -0.1% to -0.4%.
  So the experiment measured the fast subsystem only; the long-run outcome is set by the slow dynamics.
- Near the Hopf point (s0 = 0.3, c = 5) cycle amplitude falls smoothly to zero as fear rises (supercritical), for
  delta = 0.02 and 0.2; no canard explosion visible at a resolution of 0.0009 in f (it would need exponentially finer
  steps). Fear shrinks boom-bust cycles continuously.

## 2026-11-04 | Step 10: 3-D extension with living carnivores (`analysis/step10_chain3d.py`)
- First run crashed at KP = 0 (division by zero in P/KP). Fixed: KP grid now starts at 0.05 and P/KP is guarded.
  To be run on a workstation.

## 2026-11-04 | Figures
- Fig1_concept_mechanism_data: done (schematic, fear-induced Allee effect, Suraci data). PDF with editable text.
- Fig2_hysteresis_theorem: script written; first render stopped at the fold curve too close to c* (window ~1e-7 wide).
  Fixed by starting the curve at 1.01 c* and adding the exact point (c*, f_T). To be rendered on a workstation.
- Planned next: Fig3 phase portraits with basins; Fig4 in-silico playback experiment (time series, hysteresis in
  time); Fig5 cycles, Hopf and homoclinic regimes; Fig6 slow-fast and the 28-day experiment, BT curve in delta;
  Fig7 3-D carnivore extirpation and reintroduction; FigS1 robustness to other fear functions.

## 2026-11-04 | Step 10: living carnivores, f = kappa P (`analysis/step10_chain3d.py`, results/step10_chain3d.json)
Quasi-static ramps of carnivore capacity KP (0.05 to 3.2, 33 steps) up then down; window = KP range where
raccoons differ by > 0.05 between the two ramps.
- beta = 0, lam = 0 (carnivores do not eat raccoons): c = 5 window KP in [1.43, 2.51], the planar [f_T, f_F] =
  [1.37, 2.49] to grid resolution, as predicted (P -> KP). c = 0.15: no window (one-point gap 0.26 at KP = 1.33 is
  slow convergence next to the transcritical, not hysteresis).
- beta = 0.3 (carnivores benefit from raccoons): c = 5 window narrows to [1.43, 1.92]; c = 0.15 none.
- lam = 0.02 (carnivores also kill raccoons): c = 5 window shifts down to [1.13, 1.63]; c = 0.15 none.
- both: c = 5 window [1.13, 1.33]; c = 0.15 none.
Conclusion: hysteresis survives a dynamic top predator. Lethal predation moves the tipping point to lower carnivore
density; feedback of raccoons onto carnivores narrows the window. Extirpation-then-reintroduction scenario: once
raccoons have crashed, carnivores must fall below the lower edge before raccoons return, and vice versa.

## 2026-11-04 | Step 11: robustness, a general hysteresis criterion (`analysis/step11_robustness.py`)
- General form y' = delta y [g(x) Phi(y; f) - mu], g = x/(s+x), Phi(0; f) = 1/(1+f). Expanding the predator
  nullcline at E1, f = f_T, to first order in y: x = 1 - y/(1+s) + ..., so
  g(x) Phi(y) = mu + y [ g(1) Phi_y(0) - g'(1) Phi(0)/(1+s) ] + (f-derivative term)(f - f_T) + ...
  with g(1) = 1/(1+s), g'(1) = s/(1+s)^2, and the f-derivative term negative. Hence
      BACKWARD (hysteresis)  iff  E := Phi_y(0; f_T)/Phi(0; f_T)  >  s_T/(1+s_T)^2 .
  E is the low-density elasticity of the fear-diluted birth factor. Because s/(1+s)^2 <= 1/4, E > 1/4 is a universal
  sufficient condition for hysteresis, whatever the foraging parameters.
  The paper's Phi gives E = c f/(1+f), recovering c* exactly.
- Numerical test (direct root counting just past f_T), threshold c predicted vs found, s0 = 0.3, 1, 3:
  rational (paper): 0.482/0.487, 0.292/0.295, 0.123/0.124
  exponential 1/(1+f e^{-cy}) (same E): 0.482/0.486, 0.292/0.294, 0.123/0.124
  squared 1/(1+f/(1+cy)^2) (E doubled): 0.241/0.243, 0.146/0.147, 0.062/0.062
  Agreement within 1.2% (the small excess is the finite offset f_T + 1e-5 used for counting).
- Sigmoid dilution 1/(1+f/(1+(cy)^2)) has E = 0, so the first-order criterion fails, yet bistability still appears for
  c > 1.43, 0.67, 0.26 (s0 = 0.3, 1, 3): the next-order term decides. The criterion is sufficient, not necessary,
  when E = 0. Worth one remark in the paper.

## 2026-11-04 | Hand-over
- Heavy runs move to a workstation: `python run_all.py` (or `python run_all.py step8c quicklook` for the one
  remaining analysis, the global convergence test). Diagnostic plots go to results/quicklook/.

## 2026-11-04 | Workstation run (Windows), results returned (`results/step8_workstation.json`, logs in the results folder)
- All jobs reproduced the earlier numbers (symbolic, steps 5 to 11) to the printed precision. fig2 crashed (fixed below).
- Step 8c global convergence test: 60 parameter sets x 5 random starts; 3 flagged. Rechecked with Radau
  (rtol 1e-11, 20000/delta time units) in `analysis/step8d_recheck_failures.py`: ALL THREE CONVERGE to the coexistence
  state (end distance < 1e-15).
  - two flags were solver failures: the orbit passes within ~1e-17 of x = 0 (crabs nearly vanish transiently) and
    LSODA left the positive quadrant / overflowed;
  - one was a weakly damped focus (Re lambda = -0.0012) not yet settled at the end time.
  Result: 60/60 sets, 300/300 starts converge; no cycle or other attractor found where the coexistence state is the
  unique stable equilibrium. step8c now falls back to Radau automatically.
- Homoclinic point at s0 = 0.02, delta = 0.05: workstation 0.29182 vs development machine 0.29537 (f_H = 0.2958). The bisection on
  orbit fate is ill-conditioned there (saddle and focus very close). Other homoclinic values agree to 1e-8.
  Action: confirm by homoclinic continuation in MATCONT (HomCont) before quoting numbers.
- Fig2 crash: fold search too coarse for c within 0.1% of c*. New `fold_by_discriminant`: the fold is where the
  cubic's discriminant changes sign right of f_T. Agrees with grid scanning to 6 decimals where both work
  (s0 = 0.5, 1, 2; c = 2c*, 11c*) and resolves windows of width 1e-6. Fig2 now renders.
- Corollary from the corrected c*(s0): with the Suraci calibration (x* = 0.1, q = 1.94), c* < 0.49 for EVERY s0,
  so any dilution strength c > 0.49 makes fear-driven cascade reversal hysteretic regardless of foraging saturation.

## 2026-11-04 | Step 8e: homoclinic by the manifold method (`qtds_fear/homoclinic.py`, `analysis/step8e_homoclinic_manifold.py`)
- New method: follow the saddle's unstable manifold once around the focus (log variables, so orbits passing within
  1e-15 of x = 0 stay accurate) and record which side of the stable manifold it returns on. The sign flips exactly
  at the homoclinic orbit; bisect to 1e-10. This uses the manifold itself, not an arbitrary orbit near the focus.
- Disputed case s0 = 0.02, delta = 0.05, c = 5: f_hom = 0.295370721 (method above), agreeing with the first development
  run (0.2953707208) to 8 digits. The workstation value 0.2918 came from the old fate-bisection and is discarded.
- s0 = 0.03, delta = 0.05: sign flips between f = 0.4013 and 0.4014, consistent with 0.401358.
- Period cross-check (T ~ -(1/lambda_u) ln(f - f_hom), lambda_u = 0.00195 so slope about -512): first attempt started
  near the focus and measured transient spirals (slope -18, not meaningful). Now starts on the unstable manifold;
  to be run on a workstation with the full step8e job.

## 2026-11-05 | step8e speed-up after the workstation run hung (~3 h, no output)
- Cause: run_all.py hid job output until the end, and the period cross-check integrated with Radau to 800,000 time
  units without stopping. Fixes: run_all streams each job's output live; cycle_period uses LSODA in chunks and stops
  after 6 crossings; bisection tol 1e-9 (was 1e-10); scan 12 points (was 16); the 14 cases run in parallel and each
  result is saved as soon as it finishes.

## 2026-11-05 | step8e results (workstation, 14 cases in 123 s; results/step8e_homoclinic_workstation.json)
- Confirmed (single sign flip, agrees with the old fate method to 7-9 digits):
  delta 0.05: s0 0.02 -> 0.29537072, 0.03 -> 0.40135806, 0.04 -> 0.48498972
  delta 0.10: s0 0.02 -> 0.29535019, 0.03 -> 0.40112163, and NEW s0 0.04 -> 0.48420165 (old method missed it)
- PROBLEM 1, period check fails: measured dT/dln(df) is -16 to -20, theory -1/lambda_u is -190 to -512, and the
  periods (120 to 320) are shorter than a single e-folding time near the saddle (1/lambda_u = 190 to 512). So the
  cycle measured does not pass near the saddle, and it is not yet shown that these flips are homoclinic loops.
  New diagnostic step8f measures the minimum cycle-to-saddle distance as f -> f_hom and plots W^u on both sides.
- PROBLEM 2, spurious flips: for s0 >= 0.05 there are flips within 2e-4 to 6e-4 of f_T. The saddle there has
  y ~ 1e-3, below the absolute escape threshold (y < 1e-3), so W^u was misclassified. Threshold made relative
  (y < 1e-3 * y_saddle). The other extra flips (s0 = 0.05: 0.396-0.397; s0 = 0.065: 0.537-0.540; and those near f_H)
  must be rechecked after this fix. Do not quote any s0 >= 0.05 value yet.
- step8c rerun: 60 sets, 2 flagged (c = 44-45, large dilution). step8d recheck with Radau in (x, y) diverged
  (x -> -5e9), a solver failure, not a dynamics result. step8c fallback and step8d now integrate in log variables.

## 2026-11-05 | step8f verdict: the sign flip is a CANARD threshold, not a resolvable homoclinic loop
Case s0 = 0.02, delta = 0.05, c = 5, flip at f_c = 0.295370721 (results/step8f_s0_0.02_delta_0.05.json, png).
- As f -> f_c from above (df = 4e-5 down to 4e-10), the stable cycle's distance to the saddle falls only from 0.238 to
  0.164 (exponent 0.03, not 1), while its period grows logarithmically, dT/dln(df) = -18 (saddle theory: -512).
- The cycle is flat: y amplitude 0.005 to 0.019, x from about 0.40 to 0.58, hugging the top of the critical manifold
  y = (1-x)(s+x) (fold at x_h = (1-s)/2 = 0.484, y = 0.266). This is a canard cycle of the slow-fast system.
- Just below f_c the saddle's unstable manifold follows the repelling slow branch and then jumps LEFT: crabs crash to
  x ~ 0 on the fast time scale, raccoons starve slowly along x = 0, fall below the Allee threshold, and the system ends
  raccoon-free. Just above f_c it jumps RIGHT, back onto the attracting branch, and stays on the canard cycle.
- Interpretation: f_c is the maximal-canard (canard explosion) threshold. A true saddle connection must exist nearby
  (planar topology), but at a parameter exponentially close to f_c (extrapolated min distance -> 0 only near
  df ~ 1e-17), below double precision. The -18 slope is the repulsion time along the repelling slow branch.
- For the paper: report "canard-induced collapse at f_c" (numerical value, 9 digits) and the singular-limit picture;
  do not claim a numerically resolved homoclinic orbit. Biology: when fear is lowered, raccoon boom-bust cycles
  suddenly gain a crab-crash phase, and that crash takes raccoons below their fear-induced Allee threshold.

## 2026-11-05 | step8e rerun with relative threshold, and step8c rerun
- step8c with log-variable fallback: 60 parameter sets x 5 starts, 0 failures. Global convergence test clean.
- s0 = 0.08, 0.1: no flip (cycle survives down to f_T). The earlier near-f_T flips are gone, as intended.
- Main flips unchanged (s0 = 0.02, 0.03, 0.04; both delta), now read as canard thresholds f_c.
- New flips 2e-4 to 4e-4 above f_T for s0 = 0.02 to 0.04 (e.g. 0.169278 vs f_T = 0.168953): artefact. E1's
  attracting eigenvalue delta[1/((1+s)(1+f)) - mu] -> 0 at f_T, so an escaping orbit's y decays too slowly to pass a
  y-only threshold. Escape test now also accepts "x > 0.9 and y < half the saddle's y". Rerun needed.
- s0 = 0.05 and 0.065 still show a second flip mid-window (0.396-0.397; 0.537-0.540). Unexplained; check with
  step8f at s0 = 0.05 before using these cases.

## 2026-11-05 | Decision: stop the homoclinic/canard numerics
- Last workstation run returned output identical to the previous one (same near-f_T flips, step8f run for s0 = 0.02 only),
  so the latest fix was probably not in the folder that ran. Not chased further.
- Decision: the canard/homoclinic side result does not carry the paper. Keep ONE remark: canard-induced collapse at
  f_c = 0.295370721 (s0 = 0.02, delta = 0.05), with the step8f evidence. Drop s0 >= 0.05 homoclinic cases entirely.
- The paper rests on results that are already verified: equilibrium-count theorem, c* theorem (two derivations),
  general elasticity criterion (4 dilution forms), supercritical Hopf (509/509), closed-form BT curve, global
  convergence (60/60 sets, 300/300 starts), slow-fast link to the 28-day data, 3-D extension. Next: write.

## 2026-11-05 | Items 1 and 2: proofs and calibration robustness (step12a/b/c, step13; run on workstation)
Checked here before hand-over (small symbolic parts only):
- Hopf, c = 0, PROVED: at the RM Hopf point x* = (1-s)/2 the Guckenheimer-Holmes coefficient is
  a = -(1-s)^2 / (2(1+s)^3) < 0 for all 0 < s < 1, independent of delta. Supercritical everywhere.
  For c > 0: continuity in c gives supercriticality for small c; step12a part B is the grid evidence
  (uncalibrated mu: x* = 0.02 to 0.3; delta 0.01 to 3; c 0 to 100).
- Global stability, c = 0, PROVED for x* >= 1 - s: Lyapunov function
  V = int (g - g*)/g dx + (1/delta')(y - y* - y* ln(y/y*)),  dV/dt = -s (x-x*)^2 (x + x* + s - 1)/((s+x)(s+x*)) <= 0.
  For (1-s)/2 <= x* < 1-s cite RM limit-cycle uniqueness results (Cheng 1981; Kuang & Freedman 1988; verify refs).
  For c > 0: Dulac B = x^a y^b search (step12b part B), certificates by regime.
- BT nondegeneracy: Kuznetsov normal-form coefficients; test point s0 = 0.05, c = 5: f_BT = 0.64965,
  delta_BT = 4.640, a = +0.606, b = -1.633 (a b != 0, nondegenerate). Normalisation errors < 2e-16.
  Transversality in (f, delta): fold regular in f, d trJ/d delta = j22 > 0.
- step13: sup over s0 of c* across the calibration box x* in [0.02, 0.4], foraging kept in [0.15, 0.8].
  Identity: c* f_T/(1+f_T) = s_T/(1+s_T)^2 <= 1/4, so c f_T/(1+f_T) > 1/4 guarantees hysteresis for any calibration.

## 2026-11-05 | Items 1-2 results (workstation): ALL POSITIVE
- step12a Hopf: 1863 Hopf points over x* 0.02-0.3 (uncalibrated), delta 0.01-3, c 0-100, s0 0.01-0.95:
  l1 < 0 in all 1863 (max l1 = -0.030). Plus the c = 0 proof. Hopf supercritical throughout; no Bautin point.
- step12b Dulac (B = x^a y^b): raccoon-free 149/149 certified; coexistence 74/81; "bistable" 60/70.
  The 10 bistable "failures" all have NO stable interior equilibrium (upper state an unstable focus, so a cycle can
  exist): mislabelled regime, Dulac must fail there; not counterexamples. The 7 coexistence failures are weakly damped
  foci next to the Hopf curve (Re lambda = -0.005 to -0.025, one ~0), where x^a y^b is too weak; the direct global
  convergence test (60/60 sets, 300/300 starts) covers them. Net: no periodic orbit where the theory says none.
- step12c BT: 36/36 points nondegenerate (a b != 0). New: a and b each change sign along the BT curve, so
  codimension-3 degenerate BT points exist: a = 0 near s0 ~ 0.009-0.015 (c = 5, 20); b = 0 near s0 ~ 1.2-2 (c = 2),
  0.4-0.67 (c = 5), 0.13-0.23 (c = 20). Fast-mesopredator regime only (delta_BT >= 4). Remark-level result.
- step13 calibration: Suraci calibration sup over s0 of c* = 0.481 (at s0 = 0.255). Across the box (x* 0.02-0.4,
  foraging kept 0.15-0.8) sup c* ranges 0.28 to 1.88. So c > 1.88 gives hysteresis for every calibration and every s0;
  the universal form is c f_T/(1+f_T) > 1/4 (identity).
Decision: proofs and robustness complete. Next: manuscript.

## 2026-11-05 | Publication figures (make_figures.py; figures/fig1..fig8; shared qtds_fear/figtools.py)
Text-light by default (labels added in Illustrator); PDFs keep text editable (Type 42); dense colour fields are
embedded images inside the vector PDF. Each module checked at draft resolution here (QTDS_DRAFT=1); full resolution
to be rendered on a workstation.
- Fig3 phase portraits (s0 = 1, c = 5, delta = 0.3 for visible flow): basins, speed-shaded flow lines, nullclines,
  saddle stable manifold, trajectories; panel d: tipping boundaries across the window coloured by fear (the raccoon
  basin shrinks and saddle and node approach each other as f -> f_F).
- Fig4 in-silico playback (delta = 0.03): slow ramp up/down shows DYNAMIC hysteresis much wider than the static
  window: collapse only near f ~ 2.7-3.2 (past f_F = 2.49) and recovery only near f ~ 0.25 (far below f_T = 1.37),
  i.e. delayed loss of stability (slow passage through the transcritical). Worth a remark in the paper.
  Panel d: background fear f_b + temporary boost to f = 3.2 for D generations: inside [f_T, f_F] a boost of tens to
  ~100+ generations removes raccoons permanently; with no carnivores (f_b = 0, Suraci) any boost fades.
- Fig5 equilibrium surface over (f, c): folded stable/saddle sheets meeting the raccoon-free floor at (c*, f_T).
- Fig6 canard: cycle family on the critical manifold, explosion of crab amplitude as f -> f_c, time series either side.
- Fig7 regime atlas (s0, f) at c = 5, delta = 0.05 with f_T, f_F, f_H, canard points; BT curves with codim-3 points.
- Fig8 robustness: calibration heat map, elasticity criterion on identity line, 3-D carnivore hysteresis loops.

## 2026-11-05 | Manuscript draft (manuscript/main.tex, refs.bib; compiles to 15 pages)
- Structure: Introduction; Model (emergent Allee effect, calibration); Well-posedness; Theorem (count of equilibria,
  Descartes proof); Theorem c* (Sotomayor + slope) with corollary on the fold; Theorem elasticity criterion;
  Hopf/global for c = 0 (Lemma RM, Theorem supercritical a = -(1-s)^2/(2(1+s)^3), global stability via Hsu 1978
  Thm 3.3 because the prey nullcline is concave, Cheng 1981 for cycle uniqueness); numerical evidence for c > 0;
  Theorem closed-form BT curve; slow-fast and canard remark; numerical illustration; living-carnivore extension;
  Discussion; figure captions.
- Correction made while writing: playback level f = 1 is below f_T only for s0 >~ 0.25 (f_T < 1 for small s0).
- Open items listed in manuscript/NOTES_before_submission.md.

## 2026-11-05 | Figure restyle and full manuscript
- Figures: grey axes (#9a9993) offset outward, larger fonts (ticks 8.5 pt, labels 9.5 pt, panel letters 12 pt), no panel
  titles, in-plot notes off by default (QTDS_ANNOTATE=1 restores them; Fig1 keeps its schematic text).
- Manuscript expanded to 33 pages, 69 references (incl. Das & Barik 2023 iScience; Barik & Das 2023 STAR Protocols).
- New rigorous results added while writing:
  * Proposition (type of equilibria): P(u) = c D(u)^2 [y - h(X(y))] along the mesopredator nullcline, so
    sign det J = sign P'(u) at every root (checked at 10,544 equilibria, 100% agreement). Hence the unique state for
    Lambda < 0 has det > 0, and for Lambda > 0 the lower state is a saddle and the upper has det > 0.
  * det J = delta y g^2 Phi' (h' - p') (prey vs mesopredator nullcline slopes; checked at 10,505 equilibria).
  * Proposition: no interior equilibrium => E1 globally asymptotically stable (Poincare-Bendixson).
  * Elasticity criterion now proved by direct implicit differentiation (dy/df > 0 iff E > s_T/(1+s_T)^2).

## 2026-11-05 | Additional checks (analysis/step14_checks.py; draft run here, full run on workstation)
Draft-resolution results (to be confirmed by the full run):
- A, dynamic hysteresis: the recovery fear level of a slow down-ramp depends strongly on the immigration floor:
  f_recover = 0.75, 0.31, 0.14 for immigration 1e-5, 1e-7, 1e-9 (300-generation ramp; static f_T = 1.365).
  Collapse occurred only during the hold at f = 3.2 (beyond f_F = 2.49); a 100-generation ramp did not collapse at
  all. CONSEQUENCE: the manuscript's "recovery near f ~ 0.25" must be stated as depending on immigration and ramp
  speed (delayed loss of stability, delay ~ log(1/immigration)), not as a model constant.
- B, single window: 200/200 draft cases have exactly one bistable window right of f_T when backward, none when forward.
- C, stability of E+: sufficient condition (h'(x+) < 0 and delta < -h'/(y Phi')) never violated; E+ stable on the
  whole window in 61% of (s0, c) cells, on part of it in the rest (draft grid).
- D, BT coefficients: ARTEFACT FOUND AND FIXED. The kernel vector q0 from the SVD has an arbitrary sign; q0 -> -q0
  maps (a, b) -> (-a, -b). Earlier "a = 0 near s0 ~ 0.01" points (step12c, manuscript Sec. 7, Fig. 7b) were such
  simultaneous flips, not zeros. With a fixed orientation (q0[0] > 0) the only zeros are b = 0 at
  s0 = 1.444 (c = 2, delta_BT = 47.2), 0.574 (c = 5, 20.95), 0.158 (c = 20, 9.45). a has no zero on the grid.
  Rerun step12c and Fig 7; correct the manuscript.
- E, parity for general Phi (rational, exponential, squared, sigmoid): odd number of interior equilibria when the
  mesopredator invades, even otherwise, in all draft cases; at most 2 found for every form.

## 2026-11-05 — Manuscript revision pass (reviewer comments + step14 results)
- Abstract/intro rewritten: honest novelty positioning; proved vs numerical results separated; calibration called illustrative.
- Calibration section rewritten as explicit assumptions (q: Holling time budget; mu: calibration assumption; delta: order of magnitude, 97% is a treatment contrast). Table 1 has a "Status" column. Removed the circular "model reproduces 97%" claim.
- New Prop (stability of E+): tr J = g[h' + delta*y*Phi'] -> sufficient condition h'(x+)<0, delta<delta_+. Survey (step14 C): 0 violations, 501 cells; E+ stable on whole window for s0>0.16 (64% of cells).
- New Prop (parity, any continuous Phi): H(0)=-mu<0, H(1)=invasion quantity. Verified 3000/3000 for four Phi (step14 E).
- Thm cstar(b): hysteresis now requires E+ stable. Cor fold: f_F defined as supremum; boundedness proof; single window numerical (2836/2836 backward cases, step14 B).
- Elasticity theorem: existence hypotheses for f_T; local-only remark.
- BT: corrected. a never vanishes (|a|>=0.17); b=0 only at s0=1.444/0.574/0.158 for c=2/5/20 (step14 D, 360 points). New Prop: analyticity => zeros isolated. Earlier "a vanishes near s0~0.01" was an orientation artefact; removed.
- Slow-fast: x=0 branch (attracting y>s, repelling y<s, transcritical at (0,s)). Dynamic hysteresis: new Table 2 (immigration x ramp speed, step14 A). Canard labelled numerical.
- Extension moved to an appendix. Discussion/conclusions softened. Manuscript now 37 pp, compiles cleanly.

## 2026-11-05 (evening) — Second review, triage and fixes
Review was of the previous draft; points already handled: calibration reframing, BT nondegeneracy wording, Mathematical
Biosciences 1988 volume, ramp table. New in this pass:
- Abstract cut to about 245 words (QTDS limit 250). Status-of-results table (Table 1: P / N / C for every claim).
- NEW Prop. "Opening of the window" (proof by scaling c-c* = eps, f-f_T = eps^2, w = eps W + IFT):
  f_F - f_T = k^2 (c-c*)^2 / (4 a2 c* s_T Lambda'_T), k = s_T f_T/(1+f_T), a2 = s_T/(1+s_T)(2(1-mu) - s_T/(1+s_T)).
  At s0=1: coefficient 1.526; discriminant folds give 1.398, 1.481, 1.512 at c-c* = 0.03, 0.01, 0.003. Replaces kappa=1.98 fit.
  Key identity: a1(f_T) = -k (c - c*) exactly.
- The 0.6-1.2% gap in the step11 remark is the counting offset df = 1e-5 (text wrongly said 1e-3): predicted shift
  sqrt(df/C) = 0.00256, observed 0.00257 (s0=1, rational Phi).
- Prop E1gas: full proof (Poincare-Bendixson + Butler-McGehee; Lambda=0 case handled). Cites Smith & Waltman 1995.
- Thm count: remark on "multiplicity" (double root = fold) and case (c) via a1 = -k(c-c*).
- Elasticity theorem: explicit hypotheses (H1)-(H3), s_T defined, critical case excluded; how to measure E (paragraph).
- Hopf c>0: persistence near c=0 stated via IFT. BT delta>4 claim restricted to examined regimes. Canard: "numerical
  evidence", nine-digit f_c removed; "two independent methods" removed (fate-bisection was ill-conditioned).
- Discussion: generalist raccoons, habituation / time-varying f, density-dependent mortality vs fear Allee.
- Prop allee linked to the 2-D separatrix (stable manifold of E-).
- c stated as phenomenological. arXiv:2606.17586 verified (Antwi-Fordjour & Takyi 2026, preprint).
- analysis/step15_dynamics.py written (A ramp-rate scaling + entry-exit test; B calibration sensitivity of dynamic
  predictions; C critical boost duration D_c(f_b); D Hopf l1 on s0 up to 3; E delta_BT wide scan; F f_c accuracy).
  Draft-grid smoke test (development machine): A delay ~ duration^-0.5 (entry-exit sqrt law); C D_c grows ~ ln(1/(f_b-f_T));
  B all 3 test calibrations keep backward bifurcation, wide loop, finite lock-in, canard regime; E min delta_BT 4.05
  on a coarse grid. Full run pending.

## 2026-11-05 (night) — step15 results (workstation) folded into manuscript
A: no collapse for ramps <= 115 gen; entry-exit predicts exit within 0.05 (imm <= 1e-7); delay ~ N^-0.49 (1e-7), N^-0.40 (1e-9), N^-1.15 (1e-5, floor-dominated); loop/static window 1.03-2.71.
B: 27/27 calibrations: backward (c* 0.18-0.56), E+ stable mid-window, loop 1.8-5.5x window, recovery < f_T, D_c 16-183 gen, Hopf inside window at s0=0.02.
C: D_c 37 -> 317 gen as f_b - f_T: 1.07 -> 1e-3; ~45 ln(1/(f_b - f_T)).
D: 3240 sets, 2312 Hopf points, all l1 < 0 (max -0.030), none in (-1e-2, 0); no Hopf for s0 > 0.92.
E: 7844 BT points, min delta_BT 0.445 (s0=0.0045, c=16.9, x*=0.02, kept=0.6), 1% 0.77, 10% 4.03, median 35. Old ">4" claim was wrong in general; replaced.
F: f_c spread 4.6e-10 over 36 settings; rtol <= 1e-11 all give 0.2953707213714719. Reported 0.2953707212 +- 5e-10.
Refs: Sasmal-Takeuchi 2021 doi 10.1016/j.jmaa.2021.125485; Zanette-Clinchy 2019 doi 10.1016/j.cub.2019.02.042 (29(9):R309-R313).
Checklist: manuscript/REVIEW_CHECKLIST.md. Only author items remain.

## 2026-11-05 (late) — New Figure 9, exact asymptote in Fig 2e
- figures/fig9_dynamic_hysteresis.py (reads step15 A/B/C; panel a recomputes three loops, ~1 min): loops vs ramp speed; delay vs ramp duration with entry-exit prediction; D_c(f_b); dynamic vs static window over calibrations.
- qtds_fear.bifurcation.opening_coefficient(p); Fig 2e dashed lines = exact C (c-c*)^2 per s0 (was a hand-placed slope-2 guide).
- Figures copied into manuscript/figures so the LaTeX zip is self-contained. Table 1 gained two rows (dynamic hysteresis, D_c).
- 2026-11-05: Table 1 (status) removed; replaced by one sentence. run_all.py now runs Figures 1-9, uses 4 workers (QTDS_WORKERS), saves run_log after every job, supports --resume.
## 2026-11-06 | Final full run (workstation), audit
- 27/30 jobs OK. All result files reproduce the previous runs to < 1e-6 relative (only differences: eigenvalue noise at
  the BT double zero, run times, f_c spread 5.4e-10 vs 4.6e-10).
- Failed: step8 (MemoryError: LSODA stored every step over T = 60000; the fate-bisection homoclinic search is removed,
  superseded by step8e); step8c/8d failed only because they read step8.json. Fixed: end-point-only integration
  (t_eval=[T]) in log variables; step8c writes results/step8c_global_test.json; step8d reads it. To be rerun.
- New reproducible scripts for numbers that were computed in-session: step5c_type_check.py (sign det J = sign P'(u) at
  9 733 equilibria of 2e4 sets; lower state a saddle in all 1 191 two-equilibrium cases; manuscript updated from the
  earlier 10 544 / 3 356) and step6b_window_opening.py (C = 0.575, 1.526, 4.900 for s0 = 0.5, 1, 2; ratios converge).
- Every number in the manuscript checked against the final results; corrected: f_c spread (5.4e-10), recovery range
  (0.15, matching Table 2), type-check counts. Fig 4 caption completed (ramp durations, dashed lines).
- Figures: all equation-based values verified (Allee thresholds in Fig 1b; equilibria in Fig 3; c* point; canard
  diamonds; BT b = 0 points; 3-D windows). Layout fixes: Fig 6b inset ticks (f_c, f_H), Fig 8 panel spacing.
- 2026-11-06: step8c rerun (workstation): 60 parameter sets x 5 initial conditions = 300 trajectories, 0 failures; confirms the global-convergence sentence. All checks closed.
- 2026-11-06: Figure audit. Renumbered figures by order of discussion (1 concept, 2 phase portraits, 3 theorem map, 4 surface, 5 robustness, 6 regime atlas, 7 playback, 8 dynamic hysteresis, 9 canard); scripts renamed figN_*.py. Removed: green band in atlas (b) (now delta = 0.03 line and min delta_BT line), fills under curves in playback (a, b), seam lines in theorem map (a); trajectories drawn under nullclines in phase portraits; asymptote in theorem map (e) limited to c - c* <= 0.05; legends fixed. Every caption rewritten to name every line, shading and marker. Dey & Barik 2021 removed; Sasmal 2018 volume/pages and Panday 2019 DOI added; Beauchamp 2021 author list completed; GitHub link kept only in Code availability.
- 2026-11-09: Final full run (workstation, matplotlib 3.9.1): 32/32 jobs OK; all results identical to the previous run (differences only at machine precision; f_c spread 4.6e-10). Independent SymPy re-derivation of 21 manuscript formulas: all pass (scratch verify_theory.py). Hsu 1978 Thm 3.3 (concave isocline, Dulac) confirmed. Fixes: figure margin pad 0.08 (labels with subscripts were clipped in Fig 3d and 5c), Fig 3d label shortened, Fig 8b legend space (ylim top 5); text: f_c spread 'less than 6e-10'.
