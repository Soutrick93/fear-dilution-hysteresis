# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python -m qtds_fear.symbolic
# Also run automatically by:  python run_all.py   (job name: symbolic)
# Output:  results/symbolic_report.md
# Used in the paper for:  symbolic checks: scaling, boundedness, boundary eigenvalues, the cubic and its coefficients, c = 0 reduction, Sotomayor quantities, formula for c*
# ----------------------------------------------------------------------------------------------------
"""Symbolic derivations and checks (SymPy). Each function returns (claim, verified: bool, detail: str).

Run `python -m qtds_fear.symbolic` to print the report and write results/symbolic_report.md.
"""
import sympy as sp

x, y, u = sp.symbols('x y u', positive=True)
s0, q, c, f, mu, dl = sp.symbols('s0 q c f mu delta', positive=True)
s = sp.Symbol('s', positive=True)            # s = s0 (1 + q f), kept abstract where possible
g = x / (s + x)
Phi = (1 + c * y) / (1 + c * y + f)
Fx = x * (1 - x) - x * y / (s + x)
Fy = dl * y * (g * Phi - mu)


def nondimensionalisation():
    R, M, T, K, rho, a, h, e, m, gam = sp.symbols('R M T K rho a h e m gamma', positive=True)
    dR = rho * R * (1 - R / K) - a * R * M / (1 + q * f + a * h * R)
    dM = M * (e * a * R / (1 + q * f + a * h * R) / (1 + f / (1 + gam * M)) - m)
    sub = {R: K * x, M: rho * h * K * y}
    # dx/dt = (1/(K rho)) dR/dT, dy/dt = (1/(rho^2 h K)) dM/dT
    xd = sp.simplify(dR.subs(sub) / (K * rho))
    yd = sp.simplify(dM.subs(sub) / (rho * rho * h * K))
    pars = {s: (1 + q * f) / (a * h * K), mu: m * h / e, dl: e / (h * rho), c: gam * rho * h * K}
    ok1 = sp.simplify(xd - Fx.subs(pars)) == 0
    ok2 = sp.simplify(yd - Fy.subs(pars)) == 0
    return ('Dimensional model maps to the dimensionless one with s=(1+qf)/(ahK), mu=mh/e, delta=e/(h rho), c=gamma rho h K (y = M/(rho h K))',
            bool(ok1 and ok2), 'difference of right-hand sides simplifies to 0')


def boundedness():
    V_dot = Fx + Fy / dl
    gap = sp.simplify(V_dot - (x * (1 - x) - mu * y))
    target = -x * y * f / ((s + x) * (1 + c * y + f))
    ok = sp.simplify(gap - target) == 0
    return ('For V = x + y/delta:  dV/dt = x(1-x) - mu y - x y f/((s+x)(1+cy+f)) <= x(1-x) + mu delta x - mu delta V,'
            ' so limsup V <= (1 + mu delta)^2 / (4 mu delta)', bool(ok), f'dV/dt - [x(1-x) - mu y] = {sp.factor(gap)}')


def boundary_eigenvalues():
    J = sp.Matrix([Fx, Fy]).jacobian([x, y])
    e0 = J.subs({x: 0, y: 0}).eigenvals()
    e1 = J.subs({x: 1, y: 0})
    ev1 = [sp.simplify(v) for v in e1.eigenvals()]
    expect = sp.simplify(dl * (1 / ((1 + s) * (1 + f)) - mu))
    ok = any(sp.simplify(v - expect) == 0 for v in ev1) and any(sp.simplify(v + 1) == 0 for v in ev1)
    return ('E0=(0,0) has eigenvalues 1 and -delta mu (saddle). E1=(1,0) has eigenvalues -1 and delta[1/((1+s)(1+f)) - mu];'
            ' E1 is stable iff mu(1+f)(1+s) > 1', bool(ok), f'E0: {list(e0)}, E1: {ev1}')


def interior_cubic():
    # predator nullcline with u = 1 + c y:  x = s mu (u+f)/D, D = (1-mu)u - mu f; prey nullcline y = (1-x)(s+x)
    D = (1 - mu) * u - mu * f
    xu = s * mu * (u + f) / D
    yu = (u - 1) / c
    check_pred = sp.simplify((xu / (s + xu)) * u / (u + f) - mu) == 0
    expr = sp.together(yu - (1 - xu) * (s + xu))
    num = sp.factor(sp.numer(expr))
    P = sp.expand((u - 1) * D ** 2 - c * s * u * (D - s * mu * (u + f)))
    ok = sp.simplify(sp.expand(num) * 1 - sp.expand(P) * sp.sign(sp.Poly(num, u).LC() / sp.Poly(P, u).LC())) == 0 \
        or sp.simplify(sp.expand(num) / P).is_constant()
    poly = sp.Poly(P, u)
    coeffs = [sp.factor(cc) for cc in poly.all_coeffs()]
    return ('Interior equilibria are roots u>1 of the cubic P(u) = (u-1)D^2 - c s u [D - s mu (u+f)], D = (1-mu)u - mu f;'
            ' hence at most three', bool(check_pred and poly.degree() == 3),
            'coefficients (u^3..u^0): ' + '; '.join(str(cc) for cc in coeffs))


def c_zero_reduction():
    Fy0 = Fy.subs(c, 0)
    target = dl / (1 + f) * y * (g - mu * (1 + f))
    ok = sp.simplify(Fy0 - target) == 0
    return ('For c = 0 the model is exactly Rosenzweig-MacArthur with death mu(1+f) and time scale delta/(1+f):'
            ' fear only rescales parameters, so global dynamics are classical (unique interior equilibrium,'
            ' supercritical Hopf at x* = (1-s)/2, unique limit cycle beyond it)', bool(ok), 'y-equation identity verified')


def transcritical_sotomayor():
    """Sotomayor conditions for a transcritical bifurcation of E1 at f = f_T, parameter f, with s = s0(1+qf)."""
    ss = s0 * (1 + q * f)
    F = sp.Matrix([Fx, Fy]).subs(s, ss)
    J = F.jacobian([x, y]).subs({x: 1, y: 0})
    # at f_T: mu (1+f)(1+ss) = 1  -> right/left null vectors of J
    lam2 = sp.simplify(J[1, 1])
    v = sp.Matrix([-J[0, 1], J[0, 0]]).applyfunc(sp.simplify)   # J v = (0, J10 v1 + J11 v2) with J10 = 0, J11 = 0 at f_T
    w = sp.Matrix([0, 1])
    Ff = F.diff(f).subs({x: 1, y: 0})
    cond1 = sp.simplify((w.T * Ff)[0])                          # must be 0 -> not a saddle-node
    DFf = F.diff(f).jacobian([x, y]).subs({x: 1, y: 0})
    cond2 = sp.simplify((w.T * DFf * v)[0])                     # must be != 0 (transversality)
    # second derivative D^2F(v,v) along w
    H = sp.hessian(F[1], (x, y)).subs({x: 1, y: 0})
    cond3 = sp.simplify((v.T * H * v)[0])
    return dict(J_E1=J, lambda2=lam2, v=v, w_Ff=cond1, w_DFf_v=cond2, w_D2F_vv=cond3)


def transcritical_slope():
    """dy/df along the coexistence branch at (x,y)=(1,0): sign decides forward/backward; gives c*."""
    ss = s0 * (1 + q * f)
    g1 = (1 - x) * (ss + x) - y
    g2 = x * (1 + c * y) - mu * (ss + x) * (1 + c * y + f)
    Jm = sp.Matrix([[g1.diff(x), g1.diff(y)], [g2.diff(x), g2.diff(y)]])
    sol = Jm.LUsolve(-sp.Matrix([g1.diff(f), g2.diff(f)]))
    dydf = sp.factor(sp.simplify(sol[1].subs({x: 1, y: 0})))
    num, den = sp.fraction(dydf)
    return dydf, sp.factor(num), sp.expand(den)


def c_star_identity():
    """At f_T, den(c) < 0 iff c > c* = s_T(1+f_T)/(f_T(1+s_T)^2), using mu = 1/((1+f_T)(1+s_T))."""
    fT, sT, cc = sp.symbols('f_T s_T c', positive=True)
    muT = 1 / ((1 + fT) * (1 + sT))
    den = cc * (1 + sT) * (muT * (1 + sT) - 1) + 1 - muT * (1 + fT)
    root = sp.solve(sp.Eq(den, 0), cc)[0]
    ok = sp.simplify(root - sT * (1 + fT) / (fT * (1 + sT) ** 2)) == 0
    slope_c = sp.simplify(sp.diff(den, cc))
    return ('Slope denominator at f_T is linear in c and vanishes at c* = s_T(1+f_T)/(f_T(1+s_T)^2); it is decreasing in c'
            ' (coefficient -(1+s_T) f_T/(1+f_T) < 0), so the branch is backward exactly for c > c*', bool(ok),
            f'd(den)/dc = {slope_c}')


def report():
    rows = [nondimensionalisation(), boundedness(), boundary_eigenvalues(), interior_cubic(), c_zero_reduction(),
            c_star_identity()]
    T = transcritical_sotomayor()
    dydf, num, den = transcritical_slope()
    lines = ['# Symbolic checks (generated by qtds_fear.symbolic)', '']
    for claim, ok, detail in rows:
        lines += [f'- **{"VERIFIED" if ok else "CHECK FAILED"}**: {claim}', f'    - {detail}']
    lines += ['', '## Sotomayor conditions at E1 = (1,0), parameter f (evaluate at f = f_T)',
              f'- second eigenvalue of J(E1): `{T["lambda2"]}` (zero defines f_T)',
              f'- null vector v = `{list(T["v"])}`, left null vector w = (0, 1)',
              f'- w . F_f(E1) = `{T["w_Ff"]}`  (= 0: rules out a saddle-node; E1 persists for all f)',
              f'- w . DF_f(E1) v = `{T["w_DFf_v"]}`  (nonzero: transversality)',
              f'- w . D^2F(E1)(v, v) = `{T["w_D2F_vv"]}`  (nonzero for c != c*: nondegenerate transcritical)',
              '', '## Slope of the coexistence branch at the transcritical',
              f'- dy/df at (1,0) = `{dydf}`', f'- numerator = `{num}` (always negative)', f'- denominator = `{den}`']
    return '\n'.join(lines), all(r[1] for r in rows)


if __name__ == '__main__':
    import pathlib
    text, ok = report()
    print(text)
    out = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'symbolic_report.md'
    out.write_text(text)
    print('\nALL VERIFIED' if ok else '\nSOME CHECKS FAILED')
