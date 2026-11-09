# ----------------------------------------------------------------------------------------------------
# HOW TO RUN (from the repository folder, virtual environment active):
#     python analysis/verify_theory.py
# Output:  printed PASS/FAIL list (about 1 min)
# Used in the paper for:  independent symbolic check of the formulas in Propositions, Lemmas and Theorems
#   (Allee threshold, invasion eigenvalue, cubic and its coefficients, det/trace of J, c*, opening coefficient, Hopf l1 for c = 0,
#   Lyapunov derivative, fast eigenvalues of the slow-fast system)
# ----------------------------------------------------------------------------------------------------
"""Independent SymPy verification of the formulas stated in the paper. Every line should print PASS."""
import sympy as sp
x,y,s,c,f,mu,d,u,w,q,s0=sp.symbols('x y s c f mu delta u w q s0',positive=True)
ok=lambda name,expr: print(('PASS ' if sp.simplify(expr)==0 else 'FAIL ')+name)
g=x/(s+x); Phi=(1+c*y)/(1+c*y+f); h=(1-x)*(s+x)
F1=x*(1-x)-x*y/(s+x); F2=d*y*(g*Phi-mu)
J=sp.Matrix([F1,F2]).jacobian([x,y])
# 1 Allee threshold
yA=(mu*(1+f)-g)/(c*(g-mu)); ok('Prop allee: y_A solves g*Phi=mu', (g*Phi-mu).subs(y,yA))
# 2 E1 eigenvalue and Lambda
ev=J.subs({x:1,y:0}); lam=sp.simplify(ev[1,1]); Lam=mu*(1+f)*(1+s)-1
ok('E1 invasion eigenvalue = -delta*Lambda/((1+s)(1+f))', lam + d*Lam/((1+s)*(1+f)))
# 3 cubic and coefficients
D=(1-mu)*u-mu*f; P=(u-1)*D**2-c*s*u*(D-s*mu*(u+f))
X=s*mu*(u+f)/D; Y=(u-1)/c
ok('predator nullcline gives x=s mu(u+f)/D', (g*Phi-mu).subs({x:X,y:Y}).simplify())
ok('P(u)=c D^2 [y - h(X)] on predator nullcline', P - c*D**2*(Y-h.subs(x,X)))
A=mu*(1+s); B=mu*(1+f)
Pw=sp.expand(P.subs(u,1+w)); co=sp.Poly(Pw,w).all_coeffs()[::-1]
ok('a3=(1-mu)^2', co[3]-(1-mu)**2)
ok('a2=cs(A-1)+2(1-mu)(1-B)', co[2]-(c*s*(A-1)+2*(1-mu)*(1-B)))
ok('a1=cs(fA+2A-2)+(1-B)^2', co[1]-(c*s*(f*A+2*A-2)+(1-B)**2))
ok('a0=cs*Lambda', co[0]-c*s*Lam)
# 4 Lemma det / trace at equilibrium (use y=h(x), Phi = mu/g)
Phip=sp.diff(Phi,y)
JJ=J.subs(y,h)
tr_eq=(JJ[0,0]+JJ[1,1]).subs(mu, (g*Phi).subs(y,h))
ok('tr J = g h\'(x) + delta y g Phi\' at equilibrium', tr_eq - (g*sp.diff(h,x)+d*h*g*Phip.subs(y,h)))
# p'(x): predator nullcline y=p(x) defined by g(x)Phi(y)=mu -> p' = -g' Phi/(g Phi')
pp=-sp.diff(g,x)*Phi/(g*Phip)
det_eq=(JJ.det()).subs(mu,(g*Phi).subs(y,h))
ok('det J = delta y g^2 Phi\' (h\' - p\')', det_eq - (d*h*g**2*Phip*(sp.diff(h,x)-pp)).subs(y,h))
# 5 c* and elasticity for rational Phi
E=sp.simplify((sp.diff(Phi,y)/Phi).subs(y,0)); ok('E = c f/(1+f)', E - c*f/(1+f))
fT,sT=sp.symbols('f_T s_T',positive=True)
cstar=sT*(1+fT)/(fT*(1+sT)**2); ok('E = s_T/(1+s_T)^2 exactly at c = c*', (E.subs({f:fT})-sT/(1+sT)**2).subs(c,cstar))
# 6 a1 at Lambda=0 equals -k(c-c*)
muT=1/((1+fT)*(1+sT))
a1T=co[1].subs({s:sT,f:fT,mu:muT}); k=sT*fT/(1+fT)
ok('a1(f_T) = -k (c - c*)', a1T + k*(c-cstar))
a2T=co[2].subs({s:sT,f:fT,mu:muT,c:cstar}); al2=sT/(1+sT)*(2*(1-muT)-sT/(1+sT))
ok('a2(f_T,c*) = alpha_2', a2T-al2)
# 7 c=0 Hopf: x*=(1-s)/2 and first Lyapunov coefficient (Guckenheimer-Holmes normalisation) for RM
r=sp.symbols('r',positive=True)  # r = delta/(1+f) effective time scale, M = mu(1+f)
xs=(1-s)/2; ys=h.subs(x,xs)
G1=x*(1-x)-x*y/(s+x); G2=r*y*(x/(s+x)-xs/(s+xs))
JJ0=sp.Matrix([G1,G2]).jacobian([x,y]).subs({x:xs,y:ys})
ok('c=0: trace vanishes at x*=(1-s)/2', JJ0.trace())
# l1 via standard formula for planar system (GH 3.4.11) after shifting
om=sp.sqrt(JJ0.det())
X1,Y1=sp.symbols('X1 Y1')
Fs=[G1.subs({x:xs+X1,y:ys+Y1}),G2.subs({x:xs+X1,y:ys+Y1})]
# transform to normal form coordinates: columns (J12,J22)^T and (0,-om)^T  as in the paper
T=sp.Matrix([[JJ0[0,1],0],[JJ0[1,1],-om]]); Ti=T.inv()
U,V=sp.symbols('U V')
XY=T*sp.Matrix([U,V])
FF=Ti*sp.Matrix([fi.subs({X1:XY[0],Y1:XY[1]}) for fi in Fs])
fu,gv=FF[0],FF[1]
dd=lambda e,*v: sp.diff(e,*v).subs({U:0,V:0})
a_=(dd(fu,U,U,U)+dd(fu,U,V,V)+dd(gv,U,U,V)+dd(gv,V,V,V))/16 + (dd(fu,U,V)*(dd(fu,U,U)+dd(fu,V,V)) - dd(gv,U,V)*(dd(gv,U,U)+dd(gv,V,V)) - dd(fu,U,U)*dd(gv,U,U) + dd(fu,V,V)*dd(gv,V,V))/(16*om)
aval=sp.simplify(a_)
print('  l1 (GH) =', sp.factor(aval))
ok('l1 = -(1-s)^2/(2(1+s)^3) (independent of r)', aval + (1-s)**2/(2*(1+s)**3))
# 8 Lyapunov derivative for c=0 (Theorem global0)
xst=sp.symbols('xst',positive=True)
Vdot=(g-g.subs(x,xst))*(h-h.subs(x,xst))
ok('Vdot factorisation', Vdot + s*(x-xst)**2*(x+xst+s-1)/((s+x)*(s+xst)))
# 9 slow-fast: L0 eigenvalue and C0 eigenvalue
ok('fast eigenvalue on x=0 is 1 - y/s', sp.diff(F1,x).subs(x,0)-(1-y/s))
ok('fast eigenvalue on C0 is g h\'', sp.diff(F1,x).subs(y,h)-g*sp.diff(h,x))
# 10 Lambda'(f)
sf=s0*(1+q*f); Lf=mu*(1+f)*(1+sf)-1
ok("Lambda'_T = mu(1+s_T)+mu(1+f_T)s'", sp.diff(Lf,f)-(mu*(1+sf)+mu*(1+f)*s0*q))
# 11 calibration numbers
print('  q =', 1/0.34-1, '; delta from 0.5/yr and 0.045/day =', (0.5/365)/0.045)
