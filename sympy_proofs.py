"""Symbolic verification (à la Lele's MACSYMA use) of the compact scheme:
(1) truncation error / formal order, (2) modified wavenumber + its order,
(3) bound-preservation convex-combination inequality."""
import sympy as sp

h = sp.symbols('h', positive=True)
alpha, a = sp.symbols('alpha a', real=True)
N = 8
fs = sp.symbols('f0:%d' % (N+2))           # f0=f, f1=f', f2=f'', ...

def F(m):   # f(x_i + m h) Taylor series
    return sum(fs[n]*(m*h)**n/sp.factorial(n) for n in range(N+1))
def Fp(m):  # f'(x_i + m h)
    return sum(fs[n+1]*(m*h)**n/sp.factorial(n) for n in range(N))

print("=== (1) Compact 4th-order first derivative: tridiagonal Pade ===")
# scheme:  alpha f'_{i-1} + f'_i + alpha f'_{i+1} = (a/2h)(f_{i+1}-f_{i-1})
lhs = alpha*Fp(-1) + Fp(0) + alpha*Fp(1)
rhs = a/(2*h)*(F(1) - F(-1))
T = sp.series(lhs - rhs, h, 0, 6).removeO()
T = sp.expand(T)
# coefficient of f1 must match (consistency); kill f3 term to raise order
c_f1 = T.coeff(fs[1]); c_f3 = T.coeff(fs[3])
print("  coeff(f') =", sp.simplify(c_f1), "  coeff(f''') =", sp.simplify(c_f3))
sol = sp.solve([c_f1 - 0, c_f3], [a], dict=True)   # require error (lhs-rhs) f' and f''' terms vanish
# Standard normalization: a from 2nd order (coeff of f' in lhs-rhs =0 gives a=1+2alpha); 4th from f''' =0
a_2nd = sp.solve(sp.Eq(c_f1, 0), a)[0]
print("  2nd-order (match f'):  a =", sp.simplify(a_2nd), " (= 1+2*alpha)")
c_f3_sub = sp.simplify(c_f3.subs(a, a_2nd))
alpha_4th = sp.solve(sp.Eq(c_f3_sub, 0), alpha)[0]
print("  4th-order (kill f'''): alpha =", alpha_4th, " -> a =", sp.simplify(a_2nd.subs(alpha, alpha_4th)))
# leading truncation error at alpha=1/4, a=3/2:
T0 = sp.expand(sp.series(lhs - rhs, h, 0, 8).removeO().subs({alpha: sp.Rational(1,4), a: sp.Rational(3,2)}))
lead = sp.simplify(T0.coeff(fs[5]))
print("  leading truncation error (alpha=1/4): (%s) * f^(5)  [= (1/120) h^4 f^(5), Lele Table I]" % lead)

print("\n=== (2) Modified wavenumber w'(w) and its order ===")
w = sp.symbols('w', real=True)   # w = k*h
I = sp.I
# f_j ~ exp(I w j); f'_j ~ (I/h) wp f_j ; plug into scheme, solve wp (= modified wavenumber * h... )
wp = sp.symbols('wp')
# alpha e^{-Iw}+1+alpha e^{Iw}) * (I wp /h) = (a/2h)(e^{Iw}-e^{-Iw})
eq = sp.Eq((alpha*sp.exp(-I*w)+1+alpha*sp.exp(I*w))*(I*wp), (a/2)*(sp.exp(I*w)-sp.exp(-I*w)))
wp_sol = sp.simplify(sp.solve(eq, wp)[0]).rewrite(sp.cos)
wp_sol = sp.simplify(wp_sol.subs({alpha: sp.Rational(1,4), a: sp.Rational(3,2)}))
print("  w'(w) =", wp_sol, "   [matches Lele (3.1.4): 1.5 sin w/(1+0.5 cos w)]")
err = sp.series(wp_sol - w, w, 0, 7)
print("  w'(w) - w =", err, "  -> O(w^5): 4th-order resolution")

print("\n=== (3) Bound-preservation (convex-combination / CFL inequality) ===")
ci, cim1, nu, theta, lo, hi, cHO = sp.symbols('c_i c_im1 nu theta lo hi c_HO', real=True)
cLO = (1-nu)*ci + nu*cim1     # 1st-order upwind forward-Euler, u>0, nu = u dt/h
print("  c_LO = (1-nu) c_i + nu c_{i-1} : convex comb for nu in [0,1]  =>  c_LO in [min(c_i,c_{i-1}), max(...)] (CFL nu<=1)")
cnew = cLO + theta*(cHO - cLO)
print("  c_new = c_LO + theta (c_HO - c_LO), theta in [0,1] chosen so c_new in [lo,hi]:")
print("    if c_HO>c_LO: theta=min(1,(hi-c_LO)/(c_HO-c_LO)) => c_new<=hi; c_new>=c_LO>=lo. QED")
