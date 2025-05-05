import sympy as sp

k_syms = sp.symbols('k0:27')
k = sp.Matrix(k_syms)

rho = sp.symbols('ρ')
cs2 = sp.Rational(1, 3) 
omega, lambda_ = sp.symbols('ω λ')

k_eq = sp.zeros(27, 1)
k_eq[0]  = rho
k_eq[9]  = rho * cs2**2
k_eq[17] = rho * cs2**3 
k_eq[18] = rho * cs2**3 
k_eq[26] = rho * cs2**4 

print("Equilibrium Moments (k_eq):")
for i in range(27):
    if k_eq[i] != 0:
        print(f"k_eq[{i}] = {k_eq[i]}")
print("-" * 20)

diag_S = ([1] * 4) + ([omega] * 5) + ([lambda_] * 18)
S = sp.diag(*diag_S)

I = sp.eye(27)

k_star = (I - S) * k + S * k_eq

print("\nPost-Collision Central Moments (k*):")
k_star_simplified = []
for i in range(27):
    expr = sp.simplify(k_star[i])
    k_star_simplified.append(expr)
    print(f"k*[{i}] = {expr}")
