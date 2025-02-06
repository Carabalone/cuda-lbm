import sympy as sp

N = 9
f_eq    = sp.IndexedBase('f_eq', N)
WEIGHTS = sp.IndexedBase('WEIGHTS', N)
C       = sp.IndexedBase('C', N)

rho, ux, uy = sp.symbols('rho ux uy')
i = sp.symbols('i', cls=sp.Idx)

assignments = []

for j in range(N):
    # f_eq[i] = WEIGHTS[i] * rho * (1 + (C[2*i] * ux + C[2*i+1] * uy))
    expr = sp.Eq(f_eq[j],
                 WEIGHTS[j] * rho *
                 (1 + (C[2*j] * ux + C[2*j+1] * uy)))

    # print(sp.pretty(expr))
    
    code_line = sp.ccode(expr.rhs, assign_to=f"f_eq[{i}]", contract=False)
    code_line = code_line.replace("*", " * ").replace("+", " + ").replace("=", " = ")
    assignments.append(code_line)

unrolled_code = "\n".join(assignments)
print("Generated unrolled assignments:\n")
print(unrolled_code)
