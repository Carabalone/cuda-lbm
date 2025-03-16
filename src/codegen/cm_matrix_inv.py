import sympy as sp
from collections import Counter
# from common import expand_powers
from sympy.codegen.rewriting import create_expand_pow_optimization
import re

min_usage = 4
min_cost  = 1

expand = create_expand_pow_optimization(3)

def find(lst, obj):
    nl = [x[0] for x in lst]
    try:
        return lst[nl.index(obj)][1]
    except:
        print("didnt find it")
        return None
        
def get_priority_exprs():
    ux2 = ux * ux
    uy2 = uy * uy
    uxuy = ux * uy
    ux3 = ux * ux * ux
    uy3 = uy * uy * uy
    ux2uy = ux * ux * uy
    uxuy2 = ux * uy * uy
    ux2uy2 = ux * ux * uy * uy
    
    return [
        (ux2, 'ux2'),     
        (uy2, 'uy2'),     
        (uxuy, 'uxuy'),   
        (ux3, 'ux3'),     
        (uy3, 'uy3'),     
        (ux2uy, 'ux2uy'), 
        (uxuy2, 'uxuy2'), 
        (ux2uy2, 'ux2uy2')
    ]

def count_occurrences(subexpr, expr_list):
    return sum(1 for expr in expr_list for node in sp.preorder_traversal(expr) if node == subexpr)

def add_float_suffix(c_code):
    pattern = r'(\b\d*\.\d+|\b\d+\.\d*[eE][+-]?\d+)(?!\w*f\b)'
    return re.sub(pattern, r'\1f', c_code)

ux, uy = sp.symbols('ux uy')

cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
cy = [0, 0, 1, 0, -1, 1, 1, -1, -1]

M = sp.zeros(9, 9)

for j in range(9):
    cx_c = cx[j] - ux
    cy_c = cy[j] - uy
    
    M[0, j] = 1                     # ρ (density) - conserved
    M[1, j] = cx_c                # e (energy)
    M[2, j] = cy_c                # ε (energy squared)
    M[3, j] = cx_c**2 + cy_c**2 # jx (x-momentum) 
    M[4, j] = cx_c**2 - cy_c**2 # jy (y-momentum) 
    M[5, j] = cx_c * cy_c       # qx (x heat flux)
    M[6, j] = cx_c**2 * cy_c    # qy (y heat flux)
    M[7, j] = cx_c * cy_c**2    # pxx (xx stress)
    M[8, j] = cx_c**2 * cy_c**2 # pxy (xy stress)

M_inv = M.inv()

priority_exprs = get_priority_exprs()
priority_subs = {expr: sp.Symbol(name) for expr, name in priority_exprs}

expr_list = [sp.simplify(M_inv[i, j].subs(priority_subs)) for i in range(9) for j in range(9)]

replacements, simplified_exprs = sp.cse(expr_list, optimizations='basic')

filtered_replacements = []
for sym, subexpr in replacements:
    usage_count = count_occurrences(sym, simplified_exprs)
    if usage_count >= min_usage:
        filtered_replacements.append((sym, subexpr))

add = []
all_exprs = filtered_replacements + [(y,x) for x, y in priority_subs.items()] + [(ux, ux), (uy, uy)]
all_syms = [x[0] for x in all_exprs]

for sym, expr in filtered_replacements:
    for subexpr in sp.preorder_traversal(expr):
        if (isinstance(subexpr, sp.Symbol) and subexpr not in all_syms):
            actual_expr = find(replacements + [(y,x) for x, y in priority_subs.items()] + [(ux, ux), (uy, uy)], subexpr)
            add += [(subexpr, actual_expr)]
filtered_replacements += add
filtered_replacements.sort(key=lambda x: list(map(lambda y: y[0], replacements)).index(x[0]))

final_exprs = expr_list.copy()
for sym, subexpr in filtered_replacements:
    final_exprs = [expr.subs(subexpr, sym).evalf() for expr in final_exprs]


def gen_code():
    quadratures = 9
    print("// SymPy-Generated CUDA code for central moment matrix inverse")
    print("__device__ __forceinline__ static")
    print(f"void cm_matrix_inverse(float* M_inv, float ux, float uy) {{")

    # Priority subexpressions
    print("    // Priority subexpressions")
    for expr, name in priority_exprs:
        c_expr = add_float_suffix(sp.ccode(expand(expr)))
        print(f"    float {name} = {c_expr};")

    # Filtered common subexpressions
    print("\n    // Additional common subexpressions")
    for sym, subexpr in filtered_replacements:
        c_expr = add_float_suffix(sp.ccode(expand(subexpr)))
        print(f"    float {sym} = {c_expr};")

    # Matrix elements
    print("\n    // Matrix elements")
    for i in range(9):
        for j in range(9):
            idx = i * 9 + j
            c_expr = add_float_suffix(sp.ccode(expand(final_exprs[idx])))
            print(f"    M_inv[{i} * {quadratures} + {j}] = {c_expr};")
        print("")

    print("}")

gen_code()