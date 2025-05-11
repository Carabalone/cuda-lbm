import sympy as sp
from sympy.codegen.rewriting import create_expand_pow_optimization
import re
import time
import numpy as np
import os

min_usage_cse = 16
expand_powers = create_expand_pow_optimization(3) 
output_dir = "generated_collision_code"
os.makedirs(output_dir, exist_ok=True)
inv_test = False # test T * T_inv * (k + collision + forcing) to check against De Rosis universal formulations.

ux, uy, uz = sp.symbols('ux uy uz')
rho = sp.symbols('rho')
omega, lambda_ = sp.symbols('omega lambda_')
k_syms = sp.symbols(f'k[0:{27}]')
k = sp.Matrix(k_syms)
cs2 = sp.Rational(1, 3)
Fx, Fy, Fz = sp.symbols('Fx Fy Fz')

c_ix = np.array([0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1], dtype=int)
c_iy = np.array([0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1], dtype=int)
c_iz = np.array([0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1], dtype=int)

k_eq = sp.zeros(27, 1)
k_eq[0]  = rho
k_eq[9]  = 3 * rho * cs2
k_eq[17] = rho * cs2
k_eq[18] = rho * cs2**2
k_eq[26] = rho * cs2**3

R = sp.zeros(27, 1)
R[1]  = Fx
R[2]  = Fy
R[3]  = Fz
R[10] = 2 * Fx * cs2
R[11] = 2 * Fy * cs2
R[12] = 2 * Fz * cs2
R[23] = Fx * cs2**2
R[24] = Fy * cs2**2
R[25] = Fz * cs2**2

S_ACM = sp.diag(*(([1] * 4) + ([omega] * 5) + ([lambda_] * (27 - 9))))
# CM: lambda_ = 1
S_CM = sp.diag(*(([1] * 4) + ([omega] * 5) + ([1] * (27 - 9))))
collision_ACM = -S_ACM * (k - k_eq)
collision_CM  = -S_CM  * (k - k_eq)

print("Constructing Central Moment Matrix T...")
start_time = time.time()
T = sp.zeros(27, 27)
for j in range(27):
    cx, cy, cz = c_ix[j] - ux, c_iy[j] - uy, c_iz[j] - uz
    cx2, cy2, cz2 = cx**2, cy**2, cz**2
    # eq 11
    T[0, j]  = 1
    T[1, j]  = cx
    T[2, j]  = cy
    T[3, j]  = cz
    T[4, j]  = cx * cy
    T[5, j]  = cx * cz
    T[6, j]  = cy * cz
    T[7, j]  = cx2 - cy2
    T[8, j]  = cx2 - cz2
    T[9, j]  = cx2 + cy2 + cz2
    T[10, j] = cx * cy2 + cx * cz2
    T[11, j] = cx2 * cy + cy * cz2
    T[12, j] = cx2 * cz + cy2 * cz
    T[13, j] = cx * cy2 - cx * cz2
    T[14, j] = cx2 * cy - cy * cz2
    T[15, j] = cx2 * cz - cy2 * cz
    T[16, j] = cx * cy * cz
    T[17, j] = cx2 * cy2 + cx2 * cz2 + cy2 * cz2
    T[18, j] = cx2 * cy2 + cx2 * cz2 - cy2 * cz2
    T[19, j] = cx2 * cy2 - cx2 * cz2
    T[20, j] = cx2 * cy * cz
    T[21, j] = cx * cy2 * cz
    T[22, j] = cx * cy * cz2
    T[23, j] = cx * cy2 * cz2
    T[24, j] = cx2 * cy * cz2
    T[25, j] = cx2 * cy2 * cz
    T[26, j] = cx2 * cy2 * cz2
print(f"Matrix T constructed in {time.time() - start_time:.2f} seconds.")

print("Inverting T ")
T_inv = T.inv()
print(f"Matrix T inverted")

I = sp.eye(27)
forcing_term_ACM = (I - S_ACM / 2) * R
forcing_term_CM  = (I - S_CM  / 2) * R

start_time = time.time()
f_post_acm = T_inv * (k + collision_ACM + forcing_term_ACM)
print(f"ACM formulations calculated in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
f_post_cm = T_inv * (k + collision_CM + forcing_term_CM)
if (inv_test):
    test = T * f_post_acm
    test_2 = T * f_post_cm
    for i in range(27):
        print(f"ACM_{i}: {sp.simplify(test[i])}")
    for i in range(27):
        print(f"CM_{i}: {sp.simplify(test_2[i])}")
    exit(0)
print(f"CM formulations calculated in {time.time() - start_time:.2f} seconds.")

def generate_collision_code(Omega_matrix, func_name, filename, input_symbols, is_acm=True):
    print(f"\n--- Generating code for {func_name} ---")
    print("Performing CSE...")
    start_time = time.time()

    def get_priority_exprs_3d():
        ux2, uy2, uz2 = ux*ux, uy*uy, uz*uz
        uxuy, uxuz, uyuz = ux*uy, ux*uz, uy*uz
        ux2uy, uxuy2 = ux2*uy, ux*uy2
        ux2uz, uxuz2 = ux2*uz, ux*uz2
        uy2uz, uyuz2 = uy2*uz, uy*uz2
        uxuyuz = ux*uy*uz
        ux2uy2 = ux2*uy2
        ux2uz2 = ux2*uz2
        uy2uz2 = uy2*uz2
        ux2uyuz = ux2*uyuz
        uy2uxuz = uy2*uxuz
        uz2uxuy = uz2*uxuy
        ux2uy2uz = ux2uy2*uz
        ux2uz2uy = ux2uz2*uy
        uy2uz2ux = uy2uz2*ux

        return [
            (ux2, 'ux2'), (uy2, 'uy2'), (uz2, 'uz2'),
            (uxuy, 'uxuy'), (uxuz, 'uxuz'), (uyuz, 'uyuz'),
            (ux2uy, 'ux2uy'), (uxuy2, 'uxuy2'),
            (ux2uz, 'ux2uz'), (uxuz2, 'uxuz2'),
            (uy2uz, 'uy2uz'), (uyuz2, 'uyuz2'),
            (uxuyuz, 'uxuyuz'),
            (ux2uy2, 'ux2uy2'), (ux2uz2, 'ux2uz2'), (uy2uz2, 'uy2uz2'),
            (ux2uyuz, 'ux2uyuz'), (uy2uxuz, 'uy2uxuz'), (uz2uxuy, 'uz2uxuy'),
            (ux2uy2uz, 'ux2uy2uz'), (ux2uz2uy, 'ux2uz2uy'), (uy2uz2ux, 'uy2uz2ux')
        ]

    priority_exprs = get_priority_exprs_3d()
    priority_subs = {expr: sp.Symbol(name) for expr, name in priority_exprs}
    priority_subs_sorted = sorted(priority_subs.items(), key=lambda item: len(str(item[0])), reverse=True)

    expr_list_flat = []
    print("Applying priority substitutions to Omega...")
    for i in range(27):
        expr = Omega_matrix[i]
        for p_expr, p_sym in priority_subs_sorted:
             expr = expr.subs(p_expr, p_sym)
        expr_list_flat.append(sp.simplify(expr))
        print(f"Processed Ω[{i}]", end='\r')
    print("\nPriority substitutions applied.")

    print("Running SymPy CSE on Omega expressions...")
    replacements, simplified_exprs = sp.cse(expr_list_flat, optimizations='basic')
    print(f"Initial CSE found {len(replacements)} replacements.")

    print("Filtering CSE results...")
    filtered_replacements_dict = {}
    usage_counts = {}
    for sym, subexpr in replacements:
        usage_counts[sym] = sum(e.count(sym) for e in simplified_exprs) # Count in final expressions
        for _, other_expr in replacements:
             usage_counts[sym] += other_expr.count(sym)

    for sym, subexpr in replacements:
         if usage_counts.get(sym, 0) >= min_usage_cse:
             filtered_replacements_dict[sym] = subexpr

    filtered_replacements_list = sorted(
        filtered_replacements_dict.items(),
        key=lambda item: [r[0] for r in replacements].index(item[0])
    )

    final_exprs_flat = expr_list_flat[:]
    for sym, _ in filtered_replacements_list:
         for i in range(len(final_exprs_flat)):
             final_exprs_flat[i] = final_exprs_flat[i].subs(sym, sym)

    print(f"CSE finished in {time.time() - start_time:.2f} seconds.")
    print(f"Kept {len(filtered_replacements_list)} subexpressions after filtering.")

    print("Generating C++ code...")
    def apply_replacements(part):
        pattern = r'(\b\d+\.\d*|\.\d+)([eE][+-]?\d+)?(?![f])'
        part = re.sub(pattern, r'\1\2f', part)
        part = part.replace('1.0/3.0', '(1.0f/3.0f)')
        part = part.replace('1/3', '(1.0f/3.0f)')
        part = re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'\1.0f', part)
        part = re.sub(r'\bomega\b', 'omega', part)
        part = re.sub(r'\blambda_\b', 'lambda', part)
        return part

    def add_float_suffix(c_code):
        parts = re.split(r'(\[.*?\])', c_code)
        modified_parts = [apply_replacements(part) if i % 2 == 0 else part for i, part in enumerate(parts)]
        modified_code = ''.join(modified_parts)
        return modified_code

    def format_ccode(expr):
        try:
            c_expr = sp.ccode(expand_powers(expr), standard='C99')
            return add_float_suffix(c_expr)
        except Exception as e:
            print(f"ERROR: C code generation failed for: {expr} ({e})")
            return f"/* ERROR generating code for: {expr} */"

    code = []
    code.append(f"// SymPy-Generated CUDA code for 3D {'ACM' if is_acm else 'CM'}")
    code.append("__device__ __forceinline__ static")
    func_sig = f"void {func_name}("
    func_sig += "float* f_post, "
    func_sig += "const float rho, const float ux, const float uy, const float uz, "
    func_sig += "const float omega"
    if is_acm:
        func_sig += ", const float lambda"
    func_sig += ") {"
    code.append(func_sig)

    code.append("\n    // velocity monomials ")
    priority_names_added = set()
    for expr, name in priority_exprs:
        if name not in priority_names_added:
            c_expr = format_ccode(expr)
            code.append(f"    const float {name} = {c_expr};")
            priority_names_added.add(name)

    code.append("\n    // CSE")
    for sym, subexpr in filtered_replacements_list:
        cse_expr_str = format_ccode(subexpr)
        code.append(f"    const float {sym} = {cse_expr_str};")

    code.append("\n    // f_post ")
    for i in range(27):
        final_expr_str = format_ccode(final_exprs_flat[i])
        code.append(f"    f_post[{i}] = {final_expr_str};")

    code.append("}")

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write("\n".join(code))
    print(f"Generated C++ code written to {filepath}")

generate_collision_code(
    f_post_acm,
    "apply",
    "D3Q27_ACM.gen",
    k_syms + (rho, ux, uy, uz, omega, lambda_),
    is_acm=True
)

generate_collision_code(
    f_post_cm,
    "apply",
    "D3Q27_CM.gen",
    k_syms + (rho, ux, uy, uz, omega),
    is_acm=False
)