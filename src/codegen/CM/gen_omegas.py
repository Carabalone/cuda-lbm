import sympy as sp
from sympy.codegen.rewriting import create_expand_pow_optimization
import re
import time
import numpy as np
import os

min_usage_cse = 16
expand_powers = create_expand_pow_optimization(3) # Expand powers up to 3
output_dir = "generated_collision_code"
os.makedirs(output_dir, exist_ok=True)

ux, uy, uz = sp.symbols('ux uy uz')
rho = sp.symbols('rho')
omega, lambda_ = sp.symbols('omega lambda_')
k_syms = sp.symbols(f'k[0:{27}]')
k = sp.Matrix(k_syms)
cs_sq = sp.Rational(1, 3)

c_ix = np.array([0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1], dtype=int)
c_iy = np.array([0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1], dtype=int)
c_iz = np.array([0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1], dtype=int)

k_eq = sp.zeros(27, 1)
k_eq[0]  = rho
k_eq[9]  = rho * cs_sq**2  # k9 = rho * cs^4
k_eq[17] = rho * cs_sq**3  # k17 = rho * cs^6
k_eq[18] = rho * cs_sq**3  # k18 = rho * cs^6
k_eq[26] = rho * cs_sq**4  # k26 = rho * cs^8
print("Defined k_eq.")

diag_A_acm = ([1] * 4) + ([omega] * 5) + ([lambda_] * (27 - 9))
A_acm = sp.diag(*diag_A_acm)
# CM: lambda_ = 1
diag_A_cm = ([1] * 4) + ([omega] * 5) + ([1] * (27 - 9))
A_cm = sp.diag(*diag_A_cm)
print("Defined Relaxation Matrices A_acm and A_cm.")
delta_k_coll_acm = -A_acm * (k - k_eq)
delta_k_coll_cm  = -A_cm  * (k - k_eq)
print("Calculated delta_k_coll = -A * (k - k_eq).")

print("Constructing Central Moment Matrix T...")
start_time = time.time()
T = sp.zeros(27, 27)
for j in range(27):
    cjx, cjy, cjz = c_ix[j], c_iy[j], c_iz[j]
    Cjx, Cjy, Cjz = cjx - ux, cjy - uy, cjz - uz
    Cjx2, Cjy2, Cjz2 = Cjx*Cjx, Cjy*Cjy, Cjz*Cjz
    # eq 11 simplified
    T[0, j]  = 1
    T[1, j]  = Cjx
    T[2, j]  = Cjy
    T[3, j]  = Cjz
    T[4, j]  = Cjx * Cjy
    T[5, j]  = Cjx * Cjz
    T[6, j]  = Cjy * Cjz
    T[7, j]  = Cjx2 - cs_sq
    T[8, j]  = Cjy2 - cs_sq
    T[9, j]  = Cjz2 - cs_sq
    T[10, j] = Cjx * (Cjy2 + Cjz2)
    T[11, j] = Cjy * (Cjx2 + Cjz2)
    T[12, j] = Cjz * (Cjx2 + Cjy2)
    T[13, j] = Cjx * (Cjy2 - Cjz2)
    T[14, j] = Cjy * (Cjx2 - Cjz2)
    T[15, j] = Cjz * (Cjx2 - Cjy2)
    T[16, j] = Cjx * Cjy * Cjz
    T[17, j] = Cjx2*Cjy2 + Cjx2*Cjz2 + Cjy2*Cjz2
    T[18, j] = Cjx2*Cjy2 + Cjx2*Cjz2 - Cjy2*Cjz2
    T[19, j] = Cjx2*Cjy2 - Cjx2*Cjz2
    T[20, j] = Cjx2 * Cjy * Cjz
    T[21, j] = Cjy2 * Cjx * Cjz
    T[22, j] = Cjz2 * Cjx * Cjy
    T[23, j] = Cjx * Cjy2 * Cjz2
    T[24, j] = Cjy * Cjx2 * Cjz2
    T[25, j] = Cjz * Cjx2 * Cjy2
    T[26, j] = Cjx2 * Cjy2 * Cjz2
print(f"Matrix T constructed in {time.time() - start_time:.2f} seconds.")

print("Inverting T ")
T_inv = T.inv()
print(f"Matrix T inverted")


print("Calculating Omega_acm = T_inv * delta_k_coll_acm...")
start_time = time.time()
Omega_acm = T_inv * delta_k_coll_acm
print(f"Omega_acm calculated in {time.time() - start_time:.2f} seconds.")

print("Calculating Omega_cm = T_inv * delta_k_coll_cm...")
start_time = time.time()
Omega_cm = T_inv * delta_k_coll_cm
print(f"Omega_cm calculated in {time.time() - start_time:.2f} seconds.")

def generate_collision_code(Omega_matrix, func_name, filename, input_symbols, is_acm=True):
    print(f"\n--- Generating code for {func_name} ---")
    print("Performing CSE...")
    start_time = time.time()

    def get_priority_exprs_3d():
        ux2, uy2, uz2 = ux*ux, uy*uy, uz*uz
        uxuy, uxuz, uyuz = ux*uy, ux*uz, uy*uz
        ux3, uy3, uz3 = ux2*ux, uy2*uy, uz2*uz
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
            (ux3, 'ux3'), (uy3, 'uy3'), (uz3, 'uz3'),
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

    # Generate the C++ function string
    code = []
    code.append(f"// SymPy-Generated CUDA code for 3D {'ACM' if is_acm else 'CM'} Collision Operator Omega")
    code.append(f"// Omega = -T_inv * A * (k - k_eq)")
    code.append(f"// Input moments: k0..k26")
    code.append("__device__ __forceinline__ static")
    # Function signature
    func_sig = f"void {func_name}("
    func_sig += "float* Omega_out, " # Output array
    func_sig += "const float* k_in, " # Input moments k0..k26
    func_sig += "const float rho, const float ux, const float uy, const float uz, "
    func_sig += "const float omega"
    if is_acm:
        func_sig += ", const float lambda_"
    func_sig += ") {"
    code.append(func_sig)

    code.append("\n    // --- Priority velocity monomials ---")
    priority_names_added = set()
    for expr, name in priority_exprs:
        if name not in priority_names_added:
            c_expr = format_ccode(expr)
            code.append(f"    const float {name} = {c_expr};")
            priority_names_added.add(name)

    code.append("\n    // --- Common subexpressions (CSE) ---")
    for sym, subexpr in filtered_replacements_list:
        cse_expr_str = format_ccode(subexpr)
        code.append(f"    const float {sym} = {cse_expr_str};")

    code.append("\n    // --- Collision Operator Omega elements ---")
    for i in range(27):
        final_expr_str = format_ccode(final_exprs_flat[i])
        code.append(f"    Omega_out[{i}] = {final_expr_str};")

    code.append("}")

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write("\n".join(code))
    print(f"Generated C++ code written to {filepath}")

# --- Generate Code for ACM ---
generate_collision_code(
    Omega_acm,
    "acm_collision_operator_3d",
    "acm_collision_operator_3d.cu",
    k_syms + (rho, ux, uy, uz, omega, lambda_),
    is_acm=True
)

# --- Generate Code for CM ---
generate_collision_code(
    Omega_cm,
    "cm_collision_operator_3d",
    "cm_collision_operator_3d.cu",
    k_syms + (rho, ux, uy, uz, omega),
    is_acm=False
)

print("\nScript finished.")