import sympy as sp
from sympy.codegen.rewriting import create_expand_pow_optimization
import re

# Function to add 'f' suffix to float literals in CUDA code
def add_float_suffix(c_code):
    pattern = r'(\b\d*\.\d+|\b\d+\.\d*[eE][+-]?\d+)(?!\w*f\b)'
    return re.sub(pattern, r'\1f', c_code)

# Define power expansion optimization (expands pow(expr, n) for n <= 3)
expand_pow = create_expand_pow_optimization(3)

def generate_cuda_equilibrium_3d():
    # Define the discrete velocity set h_C (27 directions, 3 components each)
    h_C = [
        0, 0, 0,   # center
        1, 0, 0,   # face neighbors
        -1, 0, 0,
        0, 1, 0,
        0, -1, 0,
        0, 0, 1,
        0, 0, -1,
        1, 1, 0,   # edge neighbors
        -1, 1, 0,
        1, -1, 0,
        -1, -1, 0,
        1, 0, 1,
        -1, 0, 1,
        1, 0, -1,
        -1, 0, -1,
        0, 1, 1,
        0, -1, 1,
        0, 1, -1,
        0, -1, -1,
        1, 1, 1,   # corner neighbors
        -1, 1, 1,
        1, -1, 1,
        -1, -1, 1,
        1, 1, -1,
        -1, 1, -1,
        1, -1, -1,
        -1, -1, -1
    ]
    
    # Define symbolic variables
    rho, ux, uy, uz, cs = sp.symbols('rho ux uy uz cs')
    cs2_sym, cs4_sym, cs2_cs4_sym, half_rho_sym = sp.symbols('cs2 cs4 cs2_cs4 half_rho')
    
    # Define extra symbols for cached monomials
    ux2, uy2, uz2, one_over_cs2_cs4 = sp.symbols('ux2 uy2 uz2 1_over_cs2_cs4')
    
    # Define weights as symbolic array (weights[0] to weights[26])
    W = [sp.symbols(f'weights[{i}]') for i in range(27)]
    
    # Compute u_dot_u symbolically (this will get substituted later)
    u_dot_u = ux * ux + uy * uy + uz * uz
    
    # For each equilibrium expression, we reorganize the terms to factor out 1/(cs2*cs4)
    # so that common factors cs2*cs4 and its inverse (1_over_cs2_cs4) can be substituted.
    # The intended form is:
    #    half_rho*weights[i]*1/(cs2*cs4)*
    #       (2.0*cs2*cs4 + cs2*(c_i · u)^2 + 2.0*cs4*(c_i · u) - cs4*(u_dot_u))
    #
    # Note: For example, for index 9 where c = (1, -1, 0), (c · u) becomes (ux - uy),
    # matching the substitution you provided.
    
    f_exprs = []
    for i in range(27):
        # Extract the components of c_i from h_C
        c_x = float(h_C[3 * i])
        c_y = float(h_C[3 * i + 1])
        c_z = float(h_C[3 * i + 2])
        
        # Compute the dot product c_i · u
        ci_dot_u = c_x * ux + c_y * uy + c_z * uz
        
        # Reorganize the equilibrium expression to factor out 1/(cs2*cs4)
        expr = W[i] * half_rho_sym * (1/(cs2_sym * cs4_sym)) * (
            2.0 * cs2_sym * cs4_sym +
            cs2_sym * ci_dot_u**2 +
            2.0 * cs4_sym * ci_dot_u -
            cs4_sym * u_dot_u
        )
        
        # Simplify and expand powers
        expr = sp.simplify(expr)
        
        # Substitute common subexpressions:
        subs_dict = {
            ux*ux: ux2,
            uy*uy: uy2,
            uz*uz: uz2,
            cs2_sym * cs4_sym: cs2_cs4_sym,
            1/(cs2_sym * cs4_sym): one_over_cs2_cs4
        }
        expr = expr.subs(subs_dict)
        expr = sp.simplify(expr)
        expr = expand_pow(expr)
        expr = expr.subs(subs_dict)
        f_exprs.append(expr)
    
    # Generate CUDA code for equilibrium assignments
    code_lines = []
    for i, expr in enumerate(f_exprs):
        # Generate C code for the expression and add the float suffix
        c_expr = add_float_suffix(sp.ccode(expr))
        line = f"    f_eq[get_node_index(node, {i})] = {c_expr};"
        code_lines.append(line)
    
    # Combine into CUDA template with precalculated values
    cuda_template = '''\
float ux2 = ux * ux;
float uy2 = uy * uy;
float uz2 = uz * uz;
float u_dot_u = ux2 + uy2 + uz2;
float cs2 = cs * cs;
float cs4 = cs2 * cs2;
float cs2_cs4 = cs2 * cs4;
float 1_over_cs2_cs4 = 1.0f / cs2_cs4;
float half_rho = 0.5f * rho;
''' + "\n" + "\n".join(code_lines)
    
    return cuda_template

# Execute and print the generated CUDA code
if __name__ == '__main__':
    cuda_code = generate_cuda_equilibrium_3d()
    print(cuda_code)