import sympy as sp

def generate_cuda_equilibrium():
    rho, ux, uy, cs = sp.symbols('rho ux uy cs')
    cs2_sym, cs4_sym = sp.symbols('cs2 cs4')
    substitutions = {cs**2: cs2_sym, cs**4: cs4_sym}
    
    C = sp.symbols('C[0:18]')
    W = sp.symbols('WEIGHTS[0:9]')
    
    #   f_eq = W[i] * rho * [ 1 + (ci·u)/cs^2 + (ci·u)^2/(2*cs^4) - (u·u)/(2*cs^2) ]
    f_exprs = []
    for i in range(9):
        cx = C[2 * i]
        cy = C[2 * i + 1]
        ci_dot_u = cx * ux + cy * uy
        expr = W[i] * rho * (
            1 + ci_dot_u/(cs**2) +
            (ci_dot_u**2)/(2*(cs**4)) -
            (ux**2 + uy**2)/(2*(cs**2))
        )
        expr = expr.replace(lambda e: isinstance(e, sp.Pow) and e.base == cs and e.exp == -2, lambda e: 1.0/cs2_sym)
        expr = expr.replace(lambda e: isinstance(e, sp.Pow) and e.base == cs and e.exp == -4, lambda e: 1.0/cs4_sym)

        u_dot_u = ux**2 + uy**2
        expr = expr.subs(u_dot_u, sp.Symbol('u_dot_u'))
        f_exprs.append(expr)
    
    code_lines = []
    for i, expr in enumerate(f_exprs):
        line = "    f_eq[get_node_index(node, {i})] = {expr};".format(i=i, expr=sp.ccode(expr))
        code_lines.append(line)
    
    body = "\n".join(code_lines)
    cuda_template = '''\
    float u_dot_u = ux * ux + uy * uy;
    float cs2 = cs * cs;
    float cs4 = cs2 * cs2;
''' + body
    return cuda_template

if __name__ == '__main__':
    cuda_code = generate_cuda_equilibrium()
    print(cuda_code)
