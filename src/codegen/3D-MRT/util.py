import sympy as sp
from sympy import cse, Symbol
from collections import defaultdict
import re

def add_float_suffix(c_code):
    pattern = r'(\b\d*\.\d+|\b\d+\.\d*[eE][+-]?\d+)(?!\w*f\b)'
    return re.sub(pattern, r'\1f', c_code)

def integer_to_float(c_code):
    return re.sub(r'(\b\d+)(?![\df\.])', r'\1.0f', c_code)

def custom_cse(expr_list, min_usage=2, priority_subs=None):
    priority_subs = priority_subs or {}
    inverse_priority = {v: k for k, v in priority_subs.items()}
    
    pre_substituted = [e.subs(priority_subs) for e in expr_list]
    
    replacements, simplified_exprs = sp.cse(pre_substituted, optimizations='basic')
    
    def count_occurrences(sym, exprs):
        return sum(expr.count(sym) for expr in exprs)
    
    filtered_replacements = []
    occurrence_counts = defaultdict(int)
    
    for sym, expr in replacements:
        for e in simplified_exprs:
            occurrence_counts[sym] += e.count(sym)
    
    kept_symbols = set()
    for sym, expr in replacements:
        if occurrence_counts[sym] >= min_usage:
            filtered_replacements.append((sym, expr))
            kept_symbols.add(sym)
    
    added = True
    while added:
        added = False
        for sym, expr in replacements:
            if sym in kept_symbols or sym in inverse_priority:
                continue
                
            for subexpr in sp.preorder_traversal(expr):
                if subexpr in kept_symbols and occurrence_counts[sym] >= min_usage:
                    filtered_replacements.append((sym, expr))
                    kept_symbols.add(sym)
                    added = True
                    break
    
    replacement_symbols = [sym for sym, _ in replacements]
    filtered_replacements.sort(key=lambda x: replacement_symbols.index(x[0]))
    
    reverse_subs = {}
    for sym, expr in filtered_replacements:
        reverse_subs[expr] = sym
    
    final_exprs = []
    for expr in expr_list:
        new_expr = expr
        for old_expr, sym in sorted(reverse_subs.items(), 
                                  key=lambda x: len(str(x[0])), 
                                  reverse=True):
            new_expr = new_expr.subs(old_expr, sym)
        final_exprs.append(new_expr)
    
    return dict(filtered_replacements), final_exprs

def format_float(expr):
    if isinstance(expr, (sp.Number, float, int)):
        return f"{float(expr)}f"
    return str(expr)

def expr_to_ccode(expr):
    c_code = sp.ccode(expr, strict=False)

    c_code = add_float_suffix(c_code)

    c_code = integer_to_float(c_code)
    
    return c_code

def custom_cse_ccode(expr_list, min_usage=2, priority_subs=None):
    replacements, simplified = custom_cse(expr_list, min_usage, priority_subs)
    
    c_replacements = []
    for sym, expr in replacements.items():
        simpl_expr = sp.simplify(expr)
        c_code = expr_to_ccode(simpl_expr)
        c_replacements.append(f"const float {sym} = {c_code};")
    
    c_expressions = []
    for expr in simplified:
        simpl_expr = sp.simplify(expr)
        c_code = expr_to_ccode(simpl_expr)
        c_expressions.append(f"    {c_code},")
    
    header = "// Computed substitutions"
    subst_code = '\n'.join(c_replacements)
    
    array_code = "// Force array\nfloat F[27] = {\n" + \
                 '\n'.join(c_expressions) + \
                 "\n};"
    
    return f"{header}\n{subst_code}\n\n{array_code}"