import sympy as sp
import numpy as np

ux, uy, uz = sp.symbols('ux uy uz')

c_ix = np.array([0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1])
c_iy = np.array([0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1])
c_iz = np.array([0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1])

M = sp.zeros(27, 27)

for j in range(27):
    cx = c_ix[j]
    cy = c_iy[j]
    cz = c_iz[j]

    M[0, j]  = 1                              # T0: density
    M[1, j]  = cx                           # T1: jx
    M[2, j]  = cy                           # T2: jy
    M[3, j]  = cz                           # T3: jz

    M[4, j]  = cx * cy                    # T4: xy
    M[5, j]  = cx * cz                    # T5: xz
    M[6, j]  = cy * cz                    # T6: yz

    M[7, j]  = cx**2 - cy**2              # T7: xx-yy
    M[8, j]  = cx**2 - cz**2              # T8: xx-zz
    M[9, j]  = cx**2 + cy**2 + cz**2    # T9: e (trace)

    M[10, j] = cx * cy**2 + cx * cz**2  # T10
    M[11, j] = cx**2 * cy + cy * cz**2  # T11

    M[12, j] = cx**2 * cz + cy**2 * cz  # T12
    M[13, j] = cx * cy**2 - cx * cz**2  # T13

    M[14, j] = cx**2 * cy - cy * cz**2  # T14
    M[15, j] = cx**2 * cz - cy**2 * cz  # T15


    M[16, j] = cx * cy * cz              # T16
    M[17, j] = cx**2 * cy**2 + cx**2 * cz**2 + cy**2 * cz**2  # T17
    M[18, j] = cx**2 * cy**2 + cx**2 * cz**2 - cy**2 * cz**2  # T18
    M[19, j] = cx**2 * cy**2 - cx**2 * cz**2                    # T19
    M[20, j] = cx**2 * cy * cz                                   # T20
    M[21, j] = cx * cy**2 * cz                                   # T21
    M[22, j] = cx * cy * cz**2                                   # T22
    M[23, j] = cx * cy**2 * cz**2                               # T23
    M[24, j] = cx**2 * cy * cz**2                               # T24
    M[25, j] = cx**2 * cy**2 * cz                               # T25
    M[26, j] = cx**2 * cy**2 * cz**2                            # T26

# raw_rows = [M.row(i) for i in range(27)]
# orth_rows = sp.matrices.GramSchmidt(raw_rows)
# M = sp.Matrix(orth_rows)
# for k in range(4, 27):
#     for i in range(k):
#         proj = (M[k,:].dot(M[i,:])) / (M[i,:].dot(M[i,:]))
#         M[k,:] = sp.simplify(M[k,:] - proj * M[i,:])

M_inv = M.inv()

print(M)

# sp.pprint(M)
# sp.pprint(M_inv)

# identity = (M * M_inv).evalf()
# print("Check M * M_inv ≃ Identity:")
# sp.pprint(identity)

def format_entry(e):
    e = sp.simplify(e)
    if e.is_Rational:
        num, den = int(e.p), int(e.q)
        if num == 0:
            return "0.0f"
        if abs(num) == den:
            sign = "-" if num<0 else ""
            return f"{sign}1.0f"
        return f"{num}.0f/{den}.0f"
    else:
        # float
        return f"{float(e):.6f}f"

print("const float h_M[27*27] = {")
for i in range(27):
    row = ", ".join(format_entry(M[i,j]) for j in range(27))
    print("    " + row + ("," if i<26 else ""))
print("};\n")

print("const float h_M_inv[27*27] = {")
for i in range(27):
    row = ", ".join(format_entry(M_inv[i,j]) for j in range(27))
    print("    " + row + ("," if i<26 else ""))
print("};")