import numpy as np

h_C_2d = np.array([
    [ 0,  0],
    [ 1,  0],
    [ 0,  1],
    [-1,  0],
    [ 0, -1],
    [ 1,  1],
    [-1,  1],
    [-1, -1],
    [ 1, -1]
])

h_C_3d = np.array([
    [ 0,  0,  0],
    [ 1,  0,  0], [-1,  0,  0],
    [ 0,  1,  0], [ 0, -1,  0],
    [ 0,  0,  1], [ 0,  0, -1],
    [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
    [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
    [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
    [ 1,  1,  1], [-1,  1,  1], [ 1, -1,  1], [-1, -1,  1],
    [ 1,  1, -1], [-1,  1, -1], [ 1, -1, -1], [-1, -1, -1]
])

def make_free_slip_map(h_C, flip_dim):
    n = h_C.shape[0]
    mapping = []
    for i in range(n):
        c = h_C[i].copy()
        c[flip_dim] *= -1
        for j in range(n):
            if np.all(h_C[j] == c):
                mapping.append(j)
                break
        else:
            raise RuntimeError(f"No match for direction {i} with flipped {flip_dim}")
    return mapping

def format_cpp_array(name, arr):
    s = f"constexpr int {name}[{len(arr)}] = {{\n    "
    s += ", ".join(str(x) for x in arr)
    s += "\n};"
    return s

map2d_x = make_free_slip_map(h_C_2d, 0)
map2d_y = make_free_slip_map(h_C_2d, 1)
print("// D2Q9 Free-slip mapping arrays")
print(format_cpp_array("FREE_SLIP_MAP_X_2D", map2d_x))
print()
print(format_cpp_array("FREE_SLIP_MAP_Y_2D", map2d_y))
print()

# D3Q27
map3d_x = make_free_slip_map(h_C_3d, 0)
map3d_y = make_free_slip_map(h_C_3d, 1)
map3d_z = make_free_slip_map(h_C_3d, 2)
print("// D3Q27 Free-slip mapping arrays")
print(format_cpp_array("FREE_SLIP_MAP_X", map3d_x))
print()
print(format_cpp_array("FREE_SLIP_MAP_Y", map3d_y))
print()
print(format_cpp_array("FREE_SLIP_MAP_Z", map3d_z))
