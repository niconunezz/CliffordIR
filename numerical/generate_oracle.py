import numpy as np
import random

def initialize_matrix(comps, N):
    out = np.empty((comps, N), dtype=np.float32)
    for j in range(N):
        for i in range(comps):
            out[i, j] = random.randint(1, 20)
    
    return out


def extract_geo_prod_comps(mv_a, mv_b, matrix_a, matrix_b, comps_c):
    # get indices for a regular a*b (comps_c non zero components)
    i = 0
    cond = True
    while (cond):
        mv0 = mv_a(*matrix_a[:, i])
        mv1 = mv_b(*matrix_b[:, i])
        mv_out= (mv0 * mv1)
        vals = mv_out.value
        c_default_indices = np.nonzero(vals)[0]
        coeffs = vals[c_default_indices]
        i += 1
        if (len(coeffs) == comps_c):
            cond=False
    
    return np.nonzero(mv0.value)[0], np.nonzero(mv1.value)[0], c_default_indices


def geo_prod_reverse(mv_a, mv_b):
    return ~(mv_a * mv_b)

def generate_matrices(N, mv_a, mv_b, comps_a, comps_b, comps_c):

    C = np.empty((comps_c, N), dtype=np.float32)
    matrix_a = initialize_matrix(comps_a, N)
    matrix_b = initialize_matrix(comps_b, N)

    a_default_indices, b_default_indices, c_default_indices = extract_geo_prod_comps(mv_a, mv_b, matrix_a, matrix_b, comps_c)

    for j in range(N):

        mv0 = mv_a(*matrix_a[:, j])
        mv1 = mv_b(*matrix_b[:, j])
        mv_out= geo_prod_reverse(mv0, mv1)
        vals = mv_out.value
        coeffs = vals[c_default_indices]

        if (j == 0):  print(f"mv0 : {mv0}") ; print(f"mv1 : {mv1}") ;  print(f"out : {mv_out}")

        for i in range(comps_c): 
            C[i, j] = coeffs[i]

    np.savez("numerical/matrices/matrix_a", matrix_a.flatten())
    np.savez("numerical/matrices/matrix_b", matrix_b.flatten())
    np.savez("numerical/matrices/matrix_c", C.flatten())
    
    return c_default_indices, a_default_indices, b_default_indices