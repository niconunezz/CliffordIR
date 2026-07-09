import numpy as np
import math 
import os

def initialize_matrix(comps, N):
    return np.random.rand(comps, N).astype(np.float32)


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


def rotate(mv_a, mv_alpha):
    return math.e**(mv_alpha * mv_a)

def complete_rotation(mv_x, mv_y, mv_alpha):
    R = math.e**(mv_alpha*mv_x)
    return R*mv_y*~R

def generate_geo_prods_matrices(N, mv_a, mv_b, comps_a, comps_b, comps_c):

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


def generate_rotate_matrices(N, mv_a, mv_b, comps_a, comps_b, comps_c, mode):

    C = np.empty((comps_c, N), dtype=np.float32)
    matrix_a = initialize_matrix(comps_a, N)
    matrix_b = initialize_matrix(comps_b, N)

    for j in range(N):
        mv0 = mv_a(*matrix_a[:, j])
        mv1 = mv_b(*matrix_b[:, j])
        mv_out= rotate(mv0, mv1)
        vals = mv_out.value

        coeffs = vals[[0, 4, 5, 6]]

        if (j == 0):  print(f"mv0 : {mv0}");  print(f"mv0 : {mv0.value}") ; print(f"mv1 : {mv1}") ;  print(f"out : {mv_out}")

        for i in range(comps_c): 
            C[i, j] = coeffs[i]

    # force euclidean point : x*e01 + y*e02 + e12 
    matrix_a[-1, :] = 1

    np.savez(f"numerical/matrices/{mode}/rotation/matrix_a", matrix_a.flatten())
    np.savez(f"numerical/matrices/{mode}/rotation/matrix_b", matrix_b.flatten())
    np.savez(f"numerical/matrices/{mode}/rotation/matrix_c", C.flatten())


coeffs_indices = {
    "pga2d" : [0, 4, 5, 6],
    "pga3d" : [0, 5, 6, 7, 11, 12, 13, 14]}


def generate_rotate_appl_matrices(N, mv_x, mv_y, mv_alpha, comps_x, comps_y, comps_alpha, comps_c, mode : str):

    dir_name = f"numerical/matrices/{mode}/rotation_appl/{N}"
    if os.path.isdir(dir_name):
        return

    C = np.empty((comps_c, N), dtype=np.float32)
    matrix_x = initialize_matrix(comps_x, N)
    matrix_y = initialize_matrix(comps_y, N)

    matrix_alpha = np.random.randn(comps_alpha, N).astype(np.float32)

    for j in range(N):
        mvx = mv_x(*matrix_x[:, j])
        mvy = mv_y(*matrix_y[:, j])

        mvalpha = mv_alpha(*matrix_alpha[:, j])
        mv_out= complete_rotation(mvx, mvy, mvalpha)
        vals = mv_out.value
        
        coeffs = vals[coeffs_indices[mode]]

        if (j == 0):  print(f"mvx : {mvx}");  print(f"mvy : {mvy}") ; print(f"mvalpha : {mvalpha}") ;  print(f"out : {mv_out}")

        for i in range(comps_c):
            C[i, j] = coeffs[i]

    # force euclidean point
    #! we assume here that the last multivector is the one with coeff==1, 
    # not really sure though
    matrix_x[-1, :] = 1
    matrix_y[-1, :] = 1

    os.makedirs(f"numerical/matrices/{mode}/rotation_appl/{N}")
    np.savez(f"numerical/matrices/{mode}/rotation_appl/{N}/matrix_x", matrix_x.flatten())
    np.savez(f"numerical/matrices/{mode}/rotation_appl/{N}/matrix_y", matrix_y.flatten())
    np.savez(f"numerical/matrices/{mode}/rotation_appl/{N}/matrix_alpha", matrix_alpha.flatten())
    np.savez(f"numerical/matrices/{mode}/rotation_appl/{N}/matrix_c", C.flatten())