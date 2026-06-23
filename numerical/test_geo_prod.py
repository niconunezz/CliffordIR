from generate_oracle import generate_geo_prods_matrices, generate_rotate_matrices, generate_rotate_appl_matrices
import pytest
import numpy as np
import pycuda.driver as cuda
import subprocess
import random
from clifford import Cl
DEBUG = True

layout, blades = Cl(2, 0, 1, firstIdx=0)
e0 = blades['e0']
e1 = blades['e1']
e2 = blades['e2']
e01 = blades['e01']
e02 = blades['e02']
e12 = blades['e12']
e012 = blades['e012']

# Grado 1 (vectores)
mve0       = lambda a,b      : a + b*e0
mve1       = lambda a,b      : a + b*e1
mve2       = lambda a,b      : a + b*e2
mve0e1     = lambda a,b,c    : a + b*e0  + c*e1
mve0e2     = lambda a,b,c    : a + b*e0  + c*e2
mve1e2     = lambda a,b,c    : a + b*e1  + c*e2
mve0e1e2   = lambda a,b,c,d  : a + b*e0  + c*e1  + d*e2

# Grado 1 + grado 2 (mezclados)
mve0e12    = lambda a,b,c    : a + b*e0  + c*e12
mve1e01    = lambda a,b,c    : a + b*e1  + c*e01
mve1e02    = lambda a,b,c    : a + b*e1  + c*e02
mve2e01    = lambda a,b,c    : a + b*e2  + c*e01
mve1e12    = lambda a,b,c    : a + b*e1  + c*e12
mve2e02    = lambda a,b,c    : a + b*e2  + c*e02
mve0e01    = lambda a,b,c    : a + b*e0  + c*e01
mve2e12    = lambda a,b,c    : a + b*e2  + c*e12
mve0e1e01  = lambda a,b,c,d  : a + b*e0  + c*e1  + d*e01
mve0e1e12  = lambda a,b,c,d  : a + b*e0  + c*e1  + d*e12
mve1e2e12  = lambda a,b,c,d  : a + b*e1  + c*e2  + d*e12
mve0e2e02  = lambda a,b,c,d  : a + b*e0  + c*e2  + d*e02

# Grado 2 (bivectores puros)
mve12     = lambda a,b   : a + b*e12
mve02     = lambda a,b   : a + b*e02
mve01     = lambda a,b   : a + b*e01
mve01e02  = lambda a,b,c : a + b*e01 + c*e02
mve01e12  = lambda a,b,c : a + b*e01 + c*e12
mve02e12  = lambda a,b,c : a + b*e02 + c*e12
mvfull = lambda a,b,c,d  : a + b*e01 + c*e02 + d*e12   # alias para mvfull

# Grado 3 (pseudoescalar)
mve012     = lambda a,b      : a + b*e012

# Grado 2 + grado 3
mve01e012  = lambda a,b,c    : a + b*e01 + c*e012
mve12e012  = lambda a,b,c    : a + b*e12 + c*e012
mve01e02e012 = lambda a,b,c,d: a + b*e01 + c*e02 + d*e012
mvfull_e012  = lambda a,b,c,d,e: a + b*e01 + c*e02 + d*e12 + e*e012

# Grado 1 + grado 3
mve0e012   = lambda a,b,c    : a + b*e0  + c*e012
mve1e012   = lambda a,b,c    : a + b*e1  + c*e012

mve0e1e2e01e02e12e012 = lambda a, b, c, d ,e , f, g, h : a + b*e0 + c*e1 + d*e2 + e*e01 + f*e02 + g*e12 + h*e012
mv_point = lambda a, b, c, d: 0 + b*e02 + c*e01 + e12
mv_scalar = lambda a : a*e1*e1

def to_mask(arr):
    out = 0
    for i, el in enumerate(arr):
        if el != 0:
            out += 2**i
    return out

def init_device(ptx, matrices, result, func_name):
    mod  = cuda.module_from_buffer(ptx.encode())
    # device init
    
    dev_matrices = [cuda.mem_alloc(m.nbytes) for m in matrices]
    result_dev = cuda.mem_alloc(result.nbytes)
    for dev_m, m in zip(dev_matrices, matrices) :  cuda.memcpy_htod(dev_m, m)

    func = mod.get_function(func_name)
    return func, dev_matrices, result_dev

import math
def generate_layout(N: int, els_per_thread: int) -> str:
    assert N >= 32, "N must be at least 32"

    def dim_entries(count: int, start: int) -> tuple[str, int]:
        if count <= 1:
            return "[]", start
        steps = int(math.log2(count))
        entries = [f"[{start << k}]" for k in range(steps)]
        return "[" + ", ".join(entries) + "]", start << steps

    i = 1
    total_threads = N // els_per_thread

    register,  i = dim_entries(els_per_thread, i)
    lane,      i = dim_entries(min(total_threads, 32), i)
    warp,      i = dim_entries(min(8, total_threads // 32), i)
    block,     i = dim_entries(total_threads // 256, i)

    return f"{{register = {register}, lane = {lane}, warp = {warp}, block = {block}}}"





@pytest.fixture(scope="session", autouse=True)
def cuda_ctx():
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    yield
    ctx.pop()

CASES = [
    ("e12__e12",       mve12,    mve12,    2, 2, 2, 128, 1),
    ("e12__e12",       mve12,    mve12,    2, 2, 2, 4096, 16),
    ("e02__e02",       mve02,    mve02,    2, 2, 2, 64, 1),
    ("e01__e01",       mve01,    mve01,    2, 2, 2, 128, 4),
    ("e01e02__e01e02", mve01e02, mve01e02, 3, 3, 3, 64, 1),
    ("e01e12__e01e12", mve01e12, mve01e12, 3, 3, 4, 64, 1),
    ("e02e12__e02e12", mve02e12, mve02e12, 3, 3, 4, 4096, 4),
    ("e1e12__e0e01",   mve1e12,  mve0e01,  3, 3, 7, 64, 1),
    ("full__full",     mvfull,   mvfull,   4, 4, 4, 128, 4),
    ("scalar__full",   lambda a: a*e1*e1,  mvfull, 1, 4, 4, 64, 1),
    ("full__scalar",   mvfull,  lambda a: a*e1*e1, 4, 1, 4, 64, 1),
    ("complete__complete", mve0e1e2e01e02e12e012, mve0e1e2e01e02e12e012, 8, 8, 8, 128, 1),

    ("e0__e0",         mve0,     mve0,     2, 2, 2, 64, 1),
    ("e1__e1",         mve1,     mve1,     2, 2, 2, 64, 1),
    ("e2__e2",         mve2,     mve2,     2, 2, 2, 64, 2),
    ("e0__e1",         mve0,     mve1,     2, 2, 4, 4096, 16),
    ("e1__e2",         mve1,     mve2,     2, 2, 4, 512, 4),
    ("e0__e2",         mve0,     mve2,     2, 2, 4, 128, 1),
    ("e0e1__e0e2",     mve0e1,   mve0e2,   3, 3, 7, 64, 2),
    ("e0e1__e1e2",     mve0e1,   mve1e2,   3, 3, 7, 8192, 4),
    ("e0e2__e1e2",     mve0e2,   mve1e2,   3, 3, 7, 4096, 4),
    ("e0e1e2__e0e1e2", mve0e1e2, mve0e1e2, 4, 4, 7, 512, 2),
    ("e0__e01",        mve0,     mve01,    2, 2, 3, 1024, 8),
    ("e1__e12",        mve1,     mve12,    2, 2, 4, 32, 1),
    ("e2__e02",        mve2,     mve02,    2, 2, 4, 64, 1),
    ("e0e1__e12",      mve0e1,   mve12,    3, 2, 6, 64, 1),
    ("e0e2__e12",      mve0e2,   mve12,    3, 2, 6, 64, 2),
    ("e1e2__e12",      mve1e2,   mve12,    3, 2, 4, 4096, 1),
    ("e0e1e2__full",   mve0e1e2, mvfull,   4, 4, 8, 64, 2),
    ("e12__e1",        mve12,    mve1,     2, 2, 4, 64, 1),
    ("e01__e0",        mve01,    mve0,     2, 2, 3, 512, 8),
    ("full__e0e1e2",   mvfull,   mve0e1e2, 4, 4, 8, 4096, 1),

    ("e012__scalar",   mve012,   lambda a: a*e1*e1, 2, 1, 2, 64, 1),
    ("e012__e012",     mve012,   mve012,   2, 2, 2, 64, 1),
    ("e012__e0",       mve012,   mve0,     2, 2, 3, 64, 1),
    ("e012__e1",       mve012,   mve1,     2, 2, 4, 64, 1),
    ("e012__e0e1e2",   mve012,   mve0e1e2, 2, 4, 7, 512, 1),
    ("e012__e01",      mve012,   mve01,    2, 2, 3, 1024, 8),
    ("e012__e12",      mve012,   mve12,    2, 2, 4, 8192, 1),
    ("e012__full",     mve012,   mvfull,   2, 4, 6, 64, 1),

    ("e01e012__e01e012",     mve01e012, mve01e012,    3, 3, 3, 512, 1),
    ("e12e012__e12e012",     mve12e012, mve12e012,    3, 3, 4, 64, 1),
    ("full_e012__full_e012", mvfull_e012, mvfull_e012, 5, 5, 6, 64, 1),
    ("e0e1e2__e01e02e012",   mve0e1e2,  mve01e02e012, 4, 4, 7, 64, 1),

    ("e0e012__e0e012",   mve0e012, mve0e012, 3, 3, 3, 64, 1),
    ("e1e012__e1e012",   mve1e012, mve1e012, 3, 3, 4, 512, 1),
    ("e0e012__full",     mve0e012, mvfull,   3, 4, 6, 4096, 4),
    ("e1e012__full",     mve1e012, mvfull,   3, 4, 8, 8192, 1),

    ("e0e12__e0e12",         mve0e12,   mve0e12,   3, 3, 4, 512, 1),
    ("e0e12__full",          mve0e12,   mvfull,    3, 4, 6, 8192, 1),
    ("e1e01__e1e01",         mve1e01,   mve1e01,   3, 3, 4, 4096, 1),
    ("e1e02__e2e01",         mve1e02,   mve2e01,   3, 3, 7, 32, 1),
    ("e0e1e01__e0e1e01",     mve0e1e01, mve0e1e01, 4, 4, 4, 512, 1),
    ("e0e1e12__e1e2e12",     mve0e1e12, mve1e2e12, 4, 4, 8, 128, 4),
    ("e0e1e2__e01e02e12",    mve0e1e2,  mvfull,    4, 4, 8, 4096, 1),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_geo_prod(case, cuda_ctx):
    name, mv_a, mv_b, comps_a, comps_b, comps_c, N, els_per_thread = case

    layout = generate_layout(N, els_per_thread)
    total_threads = N // els_per_thread
    num_blocks = max(total_threads//(256), 1)
    num_threads = min(total_threads, 256)

    #todo: cache it if neccesary
    c_default_indices, a_default_indices, b_default_indices= generate_geo_prods_matrices(N, mv_a, mv_b, comps_a, comps_b, comps_c)
    result_host = np.zeros(N * comps_c, dtype=np.float32)
    
    mask_perm = [0, 1, 2, 4, 3, 5, 6, 7]
    np_mask_perm = np.array(mask_perm)
    adapted_mask_c = np_mask_perm[c_default_indices].argsort()
    adapted_mask_a = np_mask_perm[a_default_indices].argsort()
    adapted_mask_b = np_mask_perm[b_default_indices].argsort()
    A = np.load("numerical/matrices/matrix_a.npz")['arr_0']
    B = np.load("numerical/matrices/matrix_b.npz")['arr_0']

    # change from canonical -> bitmap layout, see explanation in cliffOps.cpp
    #! note we are using the same exact process, and it only works because the mask
    #! is its own inverse, if for other p,q,r this is not the case there should be 2 masks

    A = A.reshape(comps_a, N)[adapted_mask_a, :]
    A = A.flatten()

    B = B.reshape(comps_b, N)[adapted_mask_b, :]
    B = B.flatten()

    expected = np.load("numerical/matrices/matrix_c.npz")['arr_0']
   

    a_sample_coeffs = [random.randint(2, 399) for _ in range(comps_a)]
    b_sample_coeffs = [random.randint(2, 399) for _ in range(comps_b)]
    a_sample = mv_a(*a_sample_coeffs) ; b_sample = mv_b(*b_sample_coeffs)
    c_sample = (a_sample * b_sample)
   
    mask_mv_a = to_mask(a_sample.as_array()[mask_perm])
    mask_mv_b = to_mask(b_sample.as_array()[mask_perm])
    mask_mv_c = to_mask(c_sample.as_array()[mask_perm])

    # generate ptx for case
    subprocess.run(["./numerical/scripts/compile_case.sh", str(mask_mv_a), str(mask_mv_b), str(mask_mv_c), str(N), layout, "output.ptx"])

    with open("output.ptx", "r") as f:
        ptx = f.read()

    func, dev_matrices, result_dev = init_device(ptx, [A, B], result_host, "geo_prod")
    A_dev, B_dev = dev_matrices

    func(A_dev, B_dev, result_dev,
     block=(num_threads, 1, 1),
     grid=(num_blocks, 1, 1))
    
    cuda.memcpy_dtoh(result_host, result_dev)
    
    # change from bitmap layout -> canonical 
    result_host = result_host.reshape(comps_c, N)[adapted_mask_c, :]
    result_host = result_host.flatten()
    if DEBUG:
        if np.allclose(result_host, expected, atol=1e-4):
            print("✓ CORRECTO")
        else:
            diff = np.where(~np.isclose(result_host, expected, atol=1e-4))
            print(f"A outputs: {A.reshape(comps_a, N)[:, 0]}")
            print(f"B outputs : {B.reshape(comps_b, N)[:, 0]}")
            print(f"Kernel outcome : {result_host.reshape(comps_c, N)[:, 0]}")
            print(f"Expected outcome : {expected.reshape(comps_c, N)[:, 0]}")
            print(f"✗ INCORRECTO en {len(diff[0]) / (comps_c * N)} de los índices")
            print(f"✗ INCORRECTO en índices: {diff}")

            print(f"  got:      {result_host[diff]}")
            print(f"  expected: {expected[diff]}")

    assert(np.allclose(result_host, expected, atol=1e-5))


CASES_ROTATE = [
    ("N32_1",       mv_point,    mv_scalar,    4, 1, 4,   32,   1),
    ("N64_1",       mv_point,    mv_scalar,    4, 1, 4,   64,   1),
    ("N64_2",       mv_point,    mv_scalar,    4, 1, 4,   64,   2),
    ("N128_1",      mv_point,    mv_scalar,    4, 1, 4,  128,   1),
    ("N128_2",      mv_point,    mv_scalar,    4, 1, 4,  128,   2),
    ("N128_4",      mv_point,    mv_scalar,    4, 1, 4,  128,   4),
    ("N256_1",      mv_point,    mv_scalar,    4, 1, 4,  256,   1),
    ("N256_2",      mv_point,    mv_scalar,    4, 1, 4,  256,   2),
    ("N256_4",      mv_point,    mv_scalar,    4, 1, 4,  256,   4),
    ("N256_8",      mv_point,    mv_scalar,    4, 1, 4,  256,   8),
    ("N512_1",      mv_point,    mv_scalar,    4, 1, 4,  512,   1),
    ("N512_2",      mv_point,    mv_scalar,    4, 1, 4,  512,   2),
    ("N512_4",      mv_point,    mv_scalar,    4, 1, 4,  512,   4),
    ("N512_8",      mv_point,    mv_scalar,    4, 1, 4,  512,   8),
    ("N512_16",     mv_point,    mv_scalar,    4, 1, 4,  512,  16),
    ("N1024_1",     mv_point,    mv_scalar,    4, 1, 4, 1024,   1),
    ("N1024_2",     mv_point,    mv_scalar,    4, 1, 4, 1024,   2),
    ("N1024_4",     mv_point,    mv_scalar,    4, 1, 4, 1024,   4),
    ("N1024_8",     mv_point,    mv_scalar,    4, 1, 4, 1024,   8),
    ("N1024_16",    mv_point,    mv_scalar,    4, 1, 4, 1024,  16),
]

@pytest.mark.parametrize("case", CASES_ROTATE, ids=[c[0] for c in CASES_ROTATE])
def test_rotate(case, cuda_ctx):
    name , mv_a, mv_b, comps_a, comps_b, comps_c, N, els_per_thread = case

    layout = generate_layout(N, els_per_thread)
    total_threads = N // els_per_thread
    num_blocks = max(total_threads//(256), 1)
    num_threads = min(total_threads, 256)

    #todo: cache it if neccesary
    generate_rotate_matrices(N, mv_a, mv_b, comps_a, comps_b, comps_c)
    result_host = np.zeros(N * comps_c, dtype=np.float32)
    
    A = np.load("numerical/matrices/rotation/matrix_a.npz")['arr_0']
    #! todo: study why this happens
    A = A.reshape(comps_a, N)[[0, 2, 1, 3]]
    B = np.load("numerical/matrices/rotation/matrix_b.npz")['arr_0']

    expected = np.load("numerical/matrices/rotation/matrix_c.npz")['arr_0']

    # generate ptx for case
    subprocess.run(["./numerical/scripts/compile_rotation.sh", str(N), layout, "output.ptx"])

    with open("output.ptx", "r") as f:
        ptx = f.read()
    
    func, dev_matrices, result_dev = init_device(ptx, [A, B], result_host, "rotation")
    A_dev, B_dev = dev_matrices 

    func(A_dev, B_dev, result_dev,
     block=(num_threads, 1, 1),
     grid=(num_blocks, 1, 1))
    
    cuda.memcpy_dtoh(result_host, result_dev)

    assert(np.allclose(result_host, expected, atol=1e-5))
    


CASES_ROTATE_APPL = [
    ("N64",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 64),
    ("N128",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 128),
    ("N256",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 256),
    ("N512",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 512),
    ("N1024",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 1024),
    ("N2048",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 2048),
    ("N4096",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 4096),
    ("N8192",       mv_point, mv_point,    mv_scalar,    4, 4, 1, 4, 8192),

]

@pytest.mark.parametrize("case", CASES_ROTATE_APPL, ids=[c[0] for c in CASES_ROTATE_APPL])
def test_rotate_appl(case, cuda_ctx):
    name , mv_x, mv_y, mv_alpha, comps_x, comps_y, comps_alpha, comps_c, N = case
    num_blocks = max(N//256, 1)
    num_threads = min(N, 1024)
    #todo: cache it if neccesary
    generate_rotate_appl_matrices(N, mv_x, mv_y, mv_alpha, comps_x, comps_y, comps_alpha, comps_c)
    result_host = np.zeros(N * comps_c, dtype=np.float32)
    
    X = np.load("numerical/matrices/rotation_appl/matrix_x.npz")['arr_0']
    Y = np.load("numerical/matrices/rotation_appl/matrix_y.npz")['arr_0']
    #! todo: study why this happens
    X = X.reshape(comps_x, N)[[0, 2, 1, 3]]
    Y = Y.reshape(comps_y, N)[[0, 2, 1, 3]]

    alpha = np.load("numerical/matrices/rotation_appl/matrix_alpha.npz")['arr_0']
    expected = np.load("numerical/matrices/rotation_appl/matrix_c.npz")['arr_0']

    # generate ptx for case
    subprocess.run(["./numerical/scripts/compile_end_to_end.sh", str(N), "output.ptx"])

    with open("output.ptx", "r") as f:
        ptx = f.read()
    
    func, dev_matrices, result_dev = init_device(ptx, [X, Y, alpha], result_host, "rotation")
    X_dev, Y_dev, alpha_dev = dev_matrices 
    print(f"N : {N}")
    print(f"Num_blocks : {num_blocks}")

    func(X_dev, Y_dev, alpha_dev, result_dev,
     block=(num_threads, 1, 1),
     grid=(num_blocks, 1, 1))
    
    cuda.memcpy_dtoh(result_host, result_dev)
    if DEBUG:
        if np.allclose(result_host, expected, atol=1e-5):
            print("✓ CORRECTO")
        else:
            diff = np.where(~np.isclose(result_host, expected, atol=1e-5))
            print(f"X outputs: {X.reshape(comps_x, N)[:, 0]}")
            print(f"Y outputs: {Y.reshape(comps_y, N)[:, 0]}")
            print(f"alpha outputs : {alpha.reshape(comps_alpha, N)[:, 0]}")
            print(f"Kernel outcome : {result_host.reshape(comps_c, N)[:, 0]}")
            print(f"Expected outcome : {expected.reshape(comps_c, N)[:, 0]}")
            print(f"✗ INCORRECTO en {len(diff[0]) / (comps_c * N)} de los índices")
            print(f"✗ INCORRECTO en índices: {diff}")

            print(f"  got:      {result_host[diff]}")
            print(f"  expected: {expected[diff]}")

    assert(np.allclose(result_host, expected, atol=1e-5))