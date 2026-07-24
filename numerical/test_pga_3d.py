from generate_oracle import MatrixGenerator
import pytest
import numpy as np
import pycuda.driver as cuda
import subprocess
import random
from clifford import Cl
import pycuda.driver as cuda
from helpers import to_mask, init_device, generate_layout
from helpers import ObjectType, GeometricObj, run_ga_kernel_test, print_debug_info, GATestConfig

DEBUG = True
MODE = "pga3d"

layout, blades = Cl(3, 0, 1, firstIdx=0)
e0 = blades['e0']
e1 = blades['e1']
e2 = blades['e2']
e3 = blades['e3']
e01 = blades['e01']
e02 = blades['e02']
e03 = blades['e03']
e12 = blades['e12']
e13 = blades['e13']
e23 = blades['e23']
e012 = blades['e012']
e013 = blades['e013']
e023 = blades['e023']
e123 = blades['e123']
e0123 = blades['e0123']


mve0       = lambda a,b      : a + b*e0
mve1       = lambda a,b      : a + b*e1
mve0e1e2   = lambda a,b,c,d  : a + b*e0  + c*e1  + d*e2
mve0e12    = lambda a,b,c    : a + b*e0  + c*e12
mve1e01    = lambda a,b,c    : a + b*e1  + c*e01
mve1e02    = lambda a,b,c    : a + b*e1  + c*e02
mve2e01    = lambda a,b,c    : a + b*e2  + c*e01
mve1e12    = lambda a,b,c    : a + b*e1  + c*e12
mve1e2e12  = lambda a,b,c,d  : a + b*e1  + c*e2  + d*e12
mve0e2e02  = lambda a,b,c,d  : a + b*e0  + c*e2  + d*e02
mve12     = lambda a,b   : a + b*e12
mve02     = lambda a,b   : a + b*e02
mve02e12  = lambda a,b,c : a + b*e02 + c*e12
mvfull = lambda a,b,c,d  : a + b*e01 + c*e02 + d*e12   # alias para mvfull
mve012     = lambda a,b      : a + b*e012
mve01e012  = lambda a,b,c    : a + b*e01 + c*e012
mve12e012  = lambda a,b,c    : a + b*e12 + c*e012
mvfull_e012  = lambda a,b,c,d,e: a + b*e01 + c*e02 + d*e12 + e*e012
mve0e012   = lambda a,b,c    : a + b*e0  + c*e012
mve1e012   = lambda a,b,c    : a + b*e1  + c*e012
mve0e1e2e01 = lambda a, b, c, d ,e : a + b*e0 + c*e1 + d*e2 + e*e01
mve0e1e2e01e02e12e012 = lambda a, b, c, d ,e , f, g, h : a + b*e0 + c*e1 + d*e2 + e*e01 + f*e02 + g*e12 + h*e012
mve123 = lambda a, b : a + e123*b
mve023 = lambda a, b : a + e023*b
mve0123 = lambda a, b : a + e0123*b
mv_point = lambda b, c, d, _ : b*e023 + c*e013 + d*e012 + e123
mv_line = lambda a, b, c, d, e, f : a* e01 + b*e02 + c*e03 + d*e12 + e*e13 + f*e23
mv_scalar = lambda a : a*e1*e1

mve1e2e3               = lambda b, c, d : e1*b + e2*c + e3*d
mve0e1e2e3             = lambda a, b, c, d, e : a + e0*b + e1*c + e2*d + e3*e
mve01e02e03e12e13e23   = lambda a, b, c, d, e, f, g : a + e01*b + e02*c + e03*d + e12*e + e13*f + e23*g
mve123                 = lambda a, b : a + e123*b
mve012e013e023e123     = lambda a, b, c, d, e : a + e012*b + e013*c + e023*d + e123*e
mve01e23               = lambda a, b, c : a + e01*b + e23*c
mve03e12               = lambda a, b, c : a + e03*b + e12*c
mve0e123               = lambda a, b, c : a + e0*b + e123*c
mve1e23                = lambda a, b, c : a + e1*b + e23*c
mve2e13e0123           = lambda a, b, c, d : a + e2*b + e13*c + e0123*d




@pytest.fixture(scope="session", autouse=True)
def cuda_ctx():
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    yield
    ctx.pop()

CASES = [
    ("v3__v3",             mve1e2e3, mve1e2e3, 3, 3, 4, 64, 1),
    ("v4__v4",             mve0e1e2e3, mve0e1e2e3, 5, 5, 11, 64, 1),
    ("biv6__biv6",         mve01e02e03e12e13e23, mve01e02e03e12e13e23, 7, 7, 8, 64, 1),
    ("e123__e123",         mve123, mve123, 2, 2, 2, 64, 1),
    ("triv4__triv4",       mve012e013e023e123, mve012e013e023e123, 5, 5, 8, 64, 1),
    # ("v3__biv6",           mve1e2e3, mve01e02e03e12e13e23, 4, 7, 15, 64, 1),
    ("e01e23__e03e12",     mve01e23, mve03e12, 3, 3, 7, 64, 1),
    ("e0e123__e1e23",      mve0e123, mve1e23, 3, 3, 7, 64, 1),
    ("e2e13e0123__e0e1e2e01", mve2e13e0123, mve0e1e2e01, 4, 5, 15, 64, 1),
    ("e0123__e0123",       mve0123, mve0123, 2, 2, 2, 64, 1),
    ("e0123__e02e12",      mve0123, mve02e12, 2, 3, 5, 64, 1),
    ("e1e23__triv4",       mve1e23, mve012e013e023e123, 3, 5, 11, 64, 1),
    ("v4__e123",           mve0e1e2e3, mve123, 5, 2, 10, 64, 1),
    ("biv6__e123",         mve01e02e03e12e13e23, mve123, 7, 2, 14, 64, 1),
    ("e0e1e2e01__e1e23",   mve0e1e2e01, mve1e23, 5, 3, 11, 64, 1),
    ("e2e13e0123__triv4",  mve2e13e0123, mve012e013e023e123, 4, 5, 11, 64, 1),
    ("scalar__scalar",     mv_scalar, mv_scalar, 1, 1, 1, 64, 1),
    ("v4__e0e1e2e01",      mve0e1e2e3, mve0e1e2e01, 5, 5, 13, 64, 1),
    ("e123__e02e12",       mve0e1e2e01,    mv_scalar,   5, 1, 5, 64, 1),
    ("e123__e02e122",       mve023,    mv_scalar,   2, 1, 2, 64, 1),
    ("e023__e02e12",       mve023,    mve02e12,   2, 3, 5, 64, 1),

    ("e12__e12",       mve12,    mve12,    2, 2, 2, 4096, 16),
    ("e02__e02",       mve02,    mve02,    2, 2, 2, 64, 1),
    ("e01__e01",       mve01,    mve01,    2, 2, 2, 128, 4),
    ("e1e12__e0e01",   mve1e12,  mve0e01,  3, 3, 7, 64, 1),
    ("full__full",     mvfull,   mvfull,   4, 4, 4, 128, 4),
    ("scalar__full",   lambda a: a*e1*e1,  mvfull, 1, 4, 4, 64, 1),
    ("full__scalar",   mvfull,  lambda a: a*e1*e1, 4, 1, 4, 64, 1),
    ("complete__complete", mve0e1e2e01e02e12e012, mve0e1e2e01e02e12e012, 8, 8, 8, 128, 1),

    ("e0__e0",         mve0,     mve0,     2, 2, 2, 64, 1),
    ("e1__e1",         mve1,     mve1,     2, 2, 2, 64, 1),
    ("full__e0e1e2",   mvfull,   mve0e1e2, 4, 4, 8, 4096, 1),

    ("e012__e01",      mve012,   mve01,    2, 2, 3, 1024, 8),
    ("e012__full",     mve012,   mvfull,   2, 4, 6, 64, 1),

    ("e01e012__e01e012",     mve01e012, mve01e012,    3, 3, 3, 512, 1),
    ("full_e012__full_e012", mvfull_e012, mvfull_e012, 5, 5, 6, 64, 1),
    ("e0e1e2__e01e02e012",   mve0e1e2,  mve01e02e012, 4, 4, 7, 64, 1),

    ("e0e012__e0e012",   mve0e012, mve0e012, 3, 3, 3, 64, 1),
    ("e1e012__e1e012",   mve1e012, mve1e012, 3, 3, 4, 512, 1),
    ("e0e012__full",     mve0e012, mvfull,   3, 4, 6, 4096, 4),
    ("e1e012__full",     mve1e012, mvfull,   3, 4, 8, 8192, 1),

    ("e0e12__e0e12",         mve0e12,   mve0e12,   3, 3, 4, 512, 1),
    ("e0e1e12__e1e2e12",     mve0e1e12, mve1e2e12, 4, 4, 8, 128, 4),
    ("e0e1e2__e01e02e12",    mve0e1e2,  mvfull,    4, 4, 8, 4096, 1),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_geo_prod(case, cuda_ctx):
    name, mv_a, mv_b, comps_a, comps_b, comps_c, N, els_per_thread = case

    mvs = [mv_a, mv_b]
    comps = [comps_a, comps_b]

    gfunc = lambda a, b : ~(a * b)

    gaTestConfig = GATestConfig(algebra=MODE,
                                mvs = mvs,
                                comps = comps,
                                geoTys= [GeometricObj.Unknown, GeometricObj.Unknown],
                                objTys = [ObjectType.Unknown, ObjectType.Unknown],
                                comps_c= comps_c, objTy_c=ObjectType.Unknown, geoTy_c=GeometricObj.Unknown,
                                func=gfunc,
                                can_to_bit_perm=np.array([0, 1, 2, 5, 3, 6, 8, 11, 4, 7, 9, 12, 10, 13, 14, 15]),
                                bit_to_can_perm=np.array([0, 1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15]),
                                N = N,
                                els_per_thread= els_per_thread,
                                saving_path="numerical/matrices",
                                compile_script="./numerical/scripts/pga3d/compile_case.sh",
                                mlir_op_name="geo_prod")

    matrixGen = MatrixGenerator(gaTestConfig)
    result_host, expected = run_ga_kernel_test(gaTestConfig, matrixGen)
    
    if DEBUG and not np.allclose(result_host, expected, atol=1e-4):
        print_debug_info(result_host, expected, comps_c, N)

    assert(np.allclose(result_host, expected, atol=1e-5))

CASES_ROTATE = [
    ("N32_1",       mv_line,    mv_scalar,    6, 1, 7,   64,   1),
    ("N64_1",       mv_point,    mv_scalar,    3, 1, 4,   64,   1),
    ("N64_2",       mv_point,    mv_scalar,    3, 1, 4,   64,   2),
    ("N128_1",      mv_point,    mv_scalar,    3, 1, 4,  128,   1),
    ("N128_2",      mv_point,    mv_scalar,    3, 1, 4,  128,   2),
    ("N128_4",      mv_point,    mv_scalar,    3, 1, 4,  128,   4),
    ("N256_1",      mv_point,    mv_scalar,    3, 1, 4,  256,   1),
    ("N256_2",      mv_point,    mv_scalar,    3, 1, 4,  256,   2),
    ("N256_4",      mv_point,    mv_scalar,    3, 1, 4,  256,   4),
    ("N256_8",      mv_point,    mv_scalar,    3, 1, 4,  256,   8),
    ("N512_1",      mv_point,    mv_scalar,    3, 1, 4,  512,   1),
    ("N512_2",      mv_point,    mv_scalar,    3, 1, 4,  512,   2),
    ("N512_4",      mv_point,    mv_scalar,    3, 1, 4,  512,   4),
    ("N512_8",      mv_point,    mv_scalar,    3, 1, 4,  512,   8),
    ("N512_16",     mv_point,    mv_scalar,    3, 1, 4,  512,  16),
    ("N1024_1",     mv_point,    mv_scalar,    3, 1, 4, 1024,   1),
    ("N1024_2",     mv_point,    mv_scalar,    3, 1, 4, 1024,   2),
    ("N1024_4",     mv_point,    mv_scalar,    3, 1, 4, 1024,   4),
    ("N1024_8",     mv_point,    mv_scalar,    3, 1, 4, 1024,   8),
    ("N1024_16",    mv_point,    mv_scalar,    3, 1, 4, 1024,  16),
]



@pytest.mark.parametrize("case", CASES_ROTATE, ids=[c[0] for c in CASES_ROTATE])
def test_rotate(case, cuda_ctx):
    name , mv_a, mv_b, comps_a, comps_b, comps_c, N, els_per_thread = case

    layout = generate_layout(N, els_per_thread)
    total_threads = N // els_per_thread
    num_blocks = max(total_threads//(256), 1)
    num_threads = min(total_threads, 256)

    #todo: cache it if neccesary
    c_default_indices, a_default_indices, b_default_indices= generate_geo_prods_matrices(N, mv_a, mv_b, comps_a, comps_b, comps_c)
    result_host = np.zeros(N * comps_c, dtype=np.float32)
    
    # s = 0000, e0 = 0001, e1 = 0010, e2 = 0100, e3 = 1000, 4
    # e01 = 0011, e02 = 0101, e03 = 1001, e12 = 0110, e13 = 1010, e23 = 1100, 10
    # e012 = 0111, e013 = 1011, e023 = 1101, e123 = 1110, e0123 = 1111

    # s = 0000, e0 = 0001, e1 = 0010, e01 = 0011, e2 = 0100, e02 = 0101, e12 = 0110 6
    # e012 = 0111, e3 = 1000, e03 = 1001, e13 = 1010, e013 = 1011, e23 = 1100,  e023 = 1101 13
    # e123 = 1110, e0123 = 1111

    #change from canonical to bitmap layout
    np_mask_perm = np.array([0, 1, 2, 5, 3, 6, 8, 11, 4, 7, 9, 12, 10, 13, 14, 15])
    # change from bitmap layout to canonical
    inv_np_mask_perm = np.array([0, 1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15])

    adapted_mask_a = inv_np_mask_perm[a_default_indices].argsort()
    adapted_mask_b = inv_np_mask_perm[b_default_indices].argsort()

    A = np.load("numerical/matrices/matrix_a.npz")['arr_0']
    B = np.load("numerical/matrices/matrix_b.npz")['arr_0']

    # change from canonical -> bitmap layout, see explanation in cliffOps.cpp
    A = A.reshape(comps_a, N)[adapted_mask_a, :]
    A = A.flatten()

    B = B.reshape(comps_b, N)[adapted_mask_b, :]
    B = B.flatten()

    expected = np.load("numerical/matrices/matrix_c.npz")['arr_0']
   
    a_sample_coeffs = [random.randint(2, 399) for _ in range(comps_a)]
    b_sample_coeffs = [random.randint(2, 399) for _ in range(comps_b)]
    a_sample = mv_a(*a_sample_coeffs) ; b_sample = mv_b(*b_sample_coeffs)
    c_sample = (a_sample * b_sample)
   
    mask_mv_a = to_mask(a_sample.as_array()[np_mask_perm], comps_a)
    mask_mv_b = to_mask(b_sample.as_array()[np_mask_perm], comps_b)
    mask_mv_c = to_mask(c_sample.as_array()[np_mask_perm], comps_c)

    # generate ptx for case
    subprocess.run(["./numerical/scripts/pga3d/compile_case.sh", str(mask_mv_a), str(mask_mv_b), str(mask_mv_c), str(N), layout, "output.ptx"])

    with open("output.ptx", "r") as f:
        ptx = f.read()

    func, dev_matrices, result_dev = init_device(ptx, [A, B], result_host, "geo_prod")
    A_dev, B_dev = dev_matrices

    func(A_dev, B_dev, result_dev,
     block=(num_threads, 1, 1),
     grid=(num_blocks, 1, 1))
    
    cuda.memcpy_dtoh(result_host, result_dev)
    
    bitmap_c_indices = np.sort(inv_np_mask_perm[c_default_indices])
    adapted_mask_c = np_mask_perm[bitmap_c_indices].argsort()

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
