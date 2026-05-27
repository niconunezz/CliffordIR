from generate_oracle import generate_matrices
import pytest
import numpy as np
import pycuda.driver as cuda
import subprocess
import random
from clifford import Cl

layout, blades = Cl(2, 0, 1, firstIdx=0)
e0 = blades['e0']
e1 = blades['e1']
e2 = blades['e2']
e01 = blades['e01']
e02 = blades['e02']
e12 = blades['e12']
e012 = blades['e012']

mve12     = lambda a,b   : a + b*e12
mve02     = lambda a,b   : a + b*e02
mve01     = lambda a,b   : a + b*e01
mve1e12   = lambda a,b,c : a + b*e1  + c*e12
mve0e01   = lambda a,b,c : a + b*e0  + c*e01
mve01e02  = lambda a,b,c : a + b*e01 + c*e02
mve01e12  = lambda a,b,c : a + b*e01 + c*e12
mve02e12  = lambda a,b,c : a + b*e02 + c*e12
mvfull    = lambda a,b,c,d: a + b*e01 + c*e02 + d*e12



def to_mask(arr):
    out = 0
    for i, el in enumerate(arr):
        if el != 0:
            out += 2**i
    return out

@pytest.fixture(scope="session", autouse=True)
def cuda_ctx():
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    yield
    ctx.pop()


CASES = [
    # escalar + bivector con e12
    ("e12__e12",       mve12,    mve12,    2, 2, 2),
    # escalar + bivector con e02
    ("e02__e02",       mve02,    mve02,    2, 2, 2),
    # escalar + bivector con e01
    ("e01__e01",       mve01,    mve01,    2, 2, 2),
    # mezcla e01 y e02 — ejercita términos cruzados con e0²=0
    ("e01e02__e01e02", mve01e02, mve01e02, 3, 3, 3),
    # mezcla e01 y e12 — signos no triviales
    ("e01e12__e01e12", mve01e12, mve01e12, 3, 3, 4),
    # mezcla e02 y e12
    ("e02e12__e02e12", mve02e12, mve02e12, 3, 3, 4),
    # tipos distintos: (e1+e12) * (e0+e01) — producto mixto grado 1 × grado 1+2
    ("e1e12__e0e01",   mve1e12,  mve0e01,  3, 3, 7),
    # bivector completo × bivector completo — caso general
    ("full__full",     mvfull,   mvfull,   4, 4, 4),
    # asimétrico: escalar puro × bivector completo
    ("scalar__full",   lambda a: a*e1*e1,  mvfull, 1, 4, 4),
    # asimétrico: bivector completo × escalar puro
    ("full__scalar",   mvfull,  lambda a: a*e1*e1, 4, 1, 4),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_geo_prod(case, cuda_ctx):
    N = 64
    print("running")
    name, mv_a, mv_b, comps_a, comps_b, comps_c = case
    #todo: cache it if neccesary
    generate_matrices(N, mv_a, mv_b, comps_a, comps_b, comps_c)
    result_host = np.zeros(N * comps_c, dtype=np.float32)
    
    # device init
    # ----------------------------
    A = np.load("numerical/matrices/matrix_a.npz")['arr_0']
    B = np.load("numerical/matrices/matrix_b.npz")['arr_0']

    expected = np.load("numerical/matrices/matrix_c.npz")['arr_0']

    A_dev  = cuda.mem_alloc(A.nbytes)
    B_dev = cuda.mem_alloc(B.nbytes)
    result_dev = cuda.mem_alloc(result_host.nbytes)

    cuda.memcpy_htod(A_dev, A)
    cuda.memcpy_htod(B_dev, B)
    # ----------------------------
    mask_perm = [0, 1, 2, 4, 3, 5, 6, 7]
    a_sample_coeffs = [random.randint(2, 399) for _ in range(comps_a)]
    b_sample_coeffs = [random.randint(2, 399) for _ in range(comps_b)]
    a_sample = mv_a(*a_sample_coeffs) ; b_sample = mv_b(*b_sample_coeffs)
    c_sample = (a_sample * b_sample)
    mask_mv_a = to_mask(a_sample.as_array()[mask_perm])
    mask_mv_b = to_mask(b_sample.as_array()[mask_perm])
    mask_mv_c = to_mask(c_sample.as_array()[mask_perm])

    # generate ptx for case
    subprocess.run(["./numerical/compile_case.sh", str(mask_mv_a), str(mask_mv_b), str(mask_mv_c), "output.ptx"])

    with open("output.ptx", "r") as f:
        ptx = f.read()
    
    mod  = cuda.module_from_buffer(ptx.encode())

    func = mod.get_function("geo_prod_scalar_case")
    func(A_dev, B_dev, result_dev,
     block=(N, 1, 1),
     grid=(1, 1, 1))
    
    cuda.memcpy_dtoh(result_host, result_dev)

    assert(np.allclose(result_host, expected, atol=1e-5))
