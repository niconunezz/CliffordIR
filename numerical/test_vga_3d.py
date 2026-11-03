from generate_oracle import MatrixGenerator
import pytest
import numpy as np
import subprocess
import math
from clifford import Cl
import pycuda.driver as cuda
from helpers import to_mask, init_device, generate_layout
from helpers import ObjectType, GeometricObj, run_ga_kernel_test, print_debug_info, GATestConfig

DEBUG = True
MODE = "vga3d"

layout, blades = Cl(3, 0, 0, firstIdx=0)
e0 = blades['e0']
e1 = blades['e1']
e2 = blades['e2']
e01 = blades['e01']
e02 = blades['e02']
e12 = blades['e12']
e012 = blades['e012']


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

mv_e12 = lambda a : a*e12
mv_scalar = lambda a : a*e1*e1





@pytest.fixture(scope="session", autouse=True)
def cuda_ctx():
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    yield
    ctx.pop()

CASES = [

    ("e12__e12",       mve12,    mve12,    2, 2, 2, 4096, 16),
    ("e02__e02",       mve02,    mve02,    2, 2, 2, 64, 1),
    ("full__full",     mvfull,   mvfull,   4, 4, 4, 128, 4),
    ("scalar__full",   lambda a: a*e1*e1,  mvfull, 1, 4, 4, 64, 1),
    ("full__scalar",   mvfull,  lambda a: a*e1*e1, 4, 1, 4, 64, 1),
    ("complete__complete", mve0e1e2e01e02e12e012, mve0e1e2e01e02e12e012, 8, 8, 8, 128, 1),

    ("e0__e0",         mve0,     mve0,     2, 2, 2, 64, 1),
    ("e1__e1",         mve1,     mve1,     2, 2, 2, 64, 1),
    ("full__e0e1e2",   mvfull,   mve0e1e2, 4, 4, 8, 4096, 1),

    ("e012__full",     mve012,   mvfull,   2, 4, 8, 64, 1),

    ("e01e012__e01e012",     mve01e012, mve01e012,    3, 3, 4, 512, 1),
    ("full_e012__full_e012", mvfull_e012, mvfull_e012, 5, 5, 8, 64, 1),

    ("e0e012__e0e012",   mve0e012, mve0e012, 3, 3, 4, 64, 1),
    ("e1e012__e1e012",   mve1e012, mve1e012, 3, 3, 4, 512, 1),
    ("e0e012__full",     mve0e012, mvfull,   3, 4, 8, 4096, 4),
    ("e1e012__full",     mve1e012, mvfull,   3, 4, 8, 8192, 1),

    ("e0e12__e0e12",         mve0e12,   mve0e12,   3, 3, 4, 512, 1),
    ("e0e1e2__e01e02e12",    mve0e1e2,  mvfull,    4, 4, 8, 4096, 1),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_geo_prod(case, cuda_ctx):
    name, mv_a, mv_b, comps_a, comps_b, comps_c, N, els_per_thread = case

    mvs = [mv_a, mv_b]
    comps = [comps_a, comps_b]

    gfunc = lambda a, b : (a * b)

    gaTestConfig = GATestConfig(algebra=MODE,
                                mvs = mvs,
                                comps = comps,
                                geoTys= [GeometricObj.Unknown, GeometricObj.Unknown],
                                objTys = [ObjectType.Unknown, ObjectType.Unknown],
                                comps_c= comps_c, objTy_c=ObjectType.Unknown, geoTy_c=GeometricObj.Unknown,
                                func=gfunc,
                                can_to_bit_perm=np.array([0, 1, 2, 4, 3, 5, 6, 7]),
                                bit_to_can_perm=np.array([0, 1, 2, 4, 3, 5, 6, 7]),
                                N = N,
                                els_per_thread= els_per_thread,
                                saving_path="numerical/matrices/vga3d/geo_prod",
                                compile_script="./numerical/scripts/vga3d/compile_case.sh",
                                mlir_op_name="geo_prod")

    matrixGen = MatrixGenerator(gaTestConfig)
    result_host, expected = run_ga_kernel_test(gaTestConfig, matrixGen)
    
    if DEBUG and not np.allclose(result_host, expected, atol=1e-4):
        print_debug_info(result_host, expected, comps_c, N)

    assert(np.allclose(result_host, expected, atol=1e-5))