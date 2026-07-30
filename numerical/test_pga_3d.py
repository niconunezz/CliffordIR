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
mv_rotor = lambda a, b, c, d, e, f, g : a + b* e01 + c*e02 + d*e03 + e*e12 + f*e13 + g*e23
mv_point = lambda b, c, d, e : b*e012  + c*e013 + d*e023 + e*e123

mv_line = lambda a, b, c, d, e, f : a* e01 + b*e02 + c*e03 + d*e12 + e*e13 + f*e23
mv_e12 = lambda a : a*e12
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
    ("point__point",             mv_point, mv_point, 4, 4, 4, 64, 1),
    ("rotor__point",             mv_rotor, mv_point, 7, 4, 8, 64, 1),
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
    ("full__full",     mvfull,   mvfull,   4, 4, 4, 128, 4),
    ("scalar__full",   lambda a: a*e1*e1,  mvfull, 1, 4, 4, 64, 1),
    ("full__scalar",   mvfull,  lambda a: a*e1*e1, 4, 1, 4, 64, 1),
    ("complete__complete", mve0e1e2e01e02e12e012, mve0e1e2e01e02e12e012, 8, 8, 8, 128, 1),

    ("e0__e0",         mve0,     mve0,     2, 2, 2, 64, 1),
    ("e1__e1",         mve1,     mve1,     2, 2, 2, 64, 1),
    ("full__e0e1e2",   mvfull,   mve0e1e2, 4, 4, 8, 4096, 1),

    ("e012__full",     mve012,   mvfull,   2, 4, 6, 64, 1),

    ("e01e012__e01e012",     mve01e012, mve01e012,    3, 3, 3, 512, 1),
    ("full_e012__full_e012", mvfull_e012, mvfull_e012, 5, 5, 6, 64, 1),

    ("e0e012__e0e012",   mve0e012, mve0e012, 3, 3, 3, 64, 1),
    ("e1e012__e1e012",   mve1e012, mve1e012, 3, 3, 4, 512, 1),
    ("e0e012__full",     mve0e012, mvfull,   3, 4, 6, 4096, 4),
    ("e1e012__full",     mve1e012, mvfull,   3, 4, 8, 8192, 1),

    ("e0e12__e0e12",         mve0e12,   mve0e12,   3, 3, 4, 512, 1),
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
    ("N32_1",       [mv_line,    mv_scalar],    [6, 1], 7,   64,   1),
    ("N64_1",       [mv_line,    mv_scalar],    [6, 1], 7,   64,   1),
    ("N64_2",       [mv_line,    mv_scalar],    [6, 1], 7,   64,   2),
    ("N128_1",      [mv_line,    mv_scalar],    [6, 1], 7,  128,   1),
    ("N128_2",      [mv_line,    mv_scalar],    [6, 1], 7,  128,   2),
    ("N128_4",      [mv_line,    mv_scalar],    [6, 1], 7,  128,   4),
    ("N256_1",      [mv_line,    mv_scalar],    [6, 1], 7,  256,   1),
    ("N256_2",      [mv_line,    mv_scalar],    [6, 1], 7,  256,   2),
    ("N256_4",      [mv_line,    mv_scalar],    [6, 1], 7,  256,   4),
    ("N256_8",      [mv_line,    mv_scalar],    [6, 1], 7,  256,   8),
    ("N512_1",      [mv_line,    mv_scalar],    [6, 1], 7,  512,   1),
    ("N512_2",      [mv_line,    mv_scalar],    [6, 1], 7,  512,   2),
    ("N512_4",      [mv_line,    mv_scalar],    [6, 1], 7,  512,   4),
    ("N512_8",      [mv_line,    mv_scalar],    [6, 1], 7,  512,   8),
    ("N512_16",     [mv_line,    mv_scalar],    [6, 1], 7,  512,  16),
    ("N1024_1",     [mv_line,    mv_scalar],    [6, 1], 7, 1024,   1),
    ("N1024_2",     [mv_line,    mv_scalar],    [6, 1], 7, 1024,   2),
    ("N1024_4",     [mv_line,    mv_scalar],    [6, 1], 7, 1024,   4),
    ("N1024_8",     [mv_line,    mv_scalar],    [6, 1], 7, 1024,   8),
    ("N1024_16",    [mv_line,    mv_scalar],    [6, 1], 7, 1024,  16),
]





@pytest.mark.parametrize("case", CASES_ROTATE, ids=[c[0] for c in CASES_ROTATE])
def test_rotate(case, cuda_ctx):
    name , mvs, comps, comps_c, N, els_per_thread = case
    
    gfunc = lambda mv_a, mv_alpha : math.e**(mv_alpha * mv_a)
    
    gaTestConfig = GATestConfig(algebra="pga3d",
                                mvs = mvs,
                                comps = comps,
                                geoTys= [GeometricObj.Line, GeometricObj.Scalar],
                                objTys = [ObjectType.Euclidean, ObjectType.Unknown],
                                comps_c= comps_c, objTy_c=ObjectType.Unknown, geoTy_c=GeometricObj.Motor,
                                func=gfunc,
                                can_to_bit_perm=np.array([0, 1, 2, 5, 3, 6, 8, 11, 4, 7, 9, 12, 10, 13, 14, 15]),
                                bit_to_can_perm=np.array([0, 1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15]),
                                N = N,
                                els_per_thread= els_per_thread,
                                saving_path=f"numerical/matrices/pga3d/rotation",
                                compile_script="./numerical/scripts/pga3d/compile_rotation.sh",
                                mlir_op_name="rotation")
    
    matrixGen = MatrixGenerator(gaTestConfig)
    result_host, expected = run_ga_kernel_test(gaTestConfig, matrixGen)
        
    if DEBUG and not np.allclose(result_host, expected, atol=1e-4):
        print_debug_info(result_host, expected, comps_c, N)



    assert(np.allclose(result_host, expected, atol=1e-5))


CASES_ROTATE_APPL = [
    ("N64_ELS1",          64,        1),
    ("N64_ELS2",          64,        2),

    ("N128_ELS1",        128,        1),
    ("N128_ELS4",        128,        4),

    ("N256_ELS2",        256,        2),
    ("N256_ELS8",        256,        8),

    ("N512_ELS4",        512,        4),
    ("N512_ELS16",       512,       16),

    ("N1024_ELS1",      1024,        1),
    ("N1024_ELS32",     1024,       32),

    ("N2048_ELS4",      2048,        4),
    ("N2048_ELS32",     2048,       32),

    ("N4096_ELS8",      4096,        8),
    ("N4096_ELS32",     4096,       32),

    ("N8192_ELS16",     8192,       16),
    ("N8192_ELS32",     8192,       32),

    ("N65536_ELS8",    65536,        8),
    ("N65536_ELS32",   65536,       32),

    ("N1048576_ELS16",1048576,      16),
    ("N1048576_ELS32",1048576,      32),
]


@pytest.mark.parametrize("case", CASES_ROTATE_APPL, ids=[c[0] for c in CASES_ROTATE_APPL])
def test_rotate_appl(case, cuda_ctx):
    name, N, els_per_thread = case

    mv_x = mv_line; mv_y = mv_point; mv_alpha = mv_scalar
    comps_x = 6; comps_y = 4; comps_alpha = 1; comps_c = 4
    comps = [comps_x, comps_y, comps_alpha]
    mvs = [mv_x, mv_y, mv_alpha]

    def gfunc(mv_x, mv_y, mv_alpha):
        R = math.e**(mv_alpha*mv_x)
        return R*mv_y*~R


    gaTestConfig = GATestConfig(algebra=MODE,
                                mvs = mvs,
                                comps = comps,
                                geoTys= [GeometricObj.Line, GeometricObj.Point, GeometricObj.Scalar],
                                objTys = [ObjectType.Euclidean, ObjectType.Euclidean, ObjectType.Unknown],
                                comps_c= comps_c, objTy_c=ObjectType.Unknown, geoTy_c=GeometricObj.Point,
                                func=gfunc,
                                can_to_bit_perm=np.array([0, 1, 2, 5, 3, 6, 8, 11, 4, 7, 9, 12, 10, 13, 14, 15]),
                                bit_to_can_perm=np.array([0, 1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15]),
                                N = N,
                                els_per_thread= els_per_thread,
                                saving_path=f"numerical/matrices/{MODE}/rotation_appl/{N}",
                                compile_script="./numerical/scripts/pga3d/compile_end_to_end.sh",
                                mlir_op_name="rotation")

    matrixGen = MatrixGenerator(gaTestConfig)
    result_host, expected = run_ga_kernel_test(gaTestConfig, matrixGen)

    if DEBUG and not np.allclose(result_host, expected, atol=1e-4):
        print_debug_info(result_host, expected, comps_c, N)

    assert(np.allclose(result_host, expected, atol=1e-4))

