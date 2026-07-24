import pytest
import numpy as np
import pycuda.driver as cuda
from helpers import ObjectType, GeometricObj, run_ga_kernel_test, print_debug_info
from generate_oracle import MatrixGenerator
import math
from clifford import Cl
from helpers import GATestConfig
DEBUG = True


MODE = "pga2d"
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
mv_point = lambda b, c, _ : b*e01 + c*e02 + e12
mv_scalar = lambda a : a*e1*e1


@pytest.fixture(scope="session", autouse=True)
def cuda_ctx():
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    yield
    ctx.pop()

CASES = [
    ("scalar_e0e1e2",  [mv_scalar, mve0e1e2], [1, 4], 4, 64, 1),
    ("e12__e12",       [mve12, mve12], [2, 2], 2, 128, 1),
    ("e12__e12",       [mve12, mve12], [2, 2], 2, 4096, 16),
    ("e02__e02",       [mve02, mve02], [2, 2], 2, 64, 1),
    ("e01__e01",       [mve01, mve01], [2, 2], 2, 128, 4),
    ("e01e02__e01e02", [mve01e02, mve01e02], [3, 3], 3, 64, 1),
    ("e01e12__e01e12", [mve01e12, mve01e12], [3, 3], 4, 64, 1),
    ("e02e12__e02e12", [mve02e12, mve02e12], [3, 3], 4, 4096, 4),
    ("e1e12__e0e01",   [mve1e12, mve0e01], [3, 3], 7, 64, 1),
    ("full__full",     [mvfull, mvfull], [4, 4], 4, 128, 4),
    ("scalar__full",   [lambda a: a*e1*e1, mvfull], [1, 4], 4, 64, 1),
    ("full__scalar",   [mvfull, lambda a: a*e1*e1], [4, 1], 4, 64, 1),
    ("complete__complete", [mve0e1e2e01e02e12e012, mve0e1e2e01e02e12e012], [8, 8], 8, 128, 1),

    ("e0__e0",         [mve0, mve0], [2, 2], 2, 64, 1),
    ("e1__e1",         [mve1, mve1], [2, 2], 2, 64, 1),
    ("e2__e2",         [mve2, mve2], [2, 2], 2, 64, 2),
    ("e0__e1",         [mve0, mve1], [2, 2], 4, 4096, 16),
    ("e1__e2",         [mve1, mve2], [2, 2], 4, 512, 4),
    ("e0__e2",         [mve0, mve2], [2, 2], 4, 128, 1),
    ("e0e1__e0e2",     [mve0e1, mve0e2], [3, 3], 7, 64, 2),
    ("e0e1__e1e2",     [mve0e1, mve1e2], [3, 3], 7, 8192, 4),
    ("e0e2__e1e2",     [mve0e2, mve1e2], [3, 3], 7, 4096, 4),
    ("e0e1e2__e0e1e2", [mve0e1e2, mve0e1e2], [4, 4], 7, 512, 2),
    ("e0__e01",        [mve0, mve01], [2, 2], 3, 1024, 8),
    ("e1__e12",        [mve1, mve12], [2, 2], 4, 32, 1),
    ("e2__e02",        [mve2, mve02], [2, 2], 4, 64, 1),
    ("e0e1__e12",      [mve0e1, mve12], [3, 2], 6, 64, 1),
    ("e0e2__e12",      [mve0e2, mve12], [3, 2], 6, 64, 2),
    ("e1e2__e12",      [mve1e2, mve12], [3, 2], 4, 4096, 1),
    ("e0e1e2__full",   [mve0e1e2, mvfull], [4, 4], 8, 64, 2),
    ("e12__e1",        [mve12, mve1], [2, 2], 4, 64, 1),
    ("e01__e0",        [mve01, mve0], [2, 2], 3, 512, 8),
    ("full__e0e1e2",   [mvfull, mve0e1e2], [4, 4], 8, 4096, 1),

    ("e012__scalar",   [mve012, lambda a: a*e1*e1], [2, 1], 2, 64, 1),
    ("e012__e012",     [mve012, mve012], [2, 2], 2, 64, 1),
    ("e012__e0",       [mve012, mve0], [2, 2], 3, 64, 1),
    ("e012__e1",       [mve012, mve1], [2, 2], 4, 64, 1),
    ("e012__e0e1e2",   [mve012, mve0e1e2], [2, 4], 7, 512, 1),
    ("e012__e01",      [mve012, mve01], [2, 2], 3, 1024, 8),
    ("e012__e12",      [mve012, mve12], [2, 2], 4, 8192, 1),
    ("e012__full",     [mve012, mvfull], [2, 4], 6, 64, 1),

    ("e01e012__e01e012",     [mve01e012, mve01e012], [3, 3], 3, 512, 1),
    ("e12e012__e12e012",     [mve12e012, mve12e012], [3, 3], 4, 64, 1),
    ("full_e012__full_e012", [mvfull_e012, mvfull_e012], [5, 5], 6, 64, 1),
    ("e0e1e2__e01e02e012",   [mve0e1e2, mve01e02e012], [4, 4], 7, 64, 1),

    ("e0e012__e0e012",   [mve0e012, mve0e012], [3, 3], 3, 64, 1),
    ("e1e012__e1e012",   [mve1e012, mve1e012], [3, 3], 4, 512, 1),
    ("e0e012__full",     [mve0e012, mvfull], [3, 4], 6, 4096, 4),
    ("e1e012__full",     [mve1e012, mvfull], [3, 4], 8, 8192, 1),

    ("e0e12__e0e12",         [mve0e12, mve0e12], [3, 3], 4, 512, 1),
    ("e0e12__full",          [mve0e12, mvfull], [3, 4], 6, 8192, 1),
    ("e1e01__e1e01",         [mve1e01, mve1e01], [3, 3], 4, 4096, 1),
    ("e1e02__e2e01",         [mve1e02, mve2e01], [3, 3], 7, 32, 1),
    ("e0e1e01__e0e1e01",     [mve0e1e01, mve0e1e01], [4, 4], 4, 512, 1),
    ("e0e1e12__e1e2e12",     [mve0e1e12, mve1e2e12], [4, 4], 8, 128, 4),
    ("e0e1e2__e01e02e12",    [mve0e1e2, mvfull], [4, 4], 8, 4096, 1),
]

@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_geo_prod(case, cuda_ctx):
    name, mvs, comps, comps_c, N, els_per_thread = case

    gfunc = lambda a, b : ~(a * b)

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
                                saving_path="numerical/matrices",
                                compile_script="./numerical/scripts/pga2d/compile_case.sh",
                                mlir_op_name="geo_prod")

    matrixGen = MatrixGenerator(gaTestConfig)
    result_host, expected = run_ga_kernel_test(gaTestConfig, matrixGen)
    
    if DEBUG and not np.allclose(result_host, expected, atol=1e-4):
        print_debug_info(result_host, expected, comps_c, N)

    assert(np.allclose(result_host, expected, atol=1e-5))


CASES_ROTATE = [
    ("N32_1",       [mv_point,    mv_scalar],    [3, 1], 4,   32,   1),
    ("N64_1",       [mv_point,    mv_scalar],    [3, 1], 4,   64,   1),
    ("N64_2",       [mv_point,    mv_scalar],    [3, 1], 4,   64,   2),
    ("N128_1",      [mv_point,    mv_scalar],    [3, 1], 4,  128,   1),
    ("N128_2",      [mv_point,    mv_scalar],    [3, 1], 4,  128,   2),
    ("N128_4",      [mv_point,    mv_scalar],    [3, 1], 4,  128,   4),
    ("N256_1",      [mv_point,    mv_scalar],    [3, 1], 4,  256,   1),
    ("N256_2",      [mv_point,    mv_scalar],    [3, 1], 4,  256,   2),
    ("N256_4",      [mv_point,    mv_scalar],    [3, 1], 4,  256,   4),
    ("N256_8",      [mv_point,    mv_scalar],    [3, 1], 4,  256,   8),
    ("N512_1",      [mv_point,    mv_scalar],    [3, 1], 4,  512,   1),
    ("N512_2",      [mv_point,    mv_scalar],    [3, 1], 4,  512,   2),
    ("N512_4",      [mv_point,    mv_scalar],    [3, 1], 4,  512,   4),
    ("N512_8",      [mv_point,    mv_scalar],    [3, 1], 4,  512,   8),
    ("N512_16",     [mv_point,    mv_scalar],    [3, 1], 4,  512,  16),
    ("N1024_1",     [mv_point,    mv_scalar],    [3, 1], 4, 1024,   1),
    ("N1024_2",     [mv_point,    mv_scalar],    [3, 1], 4, 1024,   2),
    ("N1024_4",     [mv_point,    mv_scalar],    [3, 1], 4, 1024,   4),
    ("N1024_8",     [mv_point,    mv_scalar],    [3, 1], 4, 1024,   8),
    ("N1024_16",    [mv_point,    mv_scalar],    [3, 1], 4, 1024,  16),
]

@pytest.mark.parametrize("case", CASES_ROTATE, ids=[c[0] for c in CASES_ROTATE])
def test_rotate(case, cuda_ctx):
    name , mvs, comps, comps_c, N, els_per_thread = case
    gfunc = lambda mv_a, mv_alpha : math.e**(mv_alpha * mv_a)

    gaTestConfig = GATestConfig(algebra=MODE,
                                mvs = mvs,
                                comps = comps,
                                geoTys= [GeometricObj.Point, GeometricObj.Scalar],
                                objTys = [ObjectType.Euclidean, ObjectType.Unknown],
                                comps_c= comps_c, objTy_c=ObjectType.Unknown, geoTy_c=GeometricObj.Motor,
                                func=gfunc,
                                can_to_bit_perm=np.array([0, 1, 2, 4, 3, 5, 6, 7]),
                                bit_to_can_perm=np.array([0, 1, 2, 4, 3, 5, 6, 7]),
                                N = N,
                                els_per_thread= els_per_thread,
                                saving_path=f"numerical/matrices/{MODE}/rotation",
                                compile_script="./numerical/scripts/pga2d/compile_rotation.sh",
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

    mv_x = mv_point; mv_y = mv_point; mv_alpha = mv_scalar
    comps_x = 3; comps_y = 3; comps_alpha = 1; comps_c = 3
    comps = [comps_x, comps_y, comps_alpha]
    mvs = [mv_x, mv_y, mv_alpha]

    def gfunc(mv_x, mv_y, mv_alpha):
        R = math.e**(mv_alpha*mv_x)
        return R*mv_y*~R


    gaTestConfig = GATestConfig(algebra=MODE,
                                mvs = mvs,
                                comps = comps,
                                geoTys= [GeometricObj.Point, GeometricObj.Point, GeometricObj.Scalar],
                                objTys = [ObjectType.Euclidean, ObjectType.Euclidean, ObjectType.Unknown],
                                comps_c= comps_c, objTy_c=ObjectType.Unknown, geoTy_c=GeometricObj.Point,
                                func=gfunc,
                                can_to_bit_perm=np.array([0, 1, 2, 4, 3, 5, 6, 7]),
                                bit_to_can_perm=np.array([0, 1, 2, 4, 3, 5, 6, 7]),
                                N = N,
                                els_per_thread= els_per_thread,
                                saving_path=f"numerical/matrices/{MODE}/rotation_appl/{N}",
                                compile_script="./numerical/scripts/pga2d/compile_end_to_end.sh",
                                mlir_op_name="rotation")

    matrixGen = MatrixGenerator(gaTestConfig)
    result_host, expected = run_ga_kernel_test(gaTestConfig, matrixGen)

    if DEBUG and not np.allclose(result_host, expected, atol=1e-4):
        print_debug_info(result_host, expected, comps_c, N)

    assert(np.allclose(result_host, expected, atol=1e-4))