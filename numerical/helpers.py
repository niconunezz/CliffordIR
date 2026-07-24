import numpy as np
import pycuda.driver as cuda
from dataclasses import dataclass
from typing import Callable
import math
from enum import Enum
import subprocess


class ObjectType(Enum):
    Euclidean = 1
    Ideal = 2
    Unknown = 3

class GeometricObj(Enum):
    Scalar = 0
    Plane = 1
    Line = 2
    Point = 3
    Motor = 4
    Unknown = 5

@dataclass(frozen=True)
class GATestConfig:
    algebra : str
    mvs: list[Callable]              
    comps: list[int]             
    objTys: list[ObjectType]
    geoTys: list[GeometricObj]
    comps_c: int                     
    objTy_c : ObjectType
    geoTy_c : GeometricObj
    func: Callable                   
    can_to_bit_perm: np.ndarray       
    bit_to_can_perm: np.ndarray       
    N: int
    els_per_thread: int
    saving_path : str
    compile_script: str
    mlir_op_name : str



def to_mask(arr, comps):
    out = 0
    nnz = 0
    for i, el in enumerate(arr):
        if el != 0:
            out += 2**i
            nnz += 1
    
    assert (nnz == comps), "Not valid example" 
    return out


def init_device(ptx, matrices, result, func_name):
    mod  = cuda.module_from_buffer(ptx.encode())
    # device init
    
    dev_matrices = [cuda.mem_alloc(m.nbytes) for m in matrices]
    result_dev = cuda.mem_alloc(result.nbytes)
    for dev_m, m in zip(dev_matrices, matrices) :  cuda.memcpy_htod(dev_m, m)

    func = mod.get_function(func_name)
    return func, dev_matrices, result_dev

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


def get_masks(mvs, comps, comps_c, func, can_to_bit_perm):
    sample_coeffs = [mv(*np.random.randn((comp))) for mv, comp in zip(mvs, comps)]
    c_sample = func(*sample_coeffs)
    masks = []
    for sample, comp in zip(sample_coeffs, comps):
        masks.append(str(to_mask(sample.as_array()[can_to_bit_perm], comp)))
    masks.append(str(to_mask(c_sample.as_array()[can_to_bit_perm], comps_c)))

    return masks
    

def run_bash(num_els, els_per_thread, masks, bash_file_path):
    layout = generate_layout(num_els, els_per_thread)
    print(f"layout : {layout}")
    subprocess.run([bash_file_path, *masks, str(num_els), layout, "output.ptx"])

# change from canonical -> bitmap layout, see explanation in cliffOps.cpp
def canonical_to_bitmap(matrices, c_default_indices, in_default_indices, can_to_bit, bit_to_can):
    in_adapted_masks = [bit_to_can[default_indices].argsort() for default_indices in in_default_indices]
    permuted_matrices = [matrix[adapted_mask, :].flatten() for (matrix, adapted_mask) in zip(matrices, in_adapted_masks)]

    bitmap_c_indices = np.sort(bit_to_can[c_default_indices])
    adapted_mask_c = can_to_bit[bitmap_c_indices].argsort()
    
    return permuted_matrices, adapted_mask_c
    
def run_ga_kernel_test(config, matrixGen):
    
    num_mvs = len(config.mvs)
    masks = get_masks(config.mvs, config.comps, config.comps_c, config.func, config.can_to_bit_perm)
    run_bash(config.N, config.els_per_thread, masks, config.compile_script)
    c_default_indices, in_default_indices = matrixGen.generate_matrices()
    matrices = [np.load(f"{config.saving_path}/matrix_{i}.npz")['arr_0'] for i in range(num_mvs)]
    
    # change from canonical -> bitmap layout, see explanation in cliffOps.cpp
    permuted_matrices, adapted_mask_c = canonical_to_bitmap(matrices, c_default_indices, in_default_indices, config.can_to_bit_perm, config.bit_to_can_perm)

    with open("output.ptx", "r") as f:
        ptx = f.read()
    result_host = np.zeros(config.N * config.comps_c, dtype=np.float32)
    func, dev_matrices, result_dev = init_device(ptx, permuted_matrices, result_host, config.mlir_op_name)

    total_threads = config.N // config.els_per_thread
    num_blocks = max(total_threads//(256), 1)
    num_threads = min(total_threads, 256)

    func(*dev_matrices, result_dev,
     block=(num_threads, 1, 1),
     grid=(num_blocks, 1, 1))
    
    cuda.memcpy_dtoh(result_host, result_dev)
    
    expected = np.load(f"{config.saving_path}/matrix_c.npz")['arr_0']
    # change from bitmap layout -> canonical 
    result_host = result_host.reshape(config.comps_c, config.N)[adapted_mask_c, :]
    return result_host, expected

def print_debug_info(result_host, expected, comps_c, N):
    diff = np.where(~np.isclose(result_host, expected, atol=1e-4))
    print(f"Kernel outcome : {result_host.reshape(comps_c, N)[:, 0]}")
    print(f"Expected outcome : {expected.reshape(comps_c, N)[:, 0]}")
    print(f"✗ INCORRECTO en {len(diff[0]) / (comps_c * N)} de los índices")
    print(f"✗ INCORRECTO en índices: {diff}")
    
    print(f"  got:      {result_host[diff]}")
    print(f"  expected: {expected[diff]}")