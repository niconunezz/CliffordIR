import numpy as np
import pycuda.driver as cuda
import math
from enum import Enum

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

