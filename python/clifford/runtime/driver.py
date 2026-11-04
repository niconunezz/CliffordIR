from .._C.libclifford import driver
import torch

class GPUDriver:
    def __init__(self, compiled_asm):
        self.compiled_asm = compiled_asm
    


class CUDALauncher:
    def __init__(self, struct):
        self.struct = struct

    def __call__(self, smemBytes, ptx, args):
        total_threads = args[0].shape[-1]
        num_blocks = max(total_threads//(256), 1)
        num_threads = min(total_threads, 256)
        stream = torch.cuda.current_stream().cuda_stream
        arg_ptrs = [a.data_ptr for a in args]
        grid = [num_blocks, 1, 1]
        blockDim = [num_threads, 1, 1]

        name = self.struct.name
        signature = self.struct.signature
        # TODO: Use signature once more dtypes are expanded
        driver.launch_kernel(grid, blockDim, ptx, name, smemBytes, ["*fp32" for _ in range(len(args))], arg_ptrs, stream)



class CUDADriver:
    def __init__(self):
        self.launcher_cls = CUDALauncher


cuda_driver = CUDADriver()
