from .._C.libclifford import driver
import torch

class GPUDriver:
    def __init__(self, compiled_asm):
        self.compiled_asm = compiled_asm


    def process_tensors(self, args):
        # for arg in args:
            # assert(isinstance(arg, torch.Tensor)), "for now only torch.Tensor arguments are valid"
        return [a.data_ptr() for a in args]
    
    def _run(self, fn_name, smemBytes, signature, args):

        total_threads = args[0].shape[-1]
        num_blocks = max(total_threads//(256), 1)
        num_threads = min(total_threads, 256)
        stream = torch.cuda.current_stream().cuda_stream
        arg_ptrs = self.process_tensors(args)
        grid = [num_blocks, 1, 1]
        blockDim = [num_threads, 1, 1]

        driver.launch_kernel(grid, blockDim, self.compiled_asm.get_assembly(), fn_name, smemBytes, signature, arg_ptrs, stream)
        