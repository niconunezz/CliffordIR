from .._C.libclifford import ir, passes, llvm
import time
from ..runtime.cache import get_cache_key, CacheManager
import hashlib
from ..runtime.driver import cuda_driver


class ASTStruct:
    def __init__(self, fn, signature):
        self.fn = fn
        self.name = fn.fn_name
        self.signature = signature

    def to_ir(self, context):
        from .codegen import ast_to_cliff
        return ast_to_cliff(self.fn, self, context)

    def hash(self):
        fn_hash = str(self.fn.compute_key)
        for _, v in sorted(self.signature.items()):
            fn_hash += f"-{v}"

        return fn_hash
        

class CompiledKernel:
    def __init__(self, struct, hash, group_metadata):
        self.struct = struct
        self.hash = hash 
        self.group_metadata = group_metadata
        self._run = None

    def get_assembly(self):
        return self.asm

    def run_setup(self):
        launcher_cls = cuda_driver.launcher_cls
        self._run = launcher_cls(self.struct)

    @property
    def run(self):
        if not self._run:
            self.run_setup()

    def run_kernel(self, smemBytes, args):
        if self._run is None:
            self.run_setup()

        ptx_path = self.group_metadata[f'{self.struct.name}.ptx']

        try:
            with open(ptx_path, "r") as f:
                ptx = f.read()
                self._run(smemBytes, ptx, args)
        except Exception as E:
            raise E


    

    
class Backend:
    def __init__(self):
        ...

    @staticmethod
    def make_cliff(mod, ctx):
        pm = ir.PassManager(ctx)
        passes.cliff_passes.create_rewrite_exponential_pass(pm)
        passes.cliff_passes.create_rewrite_sandwich_pass(pm)
        passes.cliff_passes.create_geometric_type_conversion_pass(pm)
        
        pm.run(mod)
        return mod

    @staticmethod
    def make_clg(mod, ctx):
        pm = ir.PassManager(ctx)
        passes.cliff_to_clg.create_cliff_to_cliffgpu(pm)
        
        pm.run(mod)

        return mod

    @staticmethod
    def make_llir(mod, ctx):
        
        # clg -> llir
        pm = ir.PassManager(ctx)
        passes.llvm.create_cliffgpu_to_llvm(pm)

        pm.run(mod)
        # llir(mlir) -> llvm(LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.translate_mod(mod, context)

        # optimize
        # todo: add more customization
        llvm.optimize_mod(llvm_mod)
        
        out = str(llvm_mod)

        del llvm_mod
        del context
        return out

    @staticmethod
    def make_ptx(llvm_ir, _):
        cpu= "sm_80"
        features = "+ptx70"
        mod = llvm.llvm_to_ptx(llvm_ir, cpu, features)
        return mod

    @staticmethod
    def get_states():
        return [(".cliff" , Backend.make_cliff),
                (".clg" , Backend.make_clg),
                (".llir" , Backend.make_llir),
                (".ptx" , Backend.make_ptx)]

class CompileTimer:
    def __init__(self):
        self.start = time.monotonic()
        self.laps = [("start", self.start)]

    def set_lap(self, name):
        self.laps.append((name, time.monotonic()))

    @staticmethod
    def delta(start, end):
        return (end - start)

    def end(self):
        lap_names = list(self.laps)
        out = ""
        for i in range(1, len(lap_names)):
            out += f"{self.laps[i-1][0]} -> {self.laps[i][0]} : {self.delta(self.laps[i-1][1], self.laps[i][1]) * 1e3} ms\n"

        return out

    

def compile(struct, timeit=True):

    context = ir.context()
    key = get_cache_key(struct)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    cm = CacheManager(hash)

    file_name = struct.fn.fn_name

    group_metadata = cm.get_group(file_name)
    if group_metadata:
        # Cache HIT!
        print("="*50)
        print("Cache HIT!")
        print("="*50)
        return CompiledKernel(struct, hash, group_metadata["child_paths"])

    group_metadata = {}
    mod = struct.to_ir(context)
    group_metadata[f"{file_name}.source"] = cm.store(mod.str(), f"{file_name}.source")
    
    backend = Backend()
    stages = backend.get_states()

    for (ext, stage) in stages:
        
        mod = stage(mod, context)
        store_name = f"{file_name}{ext}"
        group_metadata[store_name] = cm.store(mod, store_name)

    cm.store_group(group_metadata, file_name)
    print(f"Cache failed! storing with hash:{hash}, and group_metadata {group_metadata} ")
    return CompiledKernel(struct, hash, group_metadata)
