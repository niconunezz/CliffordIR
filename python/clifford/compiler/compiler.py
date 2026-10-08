from .._C.libclifford import ir, passes, llvm

class ASTStruct:
    def __init__(self, fn, signature):
        self.fn = fn
        self.signature = signature

    def to_ir(self, context):
        from .codegen import ast_to_cliff
        return ast_to_cliff(self.fn, self, context)


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

    @staticmethod
    def make_clg(mod, ctx):
        pm = ir.PassManager(ctx)
        passes.cliff_to_clg.create_cliff_to_cliffgpu(pm)
        
        pm.run(mod)

    @staticmethod
    def make_llir(mod, ctx):
        
        # clg -> llir
        pm = ir.PassManager(ctx)
        passes.llvm.create_cliffgpu_to_llvm(pm)

        pm.run(mod)
        mod.dump()

        # llir(mlir) -> llvm(LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.translate_mod(mod, context)

        # llvm_mod.dump()

        


def compile(fn, struct):
   
    context = ir.context()

    mod = struct.to_ir(context)
    # mod.dump()

    backend = Backend()
    backend.make_cliff(mod, context)
    
    backend.make_clg(mod, context)

    backend.make_llir(mod, context)
    
    # mod.dump()

