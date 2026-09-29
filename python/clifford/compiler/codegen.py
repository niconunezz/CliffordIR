import ast
import inspect
from .._C.libclifford import ir


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, ctx, prototype, builder, module = None):
        self.ctx = ctx
        self.fn = None
        self.module = module
        self.builder = builder
        self.prototype = prototype
        if module is None:
            self.module = builder.create_module()

    def visit_FunctionDef(self, node):
        if self.fn:
            raise NotImplementedError("compiled functions cannot be nested! create two separate functions")
        sym_name = node.name
        fn_ty = self.prototype.build_fn_ty(self.builder)
        visibility = "public"
        self.fn = self.builder.get_function_def(sym_name, fn_ty, visibility)
        self.module.push_back(self.fn)
        entry = self.fn.add_entry_block()
        self.builder.set_insertion_point_to_start(entry)
        self.generic_visit(node)

    def visit_arg(self, node):
        self.generic_visit(node)

    def visit_Assign(self, node):
        ...
        self.generic_visit(node)

    def visit_Call(self, node):
        print(f"Calling {node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)

    def visit_Return(self, node):
        self.builder.create_return([])
        print("returning...")
        

class FunctionType:
    #todo: check types on types are pirstered
    def __init__(self, inputs = [], results = []):
        self.inputs = inputs
        self.results = results

    def build_fn_ty(self, builder):
        return builder.get_function_type(self.inputs, self.results)

def ast_to_cliff(fn, context, filename, line, col):
    builder = ir.CliffordOpBuilder(context)
    builder.set_location(filename, line, col)
    args = [builder.get_f32ty(), builder.get_f32ty(), builder.get_f32ty()]  
    fn_ty = FunctionType(args, [])                       
    cg = CodeGenerator(context, fn_ty, builder)
    tree = fn.parse()
    cg.visit(tree)
    func = cg.fn
    func.debug()
    




