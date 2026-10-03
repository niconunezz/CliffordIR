import ast
import inspect
from .._C.libclifford import ir
from .. import ga
from ..ga import str_to_ty


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
        fn_ty = self.prototype.build_fn_ty(str_to_ty, self.builder)
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
        

class FunctionType:
    #todo: check types on types are pirstered
    def __init__(self, inputs = [], results = []):
        self.inputs = inputs
        self.results = results

    def to_ir_list(self, builder, str_to_ty_func, types):
        return [ty.to_ir(builder, str_to_ty_func) for ty in types]

    def build_fn_ty(self, str_to_ty_func, builder):
        input_ir_list = self.to_ir_list(builder, str_to_ty_func, self.inputs)
        result_ir_list = self.to_ir_list(builder, str_to_ty_func, self.results)
        return builder.get_function_type(input_ir_list, result_ir_list)





def ast_to_cliff(fn, context, builder, prototype, filename, line, col):
    
    builder.set_location(filename, line, col)

    fn_ty = FunctionType(prototype.values(), [])                    
    cg = CodeGenerator(context, fn_ty, builder)
    tree = fn.parse()
    cg.visit(tree)
    func = cg.module
    func.dump()
    




