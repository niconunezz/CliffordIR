import ast
import inspect
from .._C.libclifford import ir
from .. import ga
from ..ga import str_to_ty, CliffordFrontend


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, ctx, fn_ty, signature, builder, globals, module = None):
        self.ctx = ctx
        self.fn = None
        self.module = module
        self.signature = signature
        self.builder = builder
        self.fn_ty = fn_ty
        self.frontend = CliffordFrontend(builder)
        self.globals = globals
        self.locals = {}
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
        if hasattr(node, "ctx"):
            return node.id

    def visit_Name(self, node):
        return self.lookup_name(node.id)

    def visit_Assign(self, node):
        targets = node.targets
        assert(len(targets) == 1)
        target = targets[0]
        target_name = target.id
        value = self.visit(node.value)
        self.set_local_value(target_name, value.get_result())

    def visit_Call(self, node):
        fn = self.visit(node.func)
        args = [self.visit_arg(arg) for arg in node.args]
        return self.call_Function(node, fn, args)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        attr = getattr(node, "attr", None)
        return getattr(lhs, attr)

    def call_Function(self, node, fn, args):
        values = [self.locals[arg] for arg in args]
        return fn(*values, self.frontend)

    def visit_Return(self, node):
        ret_val = self.visit(node.value)
        self.builder.create_return([ret_val])
        

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



def ast_to_cliff(fn, context, builder, signature, globals, filename, line, col):
    
    builder.set_location(filename, line, col)
    fn_ty = FunctionType(signature.values(), [list(signature.values())[-1]])                    
    cg = CodeGenerator(context, fn_ty, signature, builder,  globals)
    tree = fn.parse()
    cg.visit(tree)
    func = cg.module
    func.dump()
    




