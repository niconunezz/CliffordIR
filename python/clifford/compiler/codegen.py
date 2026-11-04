import ast
from .._C.libclifford import ir
from ..ga import CliffordFrontend


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, ctx, fn_ty, signature, globals, file_name, def_line, def_col, module = None):
        self.ctx = ctx
        self.fn = None
        self.signature = signature
        self.builder = ir.CliffordOpBuilder(ctx)
        self.builder.set_location(file_name, def_line, def_col)
        # todo: this should be something like
        # self.register_dialects() which calls some get_dialects_for("cliffDialect")
        ir.register_dialects(ctx)
        self.fn_ty = fn_ty
        self.frontend = CliffordFrontend(self.builder)
        self.globals = globals
        self.locals = {}

        self.module = module
        if module is None:
            self.module = self.builder.create_module()


    def _init_locals(self, entry_block):
        arguments = entry_block.get_arguments()
        for (k, arg) in zip(self.signature.keys(), arguments):
            self._set_local_value(k, arg)

    def _set_local_value(self, target, value):
        self.locals[target] = value

    def lookup_name(self, name):
        for lookup in [self.locals, self.globals]:
            v = lookup.get(name, None)
            if v is not None:
                return v
    
    def visit_FunctionDef(self, node):
        if self.fn:
            raise NotImplementedError("compiled functions cannot be nested! create two separate functions")
        sym_name = node.name
        
        fn_ty = self.fn_ty.build_fn_ty(self.builder)
        visibility = "public"
        self.fn = self.builder.get_function_def(sym_name, fn_ty, visibility)
        self.module.push_back(self.fn)
        entry = self.fn.add_entry_block()
        self.builder.set_insertion_point_to_start(entry)
        self._init_locals(entry)
        self.generic_visit(node)

    def visit_arg(self, node):
        if hasattr(node, "ctx") or hasattr(node, "value"):
            return self.visit(node)
        if hasattr(node, "annotation"):
            return
        else:
            print(node)
            raise Exception("Argument type unknown")

    def visit_Constant(self, node):
        return node.value
    
    def visit_Name(self, node):
        return self.lookup_name(node.id)

    def visit_Assign(self, node):
        targets = node.targets
        assert(len(targets) == 1)
        target = targets[0]
        target_name = target.id
        value = self.visit(node.value)
        self._set_local_value(target_name, value.get_result())

    def visit_Call(self, node):
        fn = self.visit(node.func)
        args = [self.visit_arg(arg) for arg in node.args]
        return self.call_Function(node, fn, args)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        attr = getattr(node, "attr", None)
        return getattr(lhs, attr)

    def call_Function(self, node, fn, args):
        
        values = [self.locals[arg] if isinstance(arg, str) else arg for arg in args]
        return fn(*values, self.frontend)

    def visit_Return(self, node):
        ret_val = self.visit(node.value)
        self.builder.create_return([ret_val])
        

class FunctionType:
    #todo: check types on types are pirstered
    def __init__(self, inputs = [], results = []):
        self.inputs = inputs
        self.results = results

    def _to_ir_list(self, builder, types):
        return [ty.to_ir(builder) for ty in types]

    def build_fn_ty(self, builder):
        input_ir_list = self._to_ir_list(builder, self.inputs)
        result_ir_list = self._to_ir_list(builder, self.results)
        return builder.get_function_type(input_ir_list, result_ir_list)



def ast_to_cliff(fn, struct, context):
    
    fn_ty = FunctionType(struct.signature.values(), [list(struct.signature.values())[-1]])                    
    cg = CodeGenerator(context, fn_ty, struct.signature, fn.__globals__, fn.file_name, fn.def_line, fn.def_col)
    cg.visit(fn.parse())
    return cg.module
