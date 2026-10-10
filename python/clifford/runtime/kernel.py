import inspect
import ast
import textwrap
import re
from ..compiler.compiler import compile, ASTStruct
from .driver import GPUDriver

def get_def_line(raw_src, starting_line_number):
    def_patt = r'def\s+\w+\('
    for idx, line in enumerate(raw_src):
        if re.match(def_patt, line):
            return starting_line_number + idx

    raise "Not fn definition found!"

def get_def_col(raw_src):
    indentation_patt = re.compile(r"^(?P<ind>[\t\s]+)def\s+", re.MULTILINE)
    match = indentation_patt.search(raw_src)
    if match:
        print(match.group("ind"))
        return 1 + len(match.group("ind"))

    return 1


class KernelCallable:
    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        raw_src_str = "".join(self.raw_src)
        src = textwrap.dedent(raw_src_str)
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
        self._src = src

        self.def_line = get_def_line(self.raw_src, self.starting_line_number)
        self.def_col = get_def_col(raw_src_str)
        self.file_name = fn.__code__.co_filename

    def parse(self):
        return ast.parse(self._src)


class KernelFunction(KernelCallable):

    def __init__(self, fn):
        super().__init__(fn)
        self.fn_name = fn.__name__
        self.__globals__ = fn.__globals__
        self.static_params = self.signature.parameters
        self.ASTStruct = ASTStruct

    def get_arg_names(self):
        return self.static_params.keys()

    def get_signature(self, args):
        arg_names = self.get_arg_names()
        return {name: ty for (name, ty) in zip(arg_names, args)}

    def _compile(self, signature):
        struct = self.ASTStruct(self, signature)
        return compile(self, struct)

    def __call__(self, *args):        
        signature = self.get_signature(args)
        
        compiled_asm = self._compile(signature)
        driver = GPUDriver(compiled_asm)
        driver._run(self.fn_name, 0, ["*fp32" for i in range(len(args))], args)




def kernel(fn):
    return KernelFunction(fn)



