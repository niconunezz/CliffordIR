from .base import fp32, CliffordAlgebra, multivector, rotate
from .frontend import CliffordFrontend

__all__ = [
    "fp32",
    "CliffordAlgebra",
    "multivector",
    "rotate"
]


def str_to_ty(name):

    types = {
        "float32" : fp32
    }

    return types[name]