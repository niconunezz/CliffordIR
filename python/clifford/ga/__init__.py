from .base import fp32, CliffordAlgebra, multivector

__all__ = [
    "fp32",
    "CliffordAlgebra",
    "multivector"
]


def str_to_ty(name):

    types = {
        "float32" : fp32
    }

    return types[name]