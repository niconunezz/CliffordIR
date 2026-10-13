from .base import fp32, CliffordAlgebra, multivector, rotate, geo_prod, reverse, inner_prod
from .frontend import CliffordFrontend

__all__ = [
    "fp32",
    "CliffordAlgebra",
    "multivector",
    "rotate",
    "geo_prod",
    "reverse",
    "inner_prod"
]


def str_to_ty(name):

    types = {
        "float32" : fp32
    }

    return types[name]