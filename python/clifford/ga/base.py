from .._C.libclifford import ir
from enum import Enum
import torch


class CliffordAlgebra:
    def __init__(self, p, q, r):
        self.p = p
        self.q = q
        self.r = r

    def get_algebra_signature(self):
        return (self.p, self.q, self.r)

    def to_ir(self, builder):
        return builder.get_algebra(self.p, self.q, self.r)

class ObjectKind(Enum):
    Euclidean = 1
    Ideal = 2
    Unknown = 3

class GeometricKind(Enum):
    Point = 1
    Line = 2
    Plane = 3

class GeometricElement:
    def __init__(self, geo_kind, obj_kind, normalized, algebra):
        self.geo_kind = geo_kind
        self.obj_kind = obj_kind
        self.normalized = normalized
        self.algebra = algebra

    def get_object_kind(self) -> GeometricKind:
        return self.geo_kind


class scalar_dtype:
    FLOAT_DTYPES = ['fp32']
    
    def __init__(self, name):
        self.name = name

    def to_ir(self, builder):
        if self.name == "fp32":
            return builder.get_f32ty()

fp32 = scalar_dtype("fp32")

def get_tensor_dtype(dtype : str):
    dtype = str(dtype)
    return dtype.split(".")[-1]

def torch_to_cliff_dtypes(dtype : str):
    lookup = {"float32" : "fp32"}
    return lookup[dtype]

class multivector:
    def __init__(self, mask, data, algebra : CliffordAlgebra):
        self.data = data
        assert(isinstance(data, torch.Tensor)), "data should be a torch.tensor"
        self.data_ptr = data.data_ptr
        self.dtype = get_tensor_dtype(data.dtype)
        self.mask = int(mask)
        self.algebra = algebra
        self.shape = tuple([shape for shape in data.size()])
        self.ir_value = None

    def to_ir(self, builder, str_to_ty_func):
        ir_scalar_dtype = str_to_ty_func(self.dtype).to_ir(builder)
        ir_algebra = self.algebra.to_ir(builder)
        mv_ty = builder.get_multivector_ty(self.mask, ir_scalar_dtype, ir_algebra)
        self.ir_value = builder.get_ranked_tensor_ty(self.shape[1:], mv_ty)
        return self.ir_value

    def __repr__(self):
        return f"tensor<{self.shape}xmultivector<{self.mask}, {self.dtype}>>"

# clifford ops

def rotate(x, angle, frontend):
    return frontend.rotate(x, angle)
