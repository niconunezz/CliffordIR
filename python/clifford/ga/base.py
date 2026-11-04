from .._C.libclifford import ir
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Tuple, List
from abc import ABC, abstractmethod
import torch

@dataclass(frozen=True)
class CliffordAlgebra:
    p: int
    q: int
    r: int

    def get_algebra_signature(self):
        return (self.p, self.q, self.r)
    
    def to_ir(self, builder):
        return builder.get_algebra(self.p, self.q, self.r)


class Type(ABC):

    @abstractmethod
    def to_ir(self, builder):
        ...


@dataclass(frozen=True)
class ScalarType(Type):
    name : str
    
    def to_ir(self, builder):
        if self.name == "f32":
            return builder.get_f32ty()

        else:
            raise NotImplementedError(f"type {self.name} is not supported")

@dataclass(frozen=True)
class MultivectorType(Type):
    mask : int
    algebra : CliffordAlgebra
    dtype : Type

    def to_ir(self, builder):
        ir_scalar_dtype = self.dtype.to_ir(builder)
        algebra = self.algebra
        mask = self.mask
        ir_algebra = algebra.to_ir(builder)
        return builder.get_multivector_ty(mask, ir_scalar_dtype, ir_algebra)



@dataclass(frozen=True)
class PointerType(Type):
    dtype : Type

    def to_ir(self, builder):
        return builder.get_pointer_ty(self.dtype.to_ir(builder))


class Tensor:
    def __init__(self, type, shape, data_ptr, data=None):
        self.type : Type = type
        self.data = data
        self.shape = shape
        self.data_ptr = data_ptr

    def to_ir(self, builder):
        return builder.get_ranked_tensor_ty(self.shape, self.type.to_ir(builder))
        
    def __repr__(self):
        return f"tensor<{self.shape}x{self.type}>"


# types

fp32 = ScalarType("f32")

# clifford ops

def rotate(x, angle, frontend):
    return frontend.rotate(x, angle)

def geo_prod(x, y, frontend):
    return frontend.geo_prod(x, y)

def reverse(x, frontend):
    return frontend.reverse(x)

def inner_prod(x, y, frontend):
    return frontend.inner_prod(x, y)

def get_grade(x, grade, frontend):
    return frontend.get_grade(x, grade)