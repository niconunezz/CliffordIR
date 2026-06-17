#ifndef CLIFFORD_DIALECT_CLIFFORDGPU_IR_TYPES_H
#define CLIFFORD_DIALECT_CLIFFORDGPU_IR_TYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"


#define GET_TYPEDEF_CLASSES
#include "clifford/Dialect/CliffGPU/IR/CliffGPUTypes.h.inc"


#endif // CLIFFORD_DIALECT_CLIFFORDGPU_IR_TYPES_H