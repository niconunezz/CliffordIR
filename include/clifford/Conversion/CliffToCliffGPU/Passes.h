#ifndef CLIFFORD_CONVERSION_CLIFFTOCLIFFGPU_PASSES_H
#define CLIFFORD_CONVERSION_CLIFFTOCLIFFGPU_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define GEN_PASS_DECL_CONVERTCLIFFTOCLIFFGPU
#include "clifford/Conversion/CliffToCliffGPU/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "clifford/Conversion/CliffToCliffGPU/Passes.h.inc"

#endif // CLIFFORD_CONVERSION_CLIFFTOCLIFFGPU_PASSES_H