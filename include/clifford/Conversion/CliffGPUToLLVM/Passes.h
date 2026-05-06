#ifndef CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_PASSES_H
#define CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"

namespace mlir::clg
{
#define GEN_PASS_DECL_CONVERTCLIFFGPUTOLLVM
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h.inc"
    
} // namespace mlir::cliff


#endif // CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_PASSES_H