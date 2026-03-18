#ifndef CLIFFORD_DIALECT_CLIFFORDGPU_IR_DIALECT_H
#define CLIFFORD_DIALECT_CLIFFORDGPU_IR_DIALECT_H

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"


#include "mlir/IR/DialectImplementation.h"
#include "clifford/Tools/StrUtils.h"

#include "clifford/Tools/LinearLayout.h"


#include "clifford/Dialect/CliffGPU/IR/Dialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "clifford/Dialect/CliffGPU/IR/CliffGPUAttrDefs.h.inc"


namespace mlir::clg {


    LinearEncodingAttr getDefaultGlobalEncoding(MLIRContext *ctx, ArrayRef<int64_t> shape);


} // namespace mlir::clg



#endif