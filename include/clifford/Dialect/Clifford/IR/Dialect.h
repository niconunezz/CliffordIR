#ifndef CLIFFORD_DIALECT_CLIFFORD_IR_DIALECT_H
#define CLIFFORD_DIALECT_CLIFFORD_IR_DIALECT_H

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/DialectImplementation.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h.inc"

#include "clifford/Dialect/Clifford/IR/CliffOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffAttrDefs.h.inc"

#include "clifford/Dialect/Clifford/IR/TypeInterfaces.h.inc"

#include "clifford/Dialect/Clifford/IR/CliffTypes.h"


#define GET_OP_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffOps.h.inc"


namespace mlir {
namespace cliff {

    int reorderSign(uint64_t a, uint64_t b);
    int metricSign(uint64_t a, uint64_t b, int p, int q, int r);

} // end namespace cliff 
} // end namespace mlir 



#endif