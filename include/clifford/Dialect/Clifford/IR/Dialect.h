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

    uint64_t fillWithGospersHack(int k, int n);
    bool checkWithGospersHack(int k, int n, uint64_t mask);
    uint64_t getResultMask(uint64_t lhsMask, uint64_t  rhsMask, unsigned p, unsigned q, unsigned r);
    uint32_t getDegree(uint64_t mask, CliffordAlgebraAttr space);

    int reorderSign(uint64_t a, uint64_t b);

    int metricSign(uint64_t a, uint64_t b, int p, int q, int r);

} // end namespace cliff 
} // end namespace mlir 



#endif