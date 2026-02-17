#ifndef CLIFFORD_DIALECT_CLIFFORD_IR_DIALECT_H
#define CLIFFORD_DIALECT_CLIFFORD_IR_DIALECT_H

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/DialectImplementation.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h.inc"
#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

#define GET_ATTRDEF_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffOps.h.inc"




#endif