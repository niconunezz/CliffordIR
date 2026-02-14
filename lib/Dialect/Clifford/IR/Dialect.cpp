#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "clifford/Dialect/Clifford/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::cliff;

#define GET_TYPEDEF_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffTypes.cpp.inc"


void CliffDialect::initialize() {
        addTypes<
    #define GET_TYPEDEF_LIST
    #include "clifford/Dialect/Clifford/IR/CliffTypes.cpp.inc"
        >();
    
        addOperations<
    #define GET_OP_LIST
    #include "clifford/Dialect/Clifford/IR/CliffOps.cpp.inc"
        >();
        
}