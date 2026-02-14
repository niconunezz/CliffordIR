#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cliff;

#define GET_TYPEDEF_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffTypes.cpp.inc"


