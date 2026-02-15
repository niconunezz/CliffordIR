#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Interfaces/FunctionImplementation.h"


#include "clifford/Dialect/Clifford/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::cliff;

#define GET_TYPEDEF_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffTypes.cpp.inc"

namespace mlir::cliff{

void FuncOp::print(OpAsmPrinter &p) {
    function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
    }
    
    
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}




} // end namespace mlir::cliff


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