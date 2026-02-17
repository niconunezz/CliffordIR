#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

#include "llvm/Support/Casting.h"

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

#define GET_ATTRDEF_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffAttrDefs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

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




Attribute CliffordAlgebraAttr::parse(AsmParser &parser, Type type) {
    if (parser.parseLess().failed())
        return {};
    
    DictionaryAttr dict;
    if (parser.parseAttribute(dict).failed())
        return {};
    
    if (parser.parseGreater().failed())
        return {};

    unsigned p = 0, q = 0, r = 0;

    
    for (const NamedAttribute &param : dict) {
        StringRef name = param.getName().getValue();
        Attribute value = param.getValue();
        auto intAttr = dyn_cast<IntegerAttr>(value);
        if (!intAttr) {
           parser.emitError(parser.getNameLoc(), "attribute '") << name << "' must be an integer";
            return nullptr;
        }
        unsigned intValue = intAttr.getInt();
        if (name == "p") {
            p = intValue;
        } else if (name == "q") {
            q = intValue;
        } else if (name == "r") {
            r = intValue;
        } else {
            parser.emitError(parser.getNameLoc()) 
                << "unknown attribute '" << name << "'"; 
            return nullptr;
        }
    }

    if (!dict.get("p") || !dict.get("q") || !dict.get("r")) {
        parser.emitError(parser.getNameLoc(), "missing required attributes 'p', 'q', or 'r'");

        return nullptr;
    }
    return CliffordAlgebraAttr::get(parser.getContext(), p, q, r);
};

void CliffordAlgebraAttr::print(::mlir::AsmPrinter &printer) const {
   printer << "<{p : " << getP() << ", q:" << getQ() << ", r:" << getR() << "}>"; 
}


LogicalResult CliffordAlgebraAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned p, unsigned q, unsigned r) {
    
    return success();
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
        
        addAttributes<
    #define GET_ATTRDEF_LIST
    #include "clifford/Dialect/Clifford/IR/CliffAttrDefs.cpp.inc"
        >();

}