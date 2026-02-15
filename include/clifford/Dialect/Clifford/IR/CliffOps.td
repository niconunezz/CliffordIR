
include "CliffTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface



class Cliff_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cliff_Dialect, mnemonic, traits> {
    }



def Cliff_FuncOp : Cliff_Op<"func", [FunctionOpInterface,
                                   IsolatedFromAbove,
                                   HasParent<"ModuleOp">]> {
    let summary = "Op for defining functions";

    let arguments = (ins SymbolNameAttr:$sym_name, 
                    TypeAttrOf<FunctionType>:$function_type,
                    OptionalAttr<StrAttr>:$sym_visibility,
                    OptionalAttr<DictArrayAttr>:$arg_attrs,   
                    OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

def Cliff_ret : Cliff_Op<"ret", [Pure, HasParent<"FuncOp">, MemRefsNormalizable, ReturnLike, Terminator]> {
    let summary = "Operation for returning a function";

    let arguments = (ins Variadic<AnyType>:$retVal);
    let assemblyFormat = "attr-dict ($retVal^ `:` type($retVal))?";
}


def Cliff_GeoProd : Cliff_Op<"geo_prod", [SameOperandsAndResultType]> {
    let summary = "Compute the geometric product between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($out)";

    // let hasVerifier = 1;
}

