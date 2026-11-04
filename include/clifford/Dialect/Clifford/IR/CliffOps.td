include "clifford/Dialect/Clifford/IR/CliffTypes.td"
include "clifford/Dialect/Clifford/IR/CliffAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/AttrTypeBase.td"



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
    ::mlir::SymbolTable::Visibility getVisibility() {
    return mlir::SymbolTable::getSymbolVisibility(*this);
    }

    void setVisibility(::mlir::SymbolTable::Visibility vis) {
        if (vis == ::mlir::SymbolTable::Visibility::Public) {
            setSymVisibilityAttr(nullptr);
            return;
        }
        setSymVisibilityAttr(::mlir::StringAttr::get(
            getContext(),
            vis == ::mlir::SymbolTable::Visibility::Private ? "private" : "nested"));
        }
    ::llvm::StringRef getName() { return getSymName(); }
    ::mlir::StringAttr getNameAttr() { return getSymNameAttr(); }
    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

def Cliff_ReturnOp : Cliff_Op<"ret", [Pure, HasParent<"FuncOp">, MemRefsNormalizable, ReturnLike, Terminator]> {
    let summary = "Operation for returning a function";

    let arguments = (ins Variadic<AnyType>:$retVal);
    let assemblyFormat = "attr-dict ($retVal^ `:` type($retVal))?";
}


def Cliff_Add : Cliff_Op<"add", [Pure]> {
    let summary = "Compute the addition between two multivectors";

    let arguments = (ins Cliff_GeometricKindTensor:$lhs, Cliff_GeometricKindTensor:$rhs);
    let results = (outs Cliff_GeometricKindTensor:$out);

    // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `+` type($rhs) `->` type($out)";

    let hasVerifier = 1;
}

def Cliff_GeoProd : Cliff_Op<"geo_prod", [Pure]> {
    let summary = "Compute the geometric product between two multivectors";

    let arguments = (ins Cliff_GeometricKindTensor:$lhs, Cliff_GeometricKindTensor:$rhs);
    let results = (outs Cliff_GeometricKindTensor:$out);

    // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";

    let hasVerifier = 1;
}

def Cliff_InnerProd : Cliff_Op<"inner_prod", [Pure]> {
    let summary = "Compute the inner product between two multivectors";

    let arguments = (ins Cliff_GeometricKindTensor:$lhs, Cliff_GeometricKindTensor:$rhs);
    let results = (outs Cliff_GeometricKindTensor:$out);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";
    // let hasVerifier = 1;
}

def Cliff_Translate : Cliff_Op<"translate", [Pure]> {
    let summary = "Translate a point";

    let arguments = (ins Cliff_GeometricKindTensor:$dist, Cliff_GeometricKindTensor:$src);
    let results = (outs Cliff_GeometricKindTensor:$out);

    let assemblyFormat = "$src `,` $dist attr-dict `:` type($src) `*` type($dist) `->` type($out)";
}

def Cliff_Rotate : Cliff_Op<"rotate", [Pure]> {
    let summary = "Rotate a point";

    let arguments = (ins Cliff_GeometricKindTensor:$angle, Cliff_GeometricKindTensor:$src);
    let results = (outs Cliff_GeometricKindTensor:$out);

    let assemblyFormat = "$src `,` $angle attr-dict `:` type($src) `*` type($angle) `->` type($out)";
    let hasVerifier = 1;
}

def Cliff_SinOp : Cliff_Op<"sin", [Pure, SameOperandsAndResultType]> {
    let summary = "Calculate elementwise sine";

    let arguments = (ins Cliff_MultivctTensor:$src);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$src attr-dict `:` type($src)";
}

def Cliff_CosOp : Cliff_Op<"cos", [Pure, SameOperandsAndResultType]> {
    let summary = "Calculate elementwise cosine";

    let arguments = (ins Cliff_MultivctTensor:$src);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$src attr-dict `:` type($src)";
}


def Cliff_Reverse : Cliff_Op<"reverse", [Pure]> {
    let summary = "Compute the reverse of a multivector";

    let arguments = (ins Cliff_GeometricKindTensor:$src);
    let results = (outs Cliff_GeometricKindTensor:$out);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($out)";

    // let hasVerifier = 1;
}

def Cliff_Sandwich : Cliff_Op<"sandwich", [Pure]> {
    let summary = "Compute the sandwich between two multivectors";

    let arguments = (ins Cliff_GeometricKindTensor:$lhs, Cliff_GeometricKindTensor:$rhs);
    let results = (outs Cliff_GeometricKindTensor:$out);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";

    // let hasVerifier = 1;
}


def Cliff_Exp : Cliff_Op<"exp", [Pure]> {
    let summary = "Compute the exponential of a multivector";

    let arguments = (ins Cliff_GeometricKindTensor:$src);
    let results = (outs Cliff_GeometricKindTensor:$out);
    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($out)";
}


//TODO : check result type of sign


def Cliff_Grade : Cliff_Op<"get_grade", [Pure]> {
    let summary = "Obtain a multivectors i grade";

    let arguments = (ins Cliff_GeometricKindTensor:$src, I64Attr:$grade);
    let results = (outs Cliff_GeometricKindTensor:$out);

    let assemblyFormat =
    "$src `,` $grade attr-dict `:` type($src) `->` type($out)";

    let hasVerifier = 1;
}