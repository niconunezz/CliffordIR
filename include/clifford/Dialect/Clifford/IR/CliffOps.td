
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

def Cliff_ReturnOp : Cliff_Op<"ret", [Pure, HasParent<"FuncOp">, MemRefsNormalizable, ReturnLike, Terminator]> {
    let summary = "Operation for returning a function";

    let arguments = (ins Variadic<AnyType>:$retVal);
    let assemblyFormat = "attr-dict ($retVal^ `:` type($retVal))?";
}


def Cliff_Add : Cliff_Op<"add", [Pure]> {
    let summary = "Compute the addition between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `+` type($rhs) `->` type($out)";

    let hasVerifier = 1;
}

def Cliff_Sub : Cliff_Op<"sub", [Pure, ]> {
    let summary = "Compute the substraction between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `+` type($rhs) `->` type($out)";

    let hasVerifier = 1;
}

def Cliff_Constant : Cliff_Op<"constant", [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
    let summary = "Define a constant multivector";

    //TODO : make this fp
    let arguments = (ins AnyIntElementsAttr :$value);
    let results = (outs Cliff_MultivctTensor:$result);

    let assemblyFormat = "$value attr-dict `:` type($result) ";
    // let hasVerifier = 1;
}


def Cliff_GeoProd : Cliff_Op<"geo_prod", [Pure]> {
    let summary = "Compute the geometric product between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";

    let hasVerifier = 1;
}

def Cliff_Translate : Cliff_Op<"translate", [Pure]> {
    let summary = "Translate a point";

    let arguments = (ins Cliff_ScalarTensor:$dist, Cliff_LineOrPointTensor:$ref);
    let results = (outs Cliff_MotorTensor:$out);

    let assemblyFormat = "$ref `,` $dist attr-dict `:` type($ref) `*` type($dist) `->` type($out)";
}

def Cliff_Rotate : Cliff_Op<"rotate", [Pure]> {
    let summary = "Rotate a point";

    let arguments = (ins Cliff_ScalarTensor:$angle, Cliff_LineOrPointTensor:$ref);
    let results = (outs Cliff_MotorTensor:$out);

    let assemblyFormat = "$ref `,` $angle attr-dict `:` type($ref) `*` type($angle) `->` type($out)";
}


def Cliff_Vee : Cliff_Op<"vee", [Pure]> {
    let summary = "Compute the vee between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";


    // let hasVerifier = 1;
}

def Cliff_Wedge : Cliff_Op<"wedge", [Pure]> {
    let summary = "Compute the wedge between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";

    // let hasVerifier = 1;
}

def Cliff_Dot : Cliff_Op<"dot", [Pure]> {
    let summary = "Compute the dot between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";


    // let hasVerifier = 1;
}

def Cliff_Sandwich : Cliff_Op<"sandwich", [Pure]> {
    let summary = "Compute the sandwich between two multivectors";

    let arguments = (ins Cliff_MultivctTensor:$lhs, Cliff_MultivctTensor:$rhs);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `*` type($rhs) `->` type($out)";

    // let hasVerifier = 1;
}

def Cliff_Dual : Cliff_Op<"dual", [Pure, SameOperandsAndResultType]> {
    let summary = "Compute the dual of a multivector";

    let arguments = (ins Cliff_MultivctTensor:$src);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$src attr-dict `:` type($out)";

    // let hasVerifier = 1;
}


def Cliff_Exp : Cliff_Op<"exp", [Pure]> {
    let summary = "Compute the exponential of a multivector";

    let arguments = (ins Cliff_MultivctTensor:$src);
    let results = (outs Cliff_MultivctTensor:$out);
    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($out)";
}


def Cliff_Normalize : Cliff_Op<"normalize", [Pure, SameOperandsAndResultType]> {
    let summary = "Normalize a multivector";

    let arguments = (ins Cliff_MultivctTensor:$src);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$src attr-dict `:` type($out)";

    // let hasVerifier = 1;
}

//TODO : check result type of sign

def Cliff_Sign : Cliff_Op<"sign", [Pure, SameOperandsAndResultType]> {
    let summary = "Obtain a multivectors sign";

    let arguments = (ins Cliff_MultivctTensor:$src);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$src attr-dict `:` type($out)";

    // let hasVerifier = 1;
}


def Cliff_Idx : Cliff_Op<"get_idx", [Pure]> {
    let summary = "Obtain a multivectors i part";

    let arguments = (ins Cliff_MultivctTensor:$src, Cliff_Int:$idx);
    let results = (outs Cliff_MultivctTensor:$out);

    let assemblyFormat = "$src `,` $idx attr-dict `:` type($src) `,` type($idx)  `->` type($out)";

    // let hasVerifier = 1;
}