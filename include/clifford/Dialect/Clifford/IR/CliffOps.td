
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


def Cliff_QuadProdOp : Cliff_Op<"quad_prod", []> {
    let summary = "Compute the product between two quaternions";

    let arguments = (ins Cliff_Quaternion:$LHS, Cliff_Quaternion:$RHS);
    let results = (outs Cliff_Quaternion:$out);

    let assemblyFormat = [{
    $LHS`,` $RHS attr-dict `:` type($LHS) `*` type($RHS) `->` type($out)
    }];

    let hasVerifier = 1;
}

