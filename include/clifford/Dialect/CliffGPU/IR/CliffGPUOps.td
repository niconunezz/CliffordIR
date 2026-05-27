include "CliffGPUDialect.td"
include "clifford/Dialect/Clifford/IR/CliffDialect.td"
include "clifford/Dialect/Clifford/IR/CliffTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface



class CLG_Op<string mnemonic, list<Trait> traits = []> :
    Op<CliffGPU_Dialect, mnemonic, traits> {
    }


def CLG_LoadOp : CLG_Op<"load", []> {
    let summary = "Load a tensor from a pointer";
    let arguments = (ins Cliff_Pointer:$ptr);
    let results = (outs Cliff_MultivctTensor:$value);
    let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($value)";
}

def CLG_StoreOp : CLG_Op<"store", []> {
    let summary = "Load a tensor from a pointer";
    let arguments = (ins Cliff_Pointer:$ptr, Cliff_MultivctTensor:$value);
    let assemblyFormat = "$ptr `,` $value attr-dict `:` type($ptr) `,` type($value)";
}