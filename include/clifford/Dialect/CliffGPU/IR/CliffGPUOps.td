include "CliffGPUDialect.td"
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


def CLG_GetTid<"get_tid", [Pure]> {
    let summary = "Get Tile ID";
    let results = (outs "uint32_t":$value);
    let assemblyFormat = "attr-dict `:` type($value)";
}