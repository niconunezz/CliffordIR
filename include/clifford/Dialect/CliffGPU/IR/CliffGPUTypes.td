include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "clifford/Dialect/CliffGPU/IR/CliffGPUDialect.td"

class CliffGPU_Type<string name, string typeMnemonic> : TypeDef<CliffGPU_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def CLG_Pointer : CliffGPU_Type<"CLG_Pointer", "ptr"> {
    let summary = "Pointer type";
    let parameters = (ins "Type":$dtype);
    let assemblyFormat = "`<` $dtype `>`";
}
