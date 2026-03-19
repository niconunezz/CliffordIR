include "mlir/Pass/PassBase.td"


def ConvertCliffToCliffGPU : Pass<"convert-cliff-to-cliffGPU", "mlir::ModuleOp"> {
    let summary = "Conversion pass";

    let dependentDialects = ["mlir::clg::CliffGPUDialect",
                            "mlir::cliff::CliffDialect",
                            "mlir::arith::ArithDialect"];

    
    let options = [
        Option<"numWarps", "num-warps", "int32_t", /*default=*/ "8", "Target machine number of warps">,
        Option<"threadsPerWarp", "threads-per-warp", "int32_t", /*default=*/ "32", "Target machine threads per warp">
    ];
}