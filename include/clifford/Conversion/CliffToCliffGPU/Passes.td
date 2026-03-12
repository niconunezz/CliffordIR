include "mlir/Pass/PassBase.td"


def ConvertCliffToCliffGPU : Pass<"convert-cliff-to-cliffGPU", "mlir::ModuleOp"> {
    let summary = "Conversion pass";

    let dependentDialects = ["mlir::clg::CliffGPUDialect",
                            "mlir::cliff::CliffDialect",
                            "mlir::arith::ArithDialect"];
}