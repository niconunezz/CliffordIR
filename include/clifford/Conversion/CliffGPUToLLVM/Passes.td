include "mlir/Pass/PassBase.td"


def ConvertCliffGPUToLLVM : Pass<"convert-cliffGPU-to-llvm", "mlir::ModuleOp"> {
    let summary = "Conversion pass";

    let dependentDialects = ["mlir::clg::CliffGPUDialect",
                            "mlir::cliff::CliffDialect",
                            "mlir::arith::ArithDialect",
                            "mlir::LLVM::LLVMDialect"];
}