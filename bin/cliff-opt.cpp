#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "clifford/Dialect/Clifford/Transforms/Passes.h"
#include "clifford/Conversion/CliffToCliffGPU/Passes.h"
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"


int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::clg::CliffGPUDialect>();
    registry.insert<mlir::cliff::CliffDialect>();

    registerAllDialects(registry);
    mlir::cliff::registerConvertCliffToCliffGPUPass();
    mlir::cliff::registerGeometricTypeConversionPass();
    
    mlir::clg::registerConvertCliffGPUToLLVMPass();


    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Clifford optimizer driver\n", registry));
}