#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"


int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::cliff::CliffDialect>();

    registerAllDialects(registry);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Clifford optimizer driver\n", registry));
}