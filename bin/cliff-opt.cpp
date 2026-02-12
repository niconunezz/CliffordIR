#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Clifford optimizer driver\n", registry));
}