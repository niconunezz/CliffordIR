#ifndef CLIFFORD_DIALECT
#define CLIFFORD_DIALECT

include "mlir/Pass/PassBase.td"
include "mlir/IR/DialectBase.td"


def Cliff_Dialect : Dialect {
    let name = "cliff";
    let summary = "A dialect for clifford algebra";

    let description = "A dialect for clifford algebra, for computer graphics or robotics";
    let cppNamespace = "::mlir::cliff";
    let useDefaultTypePrinterParser = 1;
}

#endif // CLIFFORD_DIALECT