#ifndef CLIFFORDGPU_DIALECT
#define CLIFFORDGPU_DIALECT

include "mlir/Pass/PassBase.td"
include "mlir/IR/DialectBase.td"


def CliffGPU_Dialect : Dialect {
    let name = "clg";
    let summary = "A dialect for clifford algebra gpu compute";

    let description = "A dialect for clifford algebra gpu compute, for computer graphics or robotics";
    let cppNamespace = "::mlir::clg";
    // let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

#endif // CLIFFORDGPU_DIALECT