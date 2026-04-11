include "mlir/Pass/PassBase.td"

def GeometricTypeConversionPass : Pass<"geometric-type-conversion"> {
    let summary = "Convert abstract types (point, line, motor, etc..) to their multivector representation";
    let description = "";

    let dependentDialects = [
        "mlir::cliff::CliffDialect",
    ];
}