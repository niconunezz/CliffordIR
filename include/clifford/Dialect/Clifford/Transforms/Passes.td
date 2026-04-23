include "mlir/Pass/PassBase.td"

def GeometricTypeConversionPass : Pass<"geometric-type-conversion"> {
    let summary = "Convert abstract types (point, line, motor, etc..) to their multivector representation";
    let description = "";

    let dependentDialects = [
        "mlir::cliff::CliffDialect",
    ];
}


def RewriteExponentialPass : Pass<"rewrite-exponential"> {
    let summary = "Rewrite exponential function to cliff.rotatate/cliff.translate";
    let description = "";

    let dependentDialects = [
        "mlir::cliff::CliffDialect",
    ];
}

def RewriteSandwichPass : Pass<"rewrite-sandwich"> {
    let summary = "Rewrite sandwich function to cliff.rotatate/cliff.translate";
    let description = "";

    let dependentDialects = [
        "mlir::cliff::CliffDialect",
    ];
}