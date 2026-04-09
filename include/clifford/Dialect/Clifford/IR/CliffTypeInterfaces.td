include "mlir/IR/OpBase.td"


def Cliff_GeometricObjectInterface : TypeInterface<"Cliff_GeometricObjectInterface"> {

    let cppNamespace = "::mlir::cliff";
    
    let methods = [
        InterfaceMethod<
            /*desc=*/"Get active basis mask.",
            /*retType=*/"uint64_t",
            /*methodName=*/"getActiveMask">
    ];
}