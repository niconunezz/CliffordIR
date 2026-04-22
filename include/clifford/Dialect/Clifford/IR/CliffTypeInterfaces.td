include "mlir/IR/OpBase.td"

// general methods that any object should have
def Cliff_AlgebraicElementInterface : TypeInterface<"Cliff_AlgebraicElementInterface"> {

    let cppNamespace = "::mlir::cliff";
    
    let methods = [
        InterfaceMethod<
            /*desc=*/"Get active basis mask.",
            /*retType=*/"uint64_t",
            /*methodName=*/"getActiveMask">,
        InterfaceMethod<
            /*desc=*/"Get space.",
            /*retType=*/"mlir::cliff::CliffordAlgebraAttr",
            /*methodName=*/"getAlgebra",
            /*args=*/(ins), [{}], [{
                return $_type.getSpace();
            }]>,
        InterfaceMethod<
            /*desc=*/"Convert to multivector.",
            /*retType=*/"mlir::Type",
            /*methodName=*/"asMultivector">

    ];

}


// oriented objects, e.g. points, lines, planes
def Cliff_GeometricElementInterface : TypeInterface<"Cliff_GeometricElementInterface", [Cliff_AlgebraicElementInterface]> {
    
    let cppNamespace = "::mlir::cliff";
    
    let methods = [
       
        InterfaceMethod<
            /*desc=*/"Obtain multivectors kind (euclidean/ideal/unkown)",
            /*retType=*/"mlir::cliff::ObjectKindAttr",
            /*methodName=*/"getObjectKind",
            /*args=*/(ins), [{}], [{
                return $_type.getKind();
            }]>
        
        InterfaceMethod<
            /*desc=*/"Get normalized status",
            /*retType=*/"bool",
            /*methodName=*/"isNormalized",
            /*args=*/(ins), [{}], [{
                return $_type.getNormalized();
            }]>

    ];
}