include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "CliffDialect.td"
include "CliffTypeInterfaces.td"


class Cliff_Type<string name, string typeMnemonic, list<Trait> traits> : TypeDef<Cliff_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Cliff_Int : AnyTypeOf<[I32, I8], "integer">;
def Cliff_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def Cliff_FloatTensor : RankedTensorOf<[Cliff_Float]>;
def Cliff_FloatLike : AnyTypeOf<[Cliff_Float, Cliff_FloatTensor]>;


def Cliff_Scalar : Cliff_Type<"Cliff_Scalar",
                                 "scalar", 
                                 [DeclareTypeInterfaceMethods<Cliff_GeometricObjectInterface>]> {
    let summary = "A type that represents a Scalar object in the specified algebra";
    let description = [{
        The `Scalar` type should be thinked of a more concret representation of a multivector.
        It is neccesary for early on optimizations that are exlusive for scalars.
    }];

    let parameters = (ins "CliffordAlgebraAttr" : $space);
    
    let assemblyFormat = "`<` $space `>`";
}

def Cliff_Point : Cliff_Type<"Cliff_Point",
                                 "point", 
                                 [DeclareTypeInterfaceMethods<Cliff_GeometricObjectInterface>]> {
    let summary = "A type that represents a point object in the specified algebra";
    let description = [{
        The `Point` type should be thinked of a more concret representation of a multivector.
        It is neccesary for early on optimizations that are exlusive for points.
    }];

    let parameters = (ins AttrParameter<"::mlir::cliff::ObjectKindAttr", "">:$kind, "bool" : $normalized, "CliffordAlgebraAttr" : $space);
    
    let assemblyFormat = "`<` $kind `,` $normalized `,` $space `>`";

}

def Cliff_Line : Cliff_Type<"Cliff_Line",
                                 "line", 
                                 [DeclareTypeInterfaceMethods<Cliff_GeometricObjectInterface>]> {
    let summary = "A type that represents a line in the specified algebra";
    let description = [{
        The `Line` type should be thinked of a more concret representation of a multivector.
        It is neccesary for early on optimizations that are exlusive for lines.
    }];

    let parameters = (ins AttrParameter<"::mlir::cliff::ObjectKindAttr", "">:$kind, "bool" : $normalized, "CliffordAlgebraAttr" : $space);
    
    let assemblyFormat = "`<` $kind `,` $normalized `,` $space `>`";

}

def Cliff_Motor : Cliff_Type<"Cliff_Motor",
                                 "motor", 
                                 [DeclareTypeInterfaceMethods<Cliff_GeometricObjectInterface>]> {
    let summary = "A type that represents a motor in the specified algebra";
    let description = [{
        The `Motor` type should be thinked of a more concret representation of a multivector.
        It is neccesary for early on optimizations that are exlusive for lines.
    }];

    let parameters = (ins "bool" : $normalized, "CliffordAlgebraAttr" : $space);
    
    let assemblyFormat = "`<` $normalized `,` $space `>`";

}


def Cliff_Multivector : Cliff_Type<"Cliff_Multivector",
                                   "multivector", 
                                   [DeclareTypeInterfaceMethods<Cliff_GeometricObjectInterface>]> {
    let summary = "A type for representing objects in cliffords algebra";
    let description = [{
        The `Multivector` type allows you to represent numbers of the cliffords
        algebra set of numbers.
    
    }];

    let parameters = (ins "uint64_t":$mask, "Type":$dtype,
                      AttrParameter<"::mlir::cliff::GeometricKindAttr", ""> : $kind,
                      "CliffordAlgebraAttr" : $space);

    let assemblyFormat = "`<` $mask `,` $kind `,` $dtype `,` $space `>`";
}

def Cliff_GeometricKind : AnyTypeOf<[Cliff_Multivector, Cliff_Point, Cliff_Scalar, Cliff_Line, Cliff_Motor]>;

def Cliff_LineOrPointTensor : RankedTensorOf<[Cliff_Point, Cliff_Line]>;
def Cliff_ScalarTensor : RankedTensorOf<[Cliff_Scalar]>;
def Cliff_MotorTensor : RankedTensorOf<[Cliff_Motor]>;


def Cliff_MultivctTensor : RankedTensorOf<[Cliff_GeometricKind]>;

def Cliff_AnyType : AnyTypeOf<[Cliff_Multivector, Cliff_MultivctTensor, Cliff_FloatLike]>;