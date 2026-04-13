include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "CliffDialect.td"


class Cliff_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Cliff_Dialect, name, traits> {
        let mnemonic = attrMnemonic;

}


def AlgebraDefinition : Cliff_Attr<"CliffordAlgebra", "algebra_definition", []> {
    let mnemonic = "algebra";
    let summary = "Attribute that defines the characteristic of our clifford algebra";
    let parameters = (ins "unsigned":$p, "unsigned":$q, "unsigned":$r);

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}


def PointKindEuclidean : I32EnumAttrCase<"Euclidean", 0, "euclidean">;
def PointKindIdeal : I32EnumAttrCase<"Ideal", 1, "ideal">;
def PointKindUnknown : I32EnumAttrCase<"Unknown", 2, "unknown">;

def PointKind : I32EnumAttr<"PointKind", "",
    [PointKindEuclidean,
    PointKindIdeal,
    PointKindUnknown]> {
    
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::cliff";
}
def PointKindAttr : EnumAttr<Cliff_Dialect, PointKind, "point_kind">;


def GeometricKindPoint : I32EnumAttrCase<"Point", 0, "point">;
def GeometricKindLine : I32EnumAttrCase<"Line", 1, "line">;
def GeometricKindMotor : I32EnumAttrCase<"Motor", 2, "motor">;
def GeometricKindScalar : I32EnumAttrCase<"Scalar", 3, "scalar">;
def GeometricKindUnknown : I32EnumAttrCase<"Unknown", 4, "unknown">;


def GeometricKind : I32EnumAttr<"GeometricKind", "",
    [GeometricKindPoint, GeometricKindLine, 
    GeometricKindMotor, GeometricKindScalar,
    GeometricKindUnknown]> {
    
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::cliff";
}

def GeometricKindAttr : EnumAttr<Cliff_Dialect, GeometricKind, "geo_kind">;