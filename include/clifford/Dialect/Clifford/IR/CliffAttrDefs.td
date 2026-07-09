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


def EuclideanKind : I32EnumAttrCase<"Euclidean", 0, "euclidean">;
def IdealKind : I32EnumAttrCase<"Ideal", 1, "ideal">;
def UnknownKind : I32EnumAttrCase<"Unknown", 2, "unknown">;

def ObjectKind : I32EnumAttr<"ObjectKind", "",
    [EuclideanKind,
    IdealKind,
    UnknownKind]> {
    
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::cliff";
}
def ObjectKindAttr : EnumAttr<Cliff_Dialect, ObjectKind, "object_kind">;


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


def MotorKindRotor : I32EnumAttrCase<"Rotor", 0, "rotor">;
def MotorKindTranslator : I32EnumAttrCase<"Translator", 1, "translator">;
def MotorKindGeneral : I32EnumAttrCase<"General", 2, "general">;

def MotorKind : I32EnumAttr<"MotorKind", "",
    [MotorKindRotor, MotorKindTranslator, 
    MotorKindGeneral]> {
    
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::cliff";
}
def MotorKindAttr : EnumAttr<Cliff_Dialect, MotorKind, "motor_kind">;