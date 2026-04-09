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

def PointKindAttr : I32EnumAttr<"PointKind", "",
    [I32EnumAttrCase<"Euclidean", 0, "euclidean">,
    I32EnumAttrCase<"Ideal", 1, "ideal">,
    I32EnumAttrCase<"Unknown", 2, "unknown">]> {

    let cppNamespace = "::mlir::cliff";
}

def GeometricKindAttr : I32EnumAttr<"GeometricKind", "",
    [I32EnumAttrCase<"Point", 0, "point">,
    I32EnumAttrCase<"Line", 1, "line">,
    I32EnumAttrCase<"Unknown", 2, "unknown">]> {
        
    let cppNamespace = "::mlir::cliff";
}