include "mlir/IR/AttrTypeBase.td"
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