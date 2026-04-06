include "mlir/IR/AttrTypeBase.td"
include "CliffGPUDialect.td"


class CliffGPU_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CliffGPU_Dialect, name, traits> {
        let mnemonic = attrMnemonic;

}

def LinearLayoutParam : AttrOrTypeParameter<"LinearLayout", "linear layout"> {
    let cppAccessorType = "const LinearLayout &";
}


def LinearEncodingAttr : CliffGPU_Attr<"LinearEncoding", "linear_encoding", []> {
    let mnemonic = "linear";
    let summary = "Attribute that defines the layout of a tensor in memory";
    let parameters = (ins LinearLayoutParam:$linearLayout);

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;

    let extraClassDeclaration= [{
            unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;
            unsigned getElemsPerThread(ArrayRef<int64_t> shape) const;
    }];
}