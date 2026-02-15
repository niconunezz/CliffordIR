include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "CliffDialect.td"

class Cliff_Type<string name, string typeMnemonic> : TypeDef<Cliff_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def Cliff_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def Cliff_FloatTensor : RankedTensorOf<[Cliff_Float]>;
def Cliff_FloatLike : AnyTypeOf<[Cliff_Float, Cliff_FloatTensor]>;


def Cliff_Multivector : Cliff_Type<"Cliff_Multivector", "multivector"> {
    let summary = "A type for representing objects in cliffords algebra";
    let description = [{
        The `Multivector` type allows you to represent numbers of the cliffords
        algebra set of numbers.
    
    }];

    let parameters = (ins "int":$dim, "uint64_t":$mask, "Type":$dtype);
    let assemblyFormat = "`<` $dim `,` $mask `,` $dtype `>`";
}

def Cliff_MultivctTensor : RankedTensorOf<[Cliff_Multivector]>;


def Cliff_AnyType : AnyTypeOf<[Cliff_Multivector, Cliff_MultivctTensor, Cliff_FloatLike]>;