include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "CliffDialect.td"

class Cliff_Type<string name, string typeMnemonic> : TypeDef<Cliff_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def Cliff_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def Cliff_FloatTensor : RankedTensorOf<[Cliff_Float]>;
def Cliff_FloatLike : AnyTypeOf<[Cliff_Float, Cliff_FloatTensor]>;


def Cliff_Quaternion : Cliff_Type<"Cliff_Quaternion", "quat"> {
    let summary = "A type for representing numbers of the quaternion set";
    let description = [{
        The `Quaternion` type allows you to represent numbers of the quaternion set,
        which have the form $a + bi + cj + dk$ and follow the rule
        $$i^2 = j^2 = k^2 = ijk = -1$$
    
    }];

    let parameters = (ins "Type":$dtype);

    let assemblyFormat = "`<` $dtype `>`";
}