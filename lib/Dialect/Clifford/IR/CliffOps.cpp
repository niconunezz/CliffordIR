#include "clifford/Dialect/Clifford/IR/Dialect.h"

#define GET_OP_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffOps.cpp.inc"

using namespace mlir::cliff;
using namespace llvm;

LogicalResult GeoProd::verify() {
    auto LhsTy = dyn_cast<RankedTensorType>(getLhs().getType()).getElementType();
    auto LhsMask = cast<Cliff_MultivectorType>(LhsTy).getMask();

    auto RhsTy = cast<RankedTensorType>(getRhs().getType()).getElementType();
    auto RhsMask = cast<Cliff_MultivectorType>(RhsTy).getMask();

    auto OutTy = cast<RankedTensorType>(getOut().getType()).getElementType();
    auto OutMask = cast<Cliff_MultivectorType>(OutTy).getMask();

    // if ((LhsMask ^ RhsMask) != OutMask) 
    //     return emitError("Mask out should be xor(maskLHS, maskRHS) but isnt");

    return success();
}

LogicalResult Add::verify() {
    auto LhsTy = dyn_cast<RankedTensorType>(getLhs().getType()).getElementType();
    auto LhsMask = cast<Cliff_MultivectorType>(LhsTy).getMask();

    auto RhsTy = cast<RankedTensorType>(getRhs().getType()).getElementType();
    auto RhsMask = cast<Cliff_MultivectorType>(RhsTy).getMask();

    auto OutTy = cast<RankedTensorType>(getOut().getType()).getElementType();
    auto OutMask = cast<Cliff_MultivectorType>(OutTy).getMask();

    if ((LhsMask | RhsMask) != OutMask) 
        return emitError("Mask out should be or(maskLHS, maskRHS) but isnt");

    return success();
}

LogicalResult Sub::verify() {
    auto LhsTy = dyn_cast<RankedTensorType>(getLhs().getType()).getElementType();
    auto LhsMask = cast<Cliff_MultivectorType>(LhsTy).getMask();

    auto RhsTy = cast<RankedTensorType>(getRhs().getType()).getElementType();
    auto RhsMask = cast<Cliff_MultivectorType>(RhsTy).getMask();

    auto OutTy = cast<RankedTensorType>(getOut().getType()).getElementType();
    auto OutMask = cast<Cliff_MultivectorType>(OutTy).getMask();

    if ((LhsMask | RhsMask) != OutMask) 
        return emitError("Mask out should be or(maskLHS, maskRHS) but isnt");

    return success();
}
