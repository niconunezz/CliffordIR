#include "clifford/Dialect/Clifford/IR/Dialect.h"

#define GET_OP_CLASSES
#include "clifford/Dialect/Clifford/IR/CliffOps.cpp.inc"

using namespace mlir::cliff;
using namespace llvm;

LogicalResult GeoProd::verify() {
    auto lhsTensor = dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhsTensor = dyn_cast<RankedTensorType>(getRhs().getType());
    auto outTensor = dyn_cast<RankedTensorType>(getOut().getType());

    if (!lhsTensor || !rhsTensor || !outTensor)
        return emitError("operands must be ranked tensors");

    auto maybeLhsConcrete = dyn_cast<Cliff_ConcreteElementInterface>(lhsTensor.getElementType());
    auto maybeRhsConcrete = dyn_cast<Cliff_ConcreteElementInterface>(rhsTensor.getElementType());

    // if we deal with objs different from multivectors we dont
    // want to rely on the getMask methods as they dont include important info.
    // e.g. a sandwich product over an object b always returns this same b object
    // this depends completely on the fact that the geo prod on the left and on the
    // right are done on the same multivector (reversed on the right). This method
    // cannot infer that, creating the upper bound and therefore, wasting memory
    if (maybeLhsConcrete || maybeRhsConcrete)
        return success();


    auto lhsTy = dyn_cast<Cliff_MultivectorType>(lhsTensor.getElementType());
    auto rhsTy = dyn_cast<Cliff_MultivectorType>(rhsTensor.getElementType());
    auto outTy = dyn_cast<Cliff_MultivectorType>(outTensor.getElementType());

    if (!lhsTy || !rhsTy || !outTy)
        return emitError("tensor elements must be multivector types");

    // same reason here, if we know the value of the output, we trust it to be right
    if (outTy.getKind().getValue() != GeometricKind::Unknown)
        return success();
    

    if (lhsTy.getSpace() != rhsTy.getSpace() || 
        lhsTy.getSpace() != outTy.getSpace())
        return emitError("all operands must belong to the same algebra");

    auto algebra = dyn_cast<CliffordAlgebraAttr>(lhsTy.getSpace());
    if (!algebra)
        return emitError("algebra attribute must be a CliffordAlgebraAttr");

    unsigned p = algebra.getP(), q = algebra.getQ(), r = algebra.getR();

    const uint64_t lhsMask = lhsTy.getMask();
    const uint64_t rhsMask = rhsTy.getMask();
    const uint64_t outMask = outTy.getMask();
    
    uint64_t lhsMaskCopy = lhsMask;
    uint64_t rhsMaskCopy = rhsMask;
    uint64_t outMaskCopy = outMask;
    
    // scalar case
    if (!lhsMask) {
        int64_t resultMask = rhsMask;

        if (resultMask != outMask) 
            return emitError("The mask of the result is ") << outMask << " when it should be " << resultMask;
    
        return success();
    }
    
    if (!rhsMask) {
        int64_t resultMask = lhsMask;
        if (resultMask != outMask) 
            return emitError("The mask of the result is ") << outMask << " when it should be " << resultMask;
    
        return success();
    }
    
    // to explain how this works, lets use an example using PG2d (p=2, q=0, r=1)
    // we represent an individual basis $e_{i}$ as a 3 bit number (as we have 8 possible combinations)
    // where each of his active basis represent his bit. This results in the next mapping,
    // s = 000, e0 = 001, e1 = 010, e2 = 100, e01 = 011, e02 = 101, e12 = 110, e012 = 111
    // Notice this numbers are an unordered list, if we ordered them it would look like
    // s, e0, e1, e01, e2, e02, e12, e123. Using this new ordered list we can represent an a
    // linear combination of this bases as an 8 bit number where each of the bits represents
    // if the corresponding basis is active (has a non-zero factor).
    // Set this ground-rules its easy to understand the following algorithm.

    uint64_t resultMask = getResultMask(lhsMaskCopy, rhsMaskCopy, p, q, r);
    
    if (resultMask != outMask) 
        return emitError("The mask of the result is ") << outMask << " when it should be " << resultMask;

    return success();
}

LogicalResult Rotate::verify() {
    // auto refTensorTy = dyn_cast<RankedTensorType>(getSrc().getType());
    // auto angleTensorTy = dyn_cast<RankedTensorType>(getAngle().getType());
    // if (!refTensorTy || angleTensorTy)
    //     return failure();
    // auto refTy = dyn_cast<Cliff_GeometricElementInterface>(refTensorTy.getElementType());
    // auto angleTy = dyn_cast<Cliff_ScalarType>(refTensorTy.getElementType());

    // if (!refTy || !angleTy)
    //     return failure();

    // if (!refTy.isNormalized())
    //     return emitError("Rotate operation only support normalized objects");

    
    return success();

}

LogicalResult Add::verify() {
    auto lhsTensorTy = dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhsTensorTy = dyn_cast<RankedTensorType>(getRhs().getType());
    auto outTensorTy = dyn_cast<RankedTensorType>(getOut().getType());

    if (!lhsTensorTy || !rhsTensorTy || !outTensorTy)
        return failure();
    
    auto lhsTy = dyn_cast<Cliff_MultivectorType>(lhsTensorTy.getElementType());
    auto rhsTy = dyn_cast<Cliff_MultivectorType>(rhsTensorTy.getElementType());
    auto outTy = dyn_cast<Cliff_MultivectorType>(outTensorTy.getElementType());

    if (!lhsTy || !rhsTy || !outTy)
        return failure();    

    auto lhsMask = lhsTy.getMask();
    auto rhsMask = rhsTy.getMask();
    auto outMask = outTy.getMask();

    if ((lhsMask | rhsMask) != outMask) 
        return emitError("Mask out should be or(maskLHS, maskRHS) but isnt");

    return success();
}

LogicalResult Sub::verify() {
    auto lhsTensorTy = dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhsTensorTy = cast<RankedTensorType>(getRhs().getType());
    auto outTensorTy = cast<RankedTensorType>(getOut().getType());

    if (!lhsTensorTy || !rhsTensorTy || !outTensorTy)
        return failure();
    
    auto lhsTy = dyn_cast<Cliff_MultivectorType>(lhsTensorTy.getElementType());
    auto rhsTy = dyn_cast<Cliff_MultivectorType>(rhsTensorTy.getElementType());
    auto outTy = dyn_cast<Cliff_MultivectorType>(outTensorTy.getElementType());

    if (!lhsTy || !rhsTy || !outTy)
        return failure();    

    auto lhsMask = lhsTy.getMask();
    auto rhsMask = rhsTy.getMask();
    auto outMask = outTy.getMask();

    if ((lhsMask | rhsMask) != outMask) 
        return emitError("Mask out should be or(maskLHS, maskRHS) but isnt");

    return success();
}
