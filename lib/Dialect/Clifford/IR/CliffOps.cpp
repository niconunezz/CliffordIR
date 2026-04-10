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

    auto lhsTy = dyn_cast<Cliff_GeometricObjectInterface>(lhsTensor.getElementType());
    auto rhsTy = dyn_cast<Cliff_GeometricObjectInterface>(rhsTensor.getElementType());
    auto outTy = dyn_cast<Cliff_GeometricObjectInterface>(outTensor.getElementType());

    if (!lhsTy || !rhsTy || !outTy)
        return emitError("tensor elements must be multivector types");

    if (lhsTy.getAlgebra() != rhsTy.getAlgebra() || 
        lhsTy.getAlgebra() != outTy.getAlgebra())
        return emitError("all operands must belong to the same algebra");

    auto algebra = dyn_cast<CliffordAlgebraAttr>(lhsTy.getAlgebra());
    if (!algebra)
        return emitError("algebra attribute must be a CliffordAlgebraAttr");

    unsigned p = algebra.getP(), q = algebra.getQ(), r = algebra.getR();

    const uint64_t lhsMask = lhsTy.getActiveMask();
    const uint64_t rhsMask = rhsTy.getActiveMask();
    const uint64_t outMask = outTy.getActiveMask();
    
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
    // s = 000, e0 = 001, e1 = 010, e2 = 100, e01 = 001, e02 = 101, e12 = 110, e012 = 111
    // Notice this numbers are an unordered list, if we ordered them it would look like
    // s, e0, e1, e01, e2, e02, e12, e123. Using this new ordered list we can represent an a
    // linear combination of this bases as an 8 bit number where each of the bits represents
    // if the corresponding basis is active (has a non-zero factor).
    // Set this ground-rules its easy to understand the following algorithm.
    // Note : The scalar case is kind of strange as 0...1 xor 0...1 = 0...0 but scalars dont dissappear,
    // for now we will force the scalar bit to be always 1 and assume there is always a scalar component. 

    if ((lhsMask & 1 == 0) || (rhsMask & 1 == 0))
        return emitError("lhsMask and rhsMask last bit should be always one!");
    

    int64_t resultMask = 0;
    while (lhsMaskCopy) {
        int lhsIndex = __builtin_ctz(lhsMaskCopy);
        int rhsMaskIter = rhsMaskCopy;
        while (rhsMaskIter) {
            int rhsIndex = __builtin_ctz(rhsMaskIter);

            int newBasis = lhsIndex ^ rhsIndex;
            int newSign = metricSign(lhsIndex, rhsIndex, p, q, r);

            if (newSign != 0) {
                resultMask |= (1ULL << newBasis);
            }
            
            rhsMaskIter &= (rhsMaskIter - 1);
        }
        lhsMaskCopy &= (lhsMaskCopy - 1);
    }

    
    if (resultMask != outMask) 
        return emitError("The mask of the result is ") << outMask << " when it should be " << resultMask;

    return success();
}

LogicalResult Add::verify() {
    auto lhsTy = dyn_cast<RankedTensorType>(getLhs().getType()).getElementType();
    auto lhsMask = cast<Cliff_MultivectorType>(lhsTy).getActiveMask();

    auto rhsTy = cast<RankedTensorType>(getRhs().getType()).getElementType();
    auto rhsMask = cast<Cliff_MultivectorType>(rhsTy).getActiveMask();

    auto outTy = cast<RankedTensorType>(getOut().getType()).getElementType();
    auto outMask = cast<Cliff_MultivectorType>(outTy).getActiveMask();

    if ((lhsMask | rhsMask) != outMask) 
        return emitError("Mask out should be or(maskLHS, maskRHS) but isnt");

    return success();
}

LogicalResult Sub::verify() {
    auto lhsTy = dyn_cast<RankedTensorType>(getLhs().getType()).getElementType();
    auto lhsMask = cast<Cliff_MultivectorType>(lhsTy).getActiveMask();

    auto rhsTy = cast<RankedTensorType>(getRhs().getType()).getElementType();
    auto rhsMask = cast<Cliff_MultivectorType>(rhsTy).getActiveMask();

    auto outTy = cast<RankedTensorType>(getOut().getType()).getElementType();
    auto outMask = cast<Cliff_MultivectorType>(outTy).getActiveMask();

    if ((lhsMask | rhsMask) != outMask) 
        return emitError("Mask out should be or(maskLHS, maskRHS) but isnt");

    return success();
}
