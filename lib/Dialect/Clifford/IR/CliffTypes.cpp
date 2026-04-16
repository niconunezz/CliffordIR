
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/Clifford/IR/CliffOpsEnums.cpp.inc"
#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

using namespace mlir;
using namespace mlir::cliff;

uint64_t fillWithGospersHack(int k, int n) {

    int set = (1 << k) - 1;
    int limit = (1 << n);
    uint64_t ret = 1;
    while (set < limit) {

        ret |= (1ULL << set);
        int c = set & -set;
        int r = set + c;
        set = (((r ^ set) >> 2) / c) | r;
    }

    return ret;
}



uint64_t Cliff_ScalarType::getMask() const { return 1; }

//todo : generalize to more than PG2D and PG3D if neccesary
uint64_t Cliff_PointType::getMask() const {
    
    auto space = getSpace();
    assert(space.getP() > 0 && space.getQ() == 0 && space.getR() == 1 && "This function is only available for PGA algebras");

    int n = space.getP();
    return fillWithGospersHack(n, n+1);
}

uint64_t Cliff_LineType::getMask() const {
    auto space = getSpace();
    assert(space.getP() > 0 && space.getQ() == 0 && space.getR() == 1 && "This function is only available for PGA algebras");

    int n = space.getP();
    return fillWithGospersHack(n-1, n+1);
}

//todo
uint64_t Cliff_MotorType::getMask() const { return 1; }



Type Cliff_PointType::asMultivector() const {
    auto mask = getMask();
    return Cliff_MultivectorType::get(getContext(), mask,
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Point), 
        getSpace());
}
Type Cliff_LineType::asMultivector() const { 
    return Cliff_MultivectorType::get(getContext(), getMask(),
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Line), 
        getSpace());
}
Type Cliff_MotorType::asMultivector() const { 
    return Cliff_MultivectorType::get(getContext(), getMask(),
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Motor), 
        getSpace());
}
Type Cliff_ScalarType::asMultivector() const { 
    return Cliff_MultivectorType::get(getContext(), getMask(),
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Scalar), 
        getSpace());
}

