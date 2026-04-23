
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/Clifford/IR/CliffOpsEnums.cpp.inc"
#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

using namespace mlir;
using namespace mlir::cliff;




/// This method aims to find the degree of the multivector. This isnt trivial
/// as we only have an unordered mask. The main idea is iterating through every
/// basis in order, for example, for PGA2D this would be
/// 111 -> 110-> 101 -> 011 -> 100 -> 010 -> 001
/// We can separate in 111, 110-> 101 -> 011 and 100 -> 010 -> 001, this is, for every degree k,
/// we need all the combinations of k bits in n possible possitions.
/// This sets can be obtained by Gosper's Hack, so that is what we will be doing.
/// Next resource can be very helpful to get an intuition
/// http://programmingforinsomniacs.blogspot.com/2018/03/gospers-hack-explained.html

uint32_t Cliff_MultivectorType::getDegree() const {
    // This method aims to find the degree of the multivector. This isnt trivial
    // as we only have an unordered mask. The main idea is iterating through every
    // basis in order, for example, for PGA2D this would be
    // 111 -> 110-> 101 -> 011 -> 100 -> 010 -> 001
    // We can separate in 111, 110-> 101 -> 011 and 100 -> 010 -> 001, this is, for every degree k,
    // we need all the combinations of k bits in n possible possitions.
    // This sets can be obtained by Gosper's Hack, so that is what we will be doing.
    // Next resource can be very helpful to get an intuition
    // http://programmingforinsomniacs.blogspot.com/2018/03/gospers-hack-explained.html
    
    auto mask = getMask();
    auto space = getSpace();
    auto n = space.getP() + space.getQ() + space.getR();
    
    // pseudoscalar case
    if ((1ULL << ((1u << n) - 1)) & mask) return n;
    
    for (int k = n-1; k > 0; --k) {
        if (checkWithGospersHack(k, n, mask)) return k;
    }
    return 0;
}

uint64_t Cliff_MultivectorType::getActiveMask() const { return getMask(); }

uint64_t Cliff_ScalarType::getActiveMask() const { return 1; }

//todo : generalize to more than PG2D and PG3D if neccesary
uint64_t Cliff_PointType::getActiveMask() const {
    
    auto space = getSpace();
    assert(space.getP() > 0 && space.getQ() == 0 && space.getR() == 1 && "This function is only available for PGA algebras");

    int n = space.getP();
    return fillWithGospersHack(n, n+1);
}

uint64_t Cliff_LineType::getActiveMask() const {
    auto space = getSpace();
    assert(space.getP() > 0 && space.getQ() == 0 && space.getR() == 1 && "This function is only available for PGA algebras");

    int n = space.getP();
    return fillWithGospersHack(n-1, n+1);
}

uint64_t Cliff_MotorType::getActiveMask() const { 
    auto space = getSpace();
    assert(space.getP() > 0 && space.getQ() == 0 && space.getR() == 1 && "This function is only available for PGA algebras");
    int n = space.getP();
    // motors are always scalar + bivector
    return fillWithGospersHack(2, n+1);
}

Type Cliff_MultivectorType::asMultivector() const { return Type(*this); }

Type Cliff_PointType::asMultivector() const {
    auto mask = getActiveMask();
    return Cliff_MultivectorType::get(getContext(), mask,
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Point), 
        getSpace());
}
Type Cliff_LineType::asMultivector() const { 
    return Cliff_MultivectorType::get(getContext(), getActiveMask(),
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Line), 
        getSpace());
}
Type Cliff_MotorType::asMultivector() const { 
    return Cliff_MultivectorType::get(getContext(), getActiveMask(),
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Motor), 
        getSpace());
}
Type Cliff_ScalarType::asMultivector() const { 
    return Cliff_MultivectorType::get(getContext(), getActiveMask(),
        Float32Type::get(getContext()), 
        GeometricKindAttr::get(getContext(), GeometricKind::Scalar), 
        getSpace());
}

