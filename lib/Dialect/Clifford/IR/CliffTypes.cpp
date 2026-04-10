
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/Clifford/IR/CliffOpsEnums.cpp.inc"
#include "clifford/Dialect/Clifford/IR/CliffTypes.h"

using namespace mlir;
using namespace mlir::cliff;


uint64_t Cliff_MultivectorType::getActiveMask() const { return getMask(); }
uint64_t Cliff_PointType::getActiveMask() const {
    
    auto space = getSpace();

    assert(space.getP() > 0 && space.getQ() == 0 && space.getR() == 1 && "This function is only available for PGND algebras");

    int n = space.getP() + 1;
    uint64_t mask = 1;
    for (int i = 0; i < n; ++i) {
        int shift = !(1ULL << i);
        mask |= (1ULL << shift);
    }
    return mask;
}

//todo : generalize to more than PG2D
uint64_t Cliff_LineType::getActiveMask() const { return 23; }

//todo
uint64_t Cliff_MotorType::getActiveMask() const { return 1; }

CliffordAlgebraAttr Cliff_MultivectorType::getAlgebra() const { return getSpace(); }
CliffordAlgebraAttr Cliff_PointType::getAlgebra() const { return getSpace(); }
CliffordAlgebraAttr Cliff_LineType::getAlgebra() const { return getSpace(); }
CliffordAlgebraAttr Cliff_MotorType::getAlgebra() const { return getSpace(); }
