#ifndef CLIFFORD_TOOLS_LINEARLAYOUT_H
#define CLIFFORD_TOOLS_LINEARLAYOUT_H

#include <cstdint>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

class LinearLayout {

private:
    
    llvm::MapVector<StringAttr,
                    std::vector<std::vector<int32_t>> bases;

    llvm::MapVector<StringAttr, int32_t> out_dims;
    int32_t rank=0;

}

#endif CLIFFORD_TOOLS_LINEARLAYOUT_H