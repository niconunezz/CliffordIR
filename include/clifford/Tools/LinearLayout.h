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


namespace mlir::clg {
    
class LinearLayout {
    
private:
        
    llvm::MapVector<StringAttr,
        std::vector<std::vector<int32_t>>> bases;
    
    llvm::MapVector<StringAttr, int32_t> outDims;
    int32_t rank=0;

public:
    using BasesT = decltype(bases);
    LinearLayout() = default;

    friend bool operator==(const LinearLayout &lhs, const LinearLayout &rhs);
    friend llvm::hash_code hash_value(const LinearLayout& arg);

    auto getBases() const { return bases; } 
    auto getInDimNames() const { return llvm::make_first_range(bases);}
    auto getOutDimNames() const { return llvm::make_first_range(outDims);}

    int32_t getOutDimSize(StringAttr Name) const {
        return 1 << getOutDimSizeLog2(Name);
    }
    int32_t getOutDimSizeLog2(StringAttr Name) const;



    explicit LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames);

    static LinearLayout empty() { return {}; }


    friend LinearLayout operator*(LinearLayout inner, LinearLayout outer);

    LinearLayout &operator*=(LinearLayout outer) {
        *this = *this * outer;
        return *this;
    }

};

    

} // end namespace mlir::clg


#endif //CLIFFORD_TOOLS_LINEARLAYOUT_H