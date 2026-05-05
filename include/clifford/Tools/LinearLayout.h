#ifndef CLIFFORD_TOOLS_LINEARLAYOUT_H
#define CLIFFORD_TOOLS_LINEARLAYOUT_H

#include <cstdint>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>


#include "clifford/Tools/StrUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/APInt.h"
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

    int32_t getInDimSize(StringAttr Name) const {
        return 1 << getInDimSizeLog2(Name);
    }

    std::vector<int32_t> getBasis(StringAttr inDimName, int32_t pos) const {
        auto it = bases.find(inDimName);
        assert (it != 0);
        assert(pos >= 0);
        return it->second[pos];
    }

    int32_t getOutDimIndex(StringAttr outDimName) const {
        int32_t i = 0;
        
        for (auto [name, _] : outDims) {
            if (outDimName == name)
                return i;
            ++i;
        }
        llvm::report_fatal_error("OutDimName doesnt belong to current outDims!");
    }
    
    int32_t getBasis(StringAttr inDimName, int32_t pos, StringAttr outDimName) const {
        int32_t outIdx = getOutDimIndex(outDimName); 
        return getBasis(inDimName, pos)[outIdx];
    }

SmallVector<std::pair<StringAttr, int32_t>> apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const;


    int32_t getInDimSizeLog2(StringAttr Name) const;

    LinearLayout sublayout(ArrayRef<StringAttr> inDimNames, ArrayRef<StringAttr> outDimNames) const;

    explicit LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames);

    explicit LinearLayout(
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
      ArrayRef<StringAttr> outDimNames);

    explicit LinearLayout(
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
        ArrayRef<std::pair<StringAttr, int32_t>> outDimNames);
    
    explicit LinearLayout(
        BasesT bases, llvm::MapVector<StringAttr, int32_t> outDims
    );

    std::string toString() const;

    static LinearLayout empty() { return {}; }
    static LinearLayout identity1D(unsigned N, StringAttr inDimName, StringAttr outDimName);
    static LinearLayout zeros1D(unsigned N, StringAttr inDimName, StringAttr outDimName);

    friend LinearLayout operator*(LinearLayout inner, LinearLayout outer);

    LinearLayout &operator*=(LinearLayout outer) {
        *this = *this * outer;
        return *this;
    }

};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const LinearLayout &layout) {
  os << layout.toString();
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                     const LinearLayout &layout) {
  os << layout.toString();
  return os;
}

    

} // end namespace mlir::clg


#endif //CLIFFORD_TOOLS_LINEARLAYOUT_H