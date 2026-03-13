#include "clifford/Tools/LinearLayout.h"

namespace mlir::clg {

bool operator==(const LinearLayout &lhs, const LinearLayout &rhs) {

    if (llvm::to_vector(lhs.getOutDimNames()) !=
        llvm::to_vector(rhs.getOutDimNames()))
        return false;

    if (lhs.bases.size() != rhs.bases.size())
        return false;

    for (auto it1 = lhs.bases.begin(), it2 = rhs.bases.begin();
         it1 != lhs.bases.end(); it1++, it2++) {
        
        if (*it1 != *it2) {
            return false;
        }
    }

    for (const auto &[lhsOutDimAndSize, rhsOutDimAndSize] :
        llvm::zip(lhs.outDims, rhs.outDims)) {
            if (lhsOutDimAndSize.second != rhsOutDimAndSize.second) 
                return false;
            
        }
    
    return true;
} 

LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
    // for know we just need the case ll * empty = ll to work
    if (inner.getInDimNames().empty())
        return outer;

    if (outer.getInDimNames().empty())
        return inner;
    
    return inner;
}

llvm::hash_code hash_value(const LinearLayout& layout) {
    size_t seed = 0;

    for (const auto &base : layout.getBases()) {
        seed = llvm::hash_combine(seed, base.first);

        for (const auto &vec : base.second) {
            for (int32_t val : vec) {
                seed = llvm::hash_combine(seed, val);
            }
        }
    }

    for (const auto &outDim : layout.getOutDimNames()) {
        seed = llvm::hash_combine(seed, outDim, layout.getOutDimSize(outDim));
    }

    return seed;
}


int32_t LinearLayout::getOutDimSizeLog2(StringAttr Name) const {
    auto it = outDims.find(Name);
    assert(it != outDims.end() && "Dim not found!");
    return llvm::Log2_32(it->second);
};


LinearLayout::LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames)
 : bases(std::move(bases)) {

    for (const auto& outDim : outDimNames) {
        outDims[outDim] = 1;
    }

    for (const auto &[inDim, inDimBases] : this->bases) {
        for (const auto &basis : inDimBases) {
            for (int i = 0; i < basis.size(); i++) {
                int32_t &size = outDims[outDimNames[i]];
                size = std::max<int32_t>(size, llvm::NextPowerOf2(basis[i]));
            }
        }
    }
    
}



} // end namespace mlir::clg

