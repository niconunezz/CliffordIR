#include "clifford/Tools/LinearLayout.h"

namespace mlir::clg {


using BasesT = LinearLayout::BasesT;


BasesT makeBasesMap(ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases) {
    BasesT retBases;
    for (auto const &[inDim, inDimBases] : bases) {
        retBases[inDim] = inDimBases;
    }
    return retBases;
}


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

/*static*/ LinearLayout LinearLayout::identity1D(unsigned N, StringAttr inDimName, StringAttr outDimName) {
    assert(llvm::isPowerOf2_32(N) && "N must be a power of 2");
    
    std::vector<std::vector<int32_t>> bases;

    BasesT ret0;
    ArrayRef ret1 = {outDimName};

    for (int i = 1; i < N; i *= 2) {
        bases.emplace_back(std::vector<int32_t>{i});
    }
    ret0[inDimName] = bases;

    return LinearLayout(ret0, ret1);
}

SmallVector<StringAttr> getSupremum(const SmallVector<StringAttr> &a, const SmallVector<StringAttr> &b) {
    SmallVector<StringAttr> ret;
    const int stride = b.size() + 1;    
    int m = a.size(), n = b.size();
    std::vector<int16_t> lcs((m+1)*stride, 0);
    for (auto i=0; i < a.size(); i++) {
        for (auto j=0; j < b.size(); j++) {
            lcs[(i + 1)*stride + j + 1] =
                (a[i] == b[j]) ? lcs[i*stride + j] + 1 : max(lcs[(i+1)*stride + j], lcs[i*stride + j+1]);
        }
    }
        
    const int16_t lcs_size = lcs[a.size()*stride + b.size()];
    // backtracing
    std::vector<pair<int16_t, int16_t>> anchorPositions(lcs_size);
    int16_t position = lcs_size;
    while ((m > 0) && (n > 0)) {
        if (a[m-1] == b[n-1]) {
            anchorPositions[position-1] = std::make_pair<int16_t,int16_t>(m-1, n-1);
            position--; m--; n--;
        }
        else if (lcs[(m-1)*stride + n] > lcs[m*stride + n-1]) {
            m--;
        }
        else {
            n--;
        }
    }
    
    int currIdxA = 0, currIdxB = 0;
    for (const auto &[idxA, idxB] : anchorPositions) {
        while (currIdxA != idxA) {
            if (llvm::is_contained(ret, a[currIdxA]))
                llvm_unreachable("Supremum does not exist");
            
            ret.push_back(a[currIdxA]);
            currIdxA++;
        }
        while (currIdxB != idxB) {
            if (llvm::is_contained(ret, b[currIdxB]))
                llvm_unreachable("Supremum does not exist");
            ret.push_back(b[currIdxB]);
            currIdxB++;
        }
        ret.push_back(a[idxA]);
    }

    while (currIdxA != a.size()) {
        if (llvm::is_contained(ret, a[currIdxA]))
            llvm_unreachable("Supremum does not exist");
        ret.push_back(a[currIdxA]);
        currIdxA++;
    }
    while (currIdxB != b.size()) {
        if (llvm::is_contained(ret, b[currIdxB]))
            llvm_unreachable("Supremum does not exist");
        ret.push_back(b[currIdxB]);
        currIdxB++;
    }

    return ret;
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

int32_t LinearLayout::getInDimSizeLog2(StringAttr Name) const {
    auto it = bases.find(Name);
    assert(it != bases.end() && "Dim not found!");
    return llvm::Log2_32(it->second.size());
};

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

LinearLayout::LinearLayout(
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
      ArrayRef<StringAttr> outDimNames) : LinearLayout(makeBasesMap(bases), outDimNames) {

}



} // end namespace mlir::clg

