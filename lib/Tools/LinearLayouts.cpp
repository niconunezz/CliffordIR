#include "clifford/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::clg {

using BasesT = LinearLayout::BasesT;
using namespace mlir::cliff;
using namespace llvm;

BasesT makeBasesMap(ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases) {
    BasesT retBases;
    for (auto const &[inDim, inDimBases] : bases) {
        retBases[inDim] = inDimBases;
    }
    return retBases;
}
llvm::MapVector<StringAttr, int32_t> makeOutDimsMap(ArrayRef<std::pair<StringAttr, int32_t>> bases) {
    llvm::MapVector<StringAttr, int32_t> retBases;
    for (auto const &[outDim, outDimBases] : bases) {
        retBases[outDim] = outDimBases;
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
    for (int32_t i = 1; i < N; i *= 2) {
        bases.emplace_back(std::vector<int32_t>{i});
    }

    return LinearLayout({{inDimName, std::move(bases)}}, {outDimName});
}

/*static*/ LinearLayout LinearLayout::zeros1D(unsigned N, StringAttr inDimName, StringAttr outDimName) {
    assert(llvm::isPowerOf2_32(N) && "N must be a power of 2");
    std::vector<std::vector<int32_t>> bases;
    for (int32_t i = 1; i < N; i *= 2) {
        bases.emplace_back(std::vector<int32_t>{0});
    }

    return LinearLayout({{inDimName, std::move(bases)}}, {outDimName});
}

LinearLayout LinearLayout::sublayout(ArrayRef<StringAttr> inDimNames, ArrayRef<StringAttr> outDimNames) const {
        SmallDenseSet<StringAttr> inDimNamesSet(inDimNames.begin(), inDimNames.end());
        SmallDenseSet<StringAttr> outDimNamesSet(outDimNames.begin(), outDimNames.end());

        SmallVector<int32_t> outDimIndices;
        SmallVector<StringAttr> validOutDims;
        for (auto [idx, outDim] : llvm::enumerate(outDimNamesSet)) {
            if (outDims.contains(outDim)) {
                outDimIndices.push_back(idx);
                
            }
        }

        BasesT newBases;
        for (auto &[dimName, inDimBases] : bases) {
            if (inDimNamesSet.contains(dimName)) {
                auto &newDimBase = newBases[dimName];
                for (auto &basis : inDimBases) {
                    auto& newBasis = newDimBase.emplace_back();
                    for (auto idx : outDimIndices) {
                        newBasis.push_back(basis[idx]);    
                    }
                }
            }
        }

        for (auto &[outDimName, outDimBases] : outDims) {
            if (outDimNamesSet.contains(outDimName))
                validOutDims.push_back(outDimName);
            
        }

        return LinearLayout(newBases, validOutDims);
    }

SmallVector<StringAttr> getSupremum(const SmallVector<StringAttr> &a, const SmallVector<StringAttr> &b) {
    SmallVector<StringAttr> ret;
    const int stride = b.size() + 1;    
    int m = a.size(), n = b.size();
    std::vector<int16_t> lcs((m+1)*stride, 0);
    for (auto i=0; i < a.size(); i++) {
        for (auto j=0; j < b.size(); j++) {
            lcs[(i + 1)*stride + j + 1] =
                (a[i] == b[j]) ? lcs[i*stride + j] + 1 : std::max(lcs[(i+1)*stride + j], lcs[i*stride + j+1]);
        }
    }
        
    const int16_t lcs_size = lcs[a.size()*stride + b.size()];
    // backtracing
    std::vector<std::pair<int16_t, int16_t>> anchorPositions(lcs_size);
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
        currIdxA++; currIdxB++;
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
    auto innerInDim = llvm::to_vector((inner.getInDimNames()));
    auto innerOutDim = llvm::to_vector((inner.getOutDimNames()));
    auto outerInDim = llvm::to_vector((outer.getInDimNames()));
    auto outerOutDim = llvm::to_vector((outer.getOutDimNames()));

    SmallVector<StringAttr> retInDims = getSupremum(innerInDim,
                                                    outerInDim);

    SmallVector<StringAttr> retOutDims = getSupremum(innerOutDim,
                                                     outerOutDim);

    // int retInDimsSize = retInDims.size();
    int retOutDimsSize = retOutDims.size();

    BasesT innerBases = inner.getBases();
    BasesT outerBases = outer.getBases();
    BasesT retBases;
    
    for (const StringAttr &inDimName : retInDims) {
        
        bool isDimInInner = innerBases.contains(inDimName);
        bool isDimInOuter = outerBases.contains(inDimName);
        
        int innerNumBases = isDimInInner ? innerBases[inDimName].size() : 0;
        int outerNumBases = isDimInOuter ? outerBases[inDimName].size() : 0;
        int numBases = innerNumBases + outerNumBases;

        int basisIdx = 0;
        std::vector<std::vector<int>> retBasis(numBases, std::vector<int>(retOutDimsSize, 0));

        if (isDimInInner) {
            std::vector<std::vector<int>> innerDimBases = innerBases[inDimName];
    
            for (const auto &innerBasis : innerDimBases) {

                for (const auto [idx, elem] : llvm::enumerate(innerBasis)) {
                    retBasis[basisIdx][idx] = elem;
                }
                basisIdx++;
            }
        }

        if (isDimInOuter) {
            
            std::vector<std::vector<int>> outerDimBases = outerBases[inDimName];

            int outerOutNumDims = llvm::to_vector(outer.getOutDimNames()).size();
            int idxOffset = retOutDimsSize - outerOutNumDims;

            for (auto const &outerBasis : outerDimBases) {
                for (auto [idx, elem] : llvm::enumerate(outerBasis)) {
                    
                    int realIdx = idxOffset + idx;
                    // if idxOffset+1 <= innerOutDimsSize -> outDim overlaps
                    // k = (innerOutDimsSize - idxOffset) dims which are the
                    // last k innerOutDims dimensions.
                    int k = innerOutDim.size() - idxOffset;

                    int shift = idx < k ? inner.getOutDimSizeLog2(outerOutDim[idx]) : 0;
                
                    retBasis[basisIdx][realIdx] = elem << shift;

                }
                basisIdx++;
            }
        }

        retBases[inDimName] = retBasis;
    }

    return LinearLayout(retBases, retOutDims);
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
    return (it->second).size();
};

int32_t LinearLayout::getTotalInDimSizeLog2() const {
    auto inDimNames = getInDimNames();
    return std::accumulate(inDimNames.begin(),
                           inDimNames.end(), 
                           0, 
                           [&](uint32_t acc, StringAttr dimName){
                                return acc + getInDimSizeLog2(dimName); }
                          );
};

int32_t LinearLayout::getTotalOutDimSizeLog2() const {
    auto outDimNames = getOutDimNames();
    return std::accumulate(outDimNames.begin(),
                           outDimNames.end(), 
                           0, 
                           [&](uint32_t acc, StringAttr dimName){
                                return acc + getOutDimSizeLog2(dimName); }
                          );
};


int32_t LinearLayout::getOutDimSizeLog2(StringAttr Name) const {
    auto it = outDims.find(Name);
    assert(it != outDims.end() && "Dim not found!");
    return llvm::Log2_32(it->second);
};

std::string LinearLayout::toString() const {
  // Start with a newline because we print out a bulleted list; it doesn't
  // make sense for the first line of this list to be on the same line as
  // any previous text.
  std::string ret = "\n";
  std::string outDimsStr =
      "[" +
      join(outDims, ", ",
           [](auto dimAndSize) {
             auto [outDim, size] = dimAndSize;
             return outDim.str() + " (size " + std::to_string(size) + ")";
           }) +
      "]";

  if (bases.empty()) {
    if (outDims.empty()) {
      return "\n(empty layout)";
    } else {
      return "\n(empty layout with out-dims " + outDimsStr + ")";
    }
  }

  // TODO: Add spaces for alignment.
  for (const auto &[inDim, inDimBases] : bases) {
    if (inDimBases.empty()) {
      ret += " - " + inDim.str() + " is a size 1 dimension\n";
      continue;
    }

    ret += " - " +
           join(llvm::seq(inDimBases.size()), "\n   ",
                [&, &inDim = inDim, &inDimBases = inDimBases](int i) {
                  return inDim.str() + "=" + std::to_string(1 << i) + " -> (" +
                         join(inDimBases[i], ", ") + ")";
                }) +
           "\n";
  }
  ret += "where out dims are: " + outDimsStr;
  return ret;
}

SmallVector<std::pair<StringAttr, int32_t>> LinearLayout::apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const {
    SmallVector<std::pair<StringAttr, int32_t>> ret;

    for (StringAttr outDim : getOutDimNames()) {
        int32_t out = 0;
        for (auto &[inDimName, val] : ins) {
            for (int i = 0; i < getInDimSizeLog2(inDimName); ++i) {
                if (val & (1<<i))
                    out ^= getBasis(inDimName, i, outDim);
            }
        }
        ret.push_back({outDim, out});
    }
    return ret;
}

LinearLayout::LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames)
 : bases(std::move(bases)) {

    for (StringAttr outDim : outDimNames) {
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
      ArrayRef<std::pair<StringAttr, int32_t>> outDims) : LinearLayout(makeBasesMap(bases), makeOutDimsMap(outDims)) {};

LinearLayout::LinearLayout(BasesT bases, llvm::MapVector<StringAttr, int32_t> outDims) : bases(bases), outDims(outDims) {};

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<StringAttr> outDimNames) : LinearLayout(makeBasesMap(bases), outDimNames) {
}



} // end namespace mlir::clg

