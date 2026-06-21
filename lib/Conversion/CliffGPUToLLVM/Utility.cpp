#include "llvm/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Tools/LinearLayout.h"

using namespace llvm;
using namespace mlir::cliff;
using namespace mlir::clg;
using namespace mlir;

namespace mlir::clg {

        
    Value matrixVectorProd(CliffordLLVMOpBuilder &b, const LinearLayout &A, Value x) {
        assert(A.getNumInDims() == 1);
        assert(A.getNumOutDims() == 1);
        auto flatten = [](const std::vector<std::vector<int32_t>> &matrix) {
            SmallVector<int32_t> ret;
            for (const auto &row : matrix) {
            ret.push_back(row[0]);
            }
            return ret;
        };
        auto nCol = A.getTotalInDimSizeLog2();
        auto nRow = A.getTotalOutDimSizeLog2();
        SmallVector<int32_t> matrix = flatten(A.getBases().begin()->second);
        assert(matrix.size() == nCol);

        // Row-wise popcount to detect rows that appear exactly once across columns.
        uint32_t rowsUnique = 0;
        {
            SmallVector<int> rowPopCnt(nRow, 0);
            for (int c = 0; c < nCol; ++c) {
            uint32_t colBits = matrix[c];
            for (int r = 0; r < nRow; ++r) {
                if (colBits & (1u << r))
                ++rowPopCnt[r];
            }
            }
            for (int r = 0; r < nRow; ++r) {
            if (rowPopCnt[r] == 1)
                rowsUnique |= 1u << r;
            }
        }

        // We iterate the matrix following the diagonals and build
        // (x & mask_i) << s_i terms. Prefer OR for diagonals whose rows are unique,
        // then XOR everything else. This tends to encourage mad.lo codegen.
        auto getMaskAndAllRowsUnique = [&](int i) -> std::pair<uint32_t, bool> {
            uint32_t mask = 0;
            int row = i < 0 ? -i : 0;
            int col = i < 0 ? 0 : i;
            bool allRowsUnique = true;
            while (row < nRow && col < nCol) {
            uint32_t bitValue = (matrix[col] >> row) & 1u;
            mask |= bitValue << col;
            allRowsUnique &= ((rowsUnique >> row) & 1u) == 1u;
            ++row;
            ++col;
            }
            return {mask, allRowsUnique};
        };

        uint32_t explicitCols = 0;

        {
            SmallVector<uint32_t> masks;
            for (int i = -nRow + 1; i < nCol; i++) {
            masks.push_back(std::get<0>(getMaskAndAllRowsUnique(i)));
            }
            bool reachedFixedPoint = false;
            while (!reachedFixedPoint) {
            reachedFixedPoint = true;
            for (uint32_t m : masks) {
                uint32_t c = m & ~explicitCols;
                if (llvm::isPowerOf2_32(c)) {
                // found a single-element diagonal
                explicitCols |= c;
                reachedFixedPoint = false;
                }
            }
            }
        }

        // handle any diagonals that have survived
        SmallVector<Value> ors;
        SmallVector<Value> xors;
        for (int i = -nRow + 1; i < nCol; i++) {
            auto [mask, allRowsUnique] = getMaskAndAllRowsUnique(i);
            mask &= ~explicitCols;
            if (mask == 0)
            continue;
            auto masked = b.and_(x, b.i32_val(mask));
            auto shifted = i >= 0 ? Value(b.lshr_(masked, b.i32_val(i)))
                                : Value(b.shl_(masked, b.i32_val(-i)));
            if (allRowsUnique) {
            ors.push_back(shifted);
            } else {
            xors.push_back(shifted);
            }
        }

        // handle any explicit columns:
        Value zero = b.i32_val(0);
        for (int i = 0; i < nCol; i++) {
            if ((explicitCols >> i) & 1) {
            int32_t basis = matrix[i];
            if (basis == 0)
                continue;
            Value term;
            Value bit = b.and_(x, b.i32_val(1 << i));
            uint32_t basisBits = basis;
            if (llvm::isPowerOf2_32(basisBits)) {
                unsigned row = llvm::countr_zero(basisBits);
                unsigned col = i;
                if (row == col)
                term = bit;
                else if (row > col)
                term = b.shl_(bit, b.i32_val(row - col));
                else
                term = b.lshr_(bit, b.i32_val(col - row));
            } else {
                Value bit_is_zero = b.icmp_eq(bit, zero);
                term = b.select(bit_is_zero, zero, b.i32_val(basis));
            }
            if ((rowsUnique & basis) == basis) {
                ors.push_back(term);
            } else {
                xors.push_back(term);
            }
            }
        }

        auto treeReduce = [&](SmallVector<Value> &terms,
                                std::function<Value(Value, Value)> op) -> Value {
            if (terms.empty())
            return b.i32_val(0);
            while (terms.size() > 1) {
            SmallVector<Value> next;
            for (size_t i = 0; i + 1 < terms.size(); i += 2)
                next.push_back(op(terms[i], terms[i + 1]));
            if (terms.size() % 2 == 1)
                next.push_back(terms.back());
            terms = std::move(next);
            }
            return terms[0];
        };

        auto orPart = treeReduce(
            ors, [&b](Value x, Value y) { return b.or_(x, y, /*disjoint=*/true); });
        auto xorPart =
            treeReduce(xors, [&b](Value x, Value y) { return b.xor_(x, y); });
        return b.or_(orPart, xorPart, /*disjoint=*/true);
            }

    
    SmallVector<Value> unpackElements(Location loc, Value llvmStruct, RewriterBase &rewriter) { 

        ArrayRef<mlir::Type> types = cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
        
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        SmallVector<Value> ret;
        for (unsigned i = 0; i < types.size(); ++i) {
            Type type = types[i];
            ret.push_back(b.extract(type, llvmStruct, i));
        }
        return ret;
    }

    Value packElements(Location loc, ValueRange resultVals,
                       const TypeConverter *typeConverter,
                       RewriterBase &rewriter, Type type) {
        auto structType = cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        Value result = LLVM::UndefOp::create(rewriter, loc, structType);
        auto structTypes = structType.getBody();

        for (int i = 0; i < resultVals.size(); ++i)
            result = b.insert(structType, result, resultVals[i], i);
        
        return result;
    }

    SmallVector<Value> computeIndices(Location loc, LinearLayout &layout,
                                                                ArrayRef<Value> x,
                                                                RewriterBase &rewriter) {
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        SmallVector<Value> ret;
        uint32_t numOutDims = llvm::size(layout.getOutDimNames());
        llvm::MapVector<uint32_t, std::vector<std::pair<StringAttr, std::vector<int32_t>>>> outDimIdxToPerDimValues;

        for (uint32_t outDimIdx = 0; outDimIdx < numOutDims; outDimIdx++) {

            for (auto &[inDimName, inDimBases] : layout.getBases()) {
                std::vector<int32_t> valuesPerDim;
                for (auto &basis : inDimBases) {
                    int32_t el = basis[outDimIdx];
                    if (el != 0)
                        valuesPerDim.push_back(el);
                }
                outDimIdxToPerDimValues[outDimIdx].emplace_back(inDimName, valuesPerDim);
            }
        }

        uint32_t numInDims = llvm::size(layout.getInDimNames());
        SmallVector<uint32_t> shifts(numInDims, 1);
        for (uint32_t outDimIdx = 0; outDimIdx < numOutDims; ++outDimIdx) {
            Value zeroV = b.i32_val(0);
            auto res = zeroV;
            uint32_t outerIdx = 0;
            
            for (auto &[inDimName, values] : outDimIdxToPerDimValues[outDimIdx]) {
                // x[outerIdx].dump();
                auto numValues = layout.getInDimSizeLog2(inDimName);
                for (auto &val : values) {
                    Value activeBit = b.and_(x[outerIdx], b.i32_val(1 << (numValues-shifts[outerIdx])));
                    Value isActive = b.icmp_ne(activeBit, zeroV);
                    Value contrib = b.select(isActive, b.i32_val(val), zeroV);
                    res = b.add(res, contrib);
                    ++shifts[outerIdx];
                }
                ++outerIdx;
            }

            ret.push_back(res);
        }
        
        return ret;
    }
    SmallVector<std::pair<StringAttr, Value>> applyLinearLayout(Location loc, LinearLayout &layout,
                                                                ArrayRef<std::pair<StringAttr, Value>> indices,
                                                                const TypeConverter *typeConverter, RewriterBase &rewriter) {

        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        SmallVector<std::pair<StringAttr, int32_t>> constantIns;
        SmallVector<std::pair<StringAttr, Value>> nonConstantIns;
        SmallVector<std::pair<StringAttr, Value>> ret;

        for (auto const &[inDimName, value] : indices) {
            APInt constant;
            if (matchPattern(value, m_ConstantInt(&constant))) {
                constantIns.push_back({inDimName, (int32_t)constant.getSExtValue()});
            }
            else {
                constantIns.push_back({inDimName, 0});
                nonConstantIns.push_back({inDimName, value});
            }
        }

        for (auto const &[outDimName, outVal] : layout.apply(constantIns)) {
            ret.push_back({outDimName, b.i32_val(outVal)});            
        }
        auto zero = b.i32_val(0);
        Value x = b.i32_val(0);
        int32_t offset = 0;
        SmallVector<StringAttr> inDimNames;
        for (auto &[inDimName, value] : nonConstantIns) {
            inDimNames.push_back(inDimName);
            x = b.or_(x, b.shl_(x, b.i32_val(offset)));
            offset += layout.getInDimSizeLog2(inDimName);
        }

        for (auto &[outDimName, outVal] : ret) {
            auto matrix = layout.sublayout(inDimNames, outDimName);
            auto out = clg::matrixVectorProd(b, matrix, x);
            outVal = b.xor_(outVal, out);
        }
        return ret;
    }
}