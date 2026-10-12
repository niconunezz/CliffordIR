#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h" 
#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"

#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "clifford/Conversion/CliffGPUToLLVM/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/PopulatePatterns.h"


using namespace mlir;
using namespace mlir::cliff;
using namespace mlir::clg;

namespace {

enum class ProductKind {
    Geometric,
    Inner
};

struct Term {
    int32_t lhs;
    int32_t rhs;
    int32_t sign;
    int32_t out;
};

int32_t posOf(int32_t mask, int32_t basis) {
    return __builtin_popcountll(mask & ((1 << basis) - 1));
}


std::vector<Term> termGenerator(ProductKind kind, int lhsMaskCopy, int rhsMaskCopy, int outMaskCopy) {
    std::vector<Term> ret;

    llvm::DenseMap<int, int> basisToOffset;
        while (lhsMaskCopy) {
            int lhsBasis = __builtin_ctz(lhsMaskCopy);
            int rhsMaskIter = rhsMaskCopy;                
            while (rhsMaskIter) {

                int rhsBasis = __builtin_ctz(rhsMaskIter);
                int newSign = 0;
                switch (kind) {
                    case ProductKind::Geometric: {
                        bool hasTwoE0s = ((lhsBasis & 1) == 1) && ((rhsBasis & 1) == 1); 
                        int sign =  (!lhsBasis || !rhsBasis) ? 1 : reorderSign(lhsBasis, rhsBasis);
                        newSign = !hasTwoE0s * sign;
                        break;
                    }
                    case ProductKind::Inner: {
                        bool sameBasis = lhsBasis == rhsBasis;
                        bool hasTwoE0s = (lhsBasis & 1) == 1;
                        newSign = (sameBasis && !hasTwoE0s);
                        break;
                    }
                    }

                    if (newSign) {
                        int newBasis = lhsBasis ^ rhsBasis;
                        Term term = {lhsBasis, rhsBasis, newSign, newBasis};
                        ret.push_back(term);
                        
                    }
                    rhsMaskIter &= rhsMaskIter - 1;
                }
                
                lhsMaskCopy &= lhsMaskCopy - 1;
            }

    return ret;
};


template <class Op>
LogicalResult genericProduct(Op op, typename Op::Adaptor adaptor, ConversionPatternRewriter &rewriter, ProductKind kind, const TypeConverter* converter) {
        RankedTensorType lhsTensor = op.getLhs().getType();
        RankedTensorType rhsTensor = op.getRhs().getType();
        RankedTensorType outTensor = cast<RankedTensorType>(op.getOut().getType());

        auto loc = op.getLoc();
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        
        auto lhsTy = cast<Cliff_MultivectorType>(lhsTensor.getElementType());
        auto rhsTy = cast<Cliff_MultivectorType>(rhsTensor.getElementType());
        auto outTy = cast<Cliff_MultivectorType>(outTensor.getElementType());
 
        auto outActiveComps = __builtin_popcountll(outTy.getMask());

        const uint64_t lhsMask = lhsTy.getMask();
        const uint64_t rhsMask = rhsTy.getMask();
        const uint64_t outMask = outTy.getMask();

        std::vector<Term> terms = termGenerator(kind, lhsMask, rhsMask, outMask);
        // codegen
        auto llAttr =
            dyn_cast<LinearEncodingAttr>(outTensor.getEncoding());

        auto ll = llAttr.getLinearLayout();
        MLIRContext *ctx = rewriter.getContext();
        auto kReg = StringAttr::get(ctx, "register");
        int32_t registerDims =
            std::max(ll.getInDimSize(kReg), 1);


        
        auto adaptorLHS = adaptor.getLhs();
        auto adaptorRHS = adaptor.getRhs();
        SmallVector<Term> positionTerms;
        for (auto& t : terms) {
            auto& [lhs, rhs, sign, newBasis] = t;
            Term posT = {posOf(lhsMask, lhs), posOf(rhsMask, rhs), sign, posOf(outMask, newBasis)};
            positionTerms.push_back(posT);
        }
        
        // code emiter
        int32_t numBasisLhs = __builtin_popcount(lhsMask);
        int32_t numBasisRhs = __builtin_popcount(rhsMask);        

        
        SmallVector<Value> outMultivectors(registerDims);
        
        for (uint32_t regId = 0; regId < registerDims; ++regId) {            
            auto lhsMultivector= LLVM::ExtractValueOp::create(rewriter, loc, adaptorLHS, regId);
            auto rhsMultivector= LLVM::ExtractValueOp::create(rewriter, loc, adaptorRHS, regId);
            
            SmallVector<bool> usedLhs(numBasisLhs, false);
            SmallVector<bool> usedRhs(numBasisRhs, false);
            SmallVector<Value> extractOpsLhs(numBasisLhs);
            SmallVector<Value> extractOpsRhs(numBasisRhs);

            for (auto &t : positionTerms) {
                auto& [i, j, sign, outIndex] = t; 
                
                if (!usedLhs[i]) {
                    extractOpsLhs[i] = LLVM::ExtractValueOp::create(rewriter, loc, lhsMultivector, i);
                    usedLhs[i] = true;
                }

                if (!usedRhs[j]) {
                    extractOpsRhs[j] = LLVM::ExtractValueOp::create(rewriter, loc, rhsMultivector, j);
                    usedRhs[j] = true;
                }
            }
            
            SmallVector<Value> accVector(outActiveComps, b.f32_val(0.0f));

            for (auto &t : positionTerms) {
                auto& [i, j, sign, outIndex] = t;
                Value currLhsBasis = extractOpsLhs[i];
                Value currRhsBasis = extractOpsRhs[j];
                Value prod = b.fmul(currLhsBasis, currRhsBasis);
                Value ret = sign==1 ? prod : b.neg(prod);
                
                auto acc = accVector[outIndex];
                accVector[outIndex] = b.fadd(acc, ret);
            }
            Value packedAccs = packElements(loc, accVector, converter, rewriter, outTy);
            outMultivectors[regId] = packedAccs;
            
        }
        Value result = packElements(loc, outMultivectors, converter, rewriter, outTensor);

        rewriter.replaceOp(op, result);
        return success();
};


//!TODO: think if warp-reduction is possible
//something like: out[3] = a[3]*b[2] + a[5]*b[2] -> reduce with a warp?
class CliffGeoProdOpPattern : public OpConversionPattern<GeoProd> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(GeoProd op, GeoProd::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        auto converter = getTypeConverter();
        return genericProduct<GeoProd>(op, adaptor, rewriter, ProductKind::Geometric, converter);
    }
};

class CliffAddOpPattern : public OpConversionPattern<Add> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(Add op, Add::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto typeConverter = getTypeConverter();
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        RankedTensorType lhsTensor = cast<RankedTensorType>(op.getLhs().getType());
        RankedTensorType rhsTensor = cast<RankedTensorType>(op.getRhs().getType());
        RankedTensorType outTensor = cast<RankedTensorType>(op.getOut().getType());

        Cliff_MultivectorType lhsMv = cast<Cliff_MultivectorType>(lhsTensor.getElementType());
        Cliff_MultivectorType rhsMv = cast<Cliff_MultivectorType>(rhsTensor.getElementType());
        Cliff_MultivectorType outMv = cast<Cliff_MultivectorType>(outTensor.getElementType());

        const auto outMask = outMv.getMask();
        auto llLhsTensors = adaptor.getLhs();
        auto llRhsTensors = adaptor.getRhs();

        SmallVector<Value> llLhsElements = unpackElements(loc, llLhsTensors, rewriter);
        SmallVector<Value> llRhsElements = unpackElements(loc, llRhsTensors, rewriter);

        assert(llLhsElements.size() == llRhsElements.size() && "lhs handles more multivectors than rhs\n");
        
        SmallVector<Value> llOutElements;
        for (uint32_t i = 0; i < llLhsElements.size(); i++) {
            auto mvLhsEls = unpackElements(loc, llLhsElements[i], rewriter);
            auto mvRhsEls = unpackElements(loc, llRhsElements[i], rewriter);
            uint32_t lhsIdx = 0;
            uint32_t rhsIdx = 0;
            SmallVector<Value> mvOutEls;
            auto outMaskCopy = outMask;

            while (outMaskCopy) {
                int outBasis = __builtin_ctz(outMaskCopy);
                Value acc = b.f32_val(0.0f);
                if (lhsMv.isActiveComponent(outBasis)) {
                    acc = b.fadd(acc, mvLhsEls[lhsIdx]);
                    ++lhsIdx;
                }
                if (rhsMv.isActiveComponent(outBasis)) {
                    acc = b.fadd(acc, mvRhsEls[rhsIdx]);
                    ++rhsIdx;
                }
                mvOutEls.push_back(acc);
                outMaskCopy &= outMaskCopy - 1;
            }

            assert(lhsIdx == mvLhsEls.size() && "not all elements of lhs were used, check multivector masks");
            assert(rhsIdx == mvRhsEls.size() && "not all elements of rhs were used, check multivector masks");

            auto mvOutElsStruct = packElements(loc, mvOutEls, typeConverter, rewriter, outMv);
            llOutElements.push_back(mvOutElsStruct);
        }

        auto ret = packElements(loc, llOutElements, typeConverter, rewriter, outTensor);
        rewriter.replaceOp(op, ret);

        return success();
    }
};


class CliffReverseOpPattern : public OpConversionPattern<Reverse> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(Reverse op, Reverse::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        auto converter = getTypeConverter();

        RankedTensorType srcTensor = cast<RankedTensorType>(op.getSrc().getType());
        RankedTensorType outTensor = cast<RankedTensorType>(op.getOut().getType());

        Cliff_MultivectorType srcMv = cast<Cliff_MultivectorType>(srcTensor.getElementType());
        
        const auto mask = srcMv.getMask();
        auto maskCopy = mask;
        auto space = srcMv.getSpace();
        SmallVector<bool> mustNegate;
        while (maskCopy)
        {
            auto grade = __builtin_popcountll(__builtin_ctz(maskCopy));
            int exponent = (grade * (grade - 1)) / 2;
            mustNegate.push_back((exponent % 2) != 0);
            maskCopy &= (maskCopy-1);
        }
        
        auto adaptorSrc = adaptor.getSrc();
        auto srcMultivectors = unpackElements(loc, adaptorSrc, rewriter);

        SmallVector<Type> newResultTypes;
        if (failed(converter->convertTypes(outTensor, newResultTypes)))
            return rewriter.notifyMatchFailure(op, "failed to convert return type");

        assert(newResultTypes.size() == 1 && "GeoProd should return exactly 1 argument");
        
        Type returnType = newResultTypes[0];
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        SmallVector<Value> reversedMvs;
        for (auto &mv : srcMultivectors) {
            SmallVector<Value> reversedCoeffs;
            auto coeffs = unpackElements(loc, mv, rewriter);
            for (auto [neg, coeff] : llvm::zip(mustNegate, coeffs)) {
                auto newMv = neg ? b.neg(coeff) : coeff;
                reversedCoeffs.push_back(newMv);
            }
            
            Value coeffsStruct = packElements(loc, reversedCoeffs, converter, rewriter, srcMv);
            reversedMvs.push_back(coeffsStruct);
        }

        Value result = packElements(loc, reversedMvs, converter, rewriter, srcTensor);
        rewriter.replaceOp(op, result);
        return success();
    }
};

class CliffSinOpPattern : public OpConversionPattern<SinOp> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(SinOp op, SinOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {
        
        auto loc = op.getLoc();
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        auto fpTy = rewriter.getF32Type();

        auto tensorTy = cast<RankedTensorType>(op.getSrc().getType());
        auto mvTy = cast<Cliff_MultivectorType>(tensorTy.getElementType());

        SmallVector<Value> multivectors = unpackElements(loc, adaptor.getSrc(), rewriter);
        SmallVector<Value> outMultivectors;
        for (uint32_t i = 0; i < multivectors.size(); ++i) {

            SmallVector<Value> elements = unpackElements(loc, multivectors[i], rewriter);
            SmallVector<Value> outElements;
    
            for (uint32_t j = 0; j < elements.size(); ++j) {
                auto sin = b.sin(fpTy, elements[j]);
                sin.setFastmathFlags(LLVM::FastmathFlags::afn);
                outElements.push_back(sin);
            }
    
            auto els_ret = packElements(loc, outElements, typeConverter, rewriter, mvTy);
            outMultivectors.push_back(els_ret);
        }
        
        auto ret = packElements(loc, outMultivectors, typeConverter, rewriter, tensorTy);
        rewriter.replaceOp(op, ret);
        return success();
    }
};

class CliffCosOpPattern : public OpConversionPattern<CosOp> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(CosOp op, CosOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        auto fpTy = rewriter.getF32Type();

        auto tensorTy = cast<RankedTensorType>(op.getSrc().getType());
        auto mvTy = cast<Cliff_MultivectorType>(tensorTy.getElementType());

        SmallVector<Value> multivectors = unpackElements(loc, adaptor.getSrc(), rewriter);
        SmallVector<Value> outMultivectors;
        for (uint32_t i = 0; i < multivectors.size(); ++i) {

            SmallVector<Value> elements = unpackElements(loc, multivectors[i], rewriter);
            SmallVector<Value> outElements;
    
            for (uint32_t j = 0; j < elements.size(); ++j) {
                auto cos = b.cos(fpTy, elements[j]);
                cos.setFastmathFlags(LLVM::FastmathFlags::afn);
                outElements.push_back(cos);
            }
    
            auto els_ret = packElements(loc, outElements, typeConverter, rewriter, mvTy);
            outMultivectors.push_back(els_ret);
        }
        
        auto ret = packElements(loc, outMultivectors, typeConverter, rewriter, tensorTy);
        rewriter.replaceOp(op, ret);
        return success();
    }
};


class CliffReturnOpPattern : public OpConversionPattern<ReturnOp> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

class CliffFuncOpPattern :  public OpConversionPattern<FuncOp> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(FuncOp op, FuncOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
            
        auto converter = getTypeConverter();
        TypeConverter::SignatureConversion sigConversion(op.getNumArguments());
        
        for (unsigned i = 0; i < op.getNumArguments(); ++i) {
            SmallVector<Type> converted;
            if (failed(converter->convertType(op.getArgument(i).getType(), converted)))
                return rewriter.notifyMatchFailure(op, "Issue converting argument types");
            sigConversion.addInputs(i, converted);
        }
            
        SmallVector<Type> newResultTypes;
        if (failed(converter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return rewriter.notifyMatchFailure(op, "Issue converting function return types");
        
        SmallVector<Type> newArgTypes;
        for (const auto &arg : sigConversion.getConvertedTypes()) 
            newArgTypes.push_back(arg);

        Type returnType = newResultTypes.empty()
            ? LLVM::LLVMVoidType::get(getContext())
            : newResultTypes[0];

        auto llvmFnType = LLVM::LLVMFunctionType::get(returnType, newArgTypes,
                                                /*isVarArg=*/false);

        auto newFuncOp = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), op.getName(), llvmFnType);
        rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.getBody().end());

        newFuncOp->setAttr("nvvm.kernel", 
            rewriter.getIntegerAttr(rewriter.getIntegerType(1, false), 1));
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *converter, &sigConversion)))
            return rewriter.notifyMatchFailure(op, "Error occurred while converting region types");
        
        rewriter.eraseOp(op);
        return success();
    }
};


} // namespace

void mlir::cliff::populateGeometricToLLVMPatterns(CliffGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext* context = patterns.getContext();
    patterns.insert<CliffReturnOpPattern, 
                    CliffFuncOpPattern, 
                    CliffGeoProdOpPattern,
                    CliffReverseOpPattern,
                    CliffSinOpPattern,
                    CliffCosOpPattern,
                    CliffAddOpPattern
                    >(typeConverter, context);
}
