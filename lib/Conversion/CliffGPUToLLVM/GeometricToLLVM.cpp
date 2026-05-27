#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

class CliffGeoProdOpPattern : public OpConversionPattern<GeoProd> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(GeoProd op, GeoProd::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override { 
        auto converter = getTypeConverter();
        RankedTensorType lhsTensor = op.getLhs().getType();
        RankedTensorType rhsTensor = op.getRhs().getType();
        RankedTensorType outTensor = op.getOut().getType();

        auto lhsTy = cast<Cliff_MultivectorType>(lhsTensor.getElementType());
        auto rhsTy = cast<Cliff_MultivectorType>(rhsTensor.getElementType());
        auto outTy = cast<Cliff_MultivectorType>(outTensor.getElementType());

        auto algebra = lhsTy.getSpace();

        unsigned p = algebra.getP(), q = algebra.getQ(), r = algebra.getR();

        const uint64_t lhsMask = lhsTy.getMask();
        const uint64_t rhsMask = rhsTy.getMask();
        const uint64_t outMask = outTy.getMask();

        uint64_t lhsMaskCopy = lhsMask;
        uint64_t rhsMaskCopy = rhsMask;
        uint64_t outMaskCopy = outMask;

        llvm::DenseMap<int, int> basisToOffset;
        int off = 0;
        while (outMaskCopy) {
            basisToOffset[__builtin_ctz(outMaskCopy)] = off;
            off++;
            outMaskCopy &= (outMaskCopy - 1);
        }

        // codegen
        auto loc = op.getLoc();
        auto fpTy = rewriter.getF32Type();
        
        auto adaptorLHS = adaptor.getLhs();
        auto lhsMultivector= LLVM::ExtractValueOp::create(rewriter, loc, adaptorLHS, 0);
        auto adaptorRHS = adaptor.getRhs();
        auto rhsMultivector= LLVM::ExtractValueOp::create(rewriter, loc, adaptorRHS, 0);

        SmallVector<Type> newResultTypes;
        if (failed(converter->convertTypes(outTensor, newResultTypes)))
            return rewriter.notifyMatchFailure(op, "failed to convert return type");

        assert(newResultTypes.size() == 1 && "GeoProd should return exactly 1 argument");
        
        Type returnType = newResultTypes[0];
        Value result = LLVM::UndefOp::create(rewriter, loc, returnType);

        // scalar case, e.g. ((e01 + 4) * (scalar))
        if (lhsMask == 1 || rhsMask == 1) {

            auto cstMultivector = (lhsMask == 1) ? lhsMultivector : rhsMultivector;
            auto dynMultivector = (lhsMask == 1) ? rhsMultivector : lhsMultivector;
            auto dynMask = (lhsMask == 1) ? rhsMaskCopy : lhsMaskCopy;
            auto cstScalar = LLVM::ExtractValueOp::create(rewriter, loc, cstMultivector, 0);
                
            int idx = 0;
            while (dynMask) {

                auto dynBasis = LLVM::ExtractValueOp::create(rewriter, loc, dynMultivector, idx);
                Value ret = b.fmul(cstScalar, dynBasis);
                result = LLVM::InsertValueOp::create(rewriter, loc, result, ret, {0, idx});
                dynMask &= dynMask - 1;
                idx++;
            }

            rewriter.replaceOp(op, result);
            return success();
        }

        int i = 0;

        while (lhsMaskCopy) {
            int j = 0;
            int lhsBasis = __builtin_ctz(lhsMaskCopy);
            int rhsMaskIter = rhsMaskCopy;

            auto currLhsBasis = LLVM::ExtractValueOp::create(rewriter, loc, lhsMultivector, i);

            while (rhsMaskIter) {
                int rhsBasis = __builtin_ctz(rhsMaskIter);
                int newBasis = lhsBasis ^ rhsBasis;
                int newSign = (!lhsBasis || !rhsBasis) ? 1 : 
                               reorderSign(lhsBasis, rhsBasis) * metricSign(lhsBasis, rhsBasis, p, q, r);

                if (newSign != 0) {

                    auto currRhsBasis = LLVM::ExtractValueOp::create(rewriter, loc, rhsMultivector, j);
                    Value prod = b.fmul(currLhsBasis, currRhsBasis);
                    Value ret = newSign==1 ? prod : b.neg(prod);
                    
                    int outIndex = basisToOffset[newBasis];
                    auto acc = LLVM::ExtractValueOp::create(rewriter, loc, result, {0, outIndex});
                    Value currOutVal = b.fadd(acc, ret);                    
                    result = LLVM::InsertValueOp::create(rewriter, loc, result, currOutVal, {0, outIndex});

                }
                
                rhsMaskIter &= rhsMaskIter - 1;
                ++j;
            }

            lhsMaskCopy &= lhsMaskCopy - 1;
            ++i;

        }
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

class CliffReverseOpPattern : public OpConversionPattern<Reverse> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(Reverse op, Reverse::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        auto converter = getTypeConverter();

        RankedTensorType srcTensor = dyn_cast<RankedTensorType>(op.getSrc().getType());
        RankedTensorType outTensor = dyn_cast<RankedTensorType>(op.getOut().getType());

        Cliff_MultivectorType srcMv = dyn_cast<Cliff_MultivectorType>(srcTensor.getElementType());
        
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
            // llvm::errs() << "arg " << i << " converted to " << converted.size() << " types\n";
            // for (auto t : converted) llvm::errs() << "  -> " << t << "\n";
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
                    CliffReverseOpPattern
                    >(typeConverter, context);
}
