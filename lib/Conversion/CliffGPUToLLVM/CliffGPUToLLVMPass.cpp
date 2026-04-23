#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"

#include "clifford/Dialect/CliffGPU/IR/Dialect.h"


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "clifford/Conversion/CliffGPUToLLVM/TypeConverter.h"

namespace mlir::clg {
#define GEN_PASS_DEF_CONVERTCLIFFGPUTOLLVM
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h.inc"

} // namespace mlir::clg



namespace {

using namespace mlir;
using namespace mlir::cliff;
using namespace mlir::clg;

class CliffGeoProdOpPattern : public OpConversionPattern<GeoProd> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(GeoProd op, GeoProd::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override { 
        auto converter = getTypeConverter();
        RankedTensorType lhsTensor = dyn_cast<RankedTensorType>(op.getLhs().getType());
        RankedTensorType rhsTensor = dyn_cast<RankedTensorType>(op.getRhs().getType());
        RankedTensorType outTensor = dyn_cast<RankedTensorType>(op.getOut().getType());
        if (!lhsTensor || !rhsTensor || !outTensor)
            return rewriter.notifyMatchFailure(op, "operands must be ranked tensors");
        auto lhsTy = dyn_cast<Cliff_MultivectorType>(lhsTensor.getElementType());
        auto rhsTy = dyn_cast<Cliff_MultivectorType>(rhsTensor.getElementType());
        auto outTy = dyn_cast<Cliff_MultivectorType>(outTensor.getElementType());
        if (!lhsTy || !rhsTy || !outTy)
            return rewriter.notifyMatchFailure(op, "tensor elements must be multivector types");
        
        if (lhsTy.getSpace() != rhsTy.getSpace() || 
            lhsTy.getSpace() != outTy.getSpace())
            return rewriter.notifyMatchFailure(op, "all operands must belong to the same algebra");

        auto algebra = dyn_cast<CliffordAlgebraAttr>(lhsTy.getSpace());
        if (!algebra)
            return rewriter.notifyMatchFailure(op, "algebra attribute must be a CliffordAlgebraAttr");

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
                Value ret = arith::MulFOp::create(rewriter, loc, cstScalar, dynBasis);
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
                    Value prod = arith::MulFOp::create(rewriter, loc, currLhsBasis, currRhsBasis);
                    Value ret = newSign==1 ? prod : arith::NegFOp::create(rewriter, loc, prod);
                    
                    int outIndex = basisToOffset[newBasis];
                    auto acc = LLVM::ExtractValueOp::create(rewriter, loc, result, {0, outIndex});
                    Value currOutVal = arith::AddFOp::create(rewriter, loc, acc, ret);                    
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

        
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *converter, &sigConversion)))
            return rewriter.notifyMatchFailure(op, "Error occurred while converting region types");
        
        rewriter.eraseOp(op);
        return success();
    }
};

class RewriteRotatePattern : public OpRewritePattern<Rotate> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(Rotate op,
                                  PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto fpTy = rewriter.getF32Type();
        
        Value cos = math::CosOp::create(rewriter, loc, fpTy, op.getAngle());
        Value sin = math::SinOp::create(rewriter, loc, fpTy, op.getAngle());
        Value geoProd = GeoProd::create(rewriter, loc, fpTy, sin, op.getSrc());
        Value ret = arith::AddFOp::create(rewriter, loc, cos, geoProd);

        rewriter.replaceOp(op, ret);
        return success();
    }

};


class RewriteTranslatePattern : public OpRewritePattern<Translate> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(Translate op,
                                  PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto fpTy = rewriter.getF32Type();
        
        Value one = arith::ConstantOp::create(rewriter, loc,
                    rewriter.getFloatAttr(fpTy, 1.0));
        Value scaledMv = GeoProd::create(rewriter, loc, fpTy, op.getDist(), op.getSrc());
        Value ret = arith::AddFOp::create(rewriter, loc, one, scaledMv);

        rewriter.replaceOp(op, ret);
        return success();
    }

};

void populateCliffGPUToLLVMPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext* context = patterns.getContext();
    patterns.insert<CliffReturnOpPattern, CliffFuncOpPattern, CliffGeoProdOpPattern>(typeConverter, context);
}


class ConvertCliffGPUToLLVM : public clg::impl::ConvertCliffGPUToLLVMBase<ConvertCliffGPUToLLVM> {
public:
using ConvertCliffGPUToLLVMBase::ConvertCliffGPUToLLVMBase;

    void runOnOperation() override {

        {
            // prologue
            RewritePatternSet patterns(&getContext());
            patterns.insert<RewriteRotatePattern,
                            RewriteTranslatePattern>(&getContext());
            
            if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
                signalPassFailure();
        }
        {

            MLIRContext *context = &getContext();
            CliffGPUToLLVMTypeConverter typeConverter(context);
            CliffGPUToLLVMConversionTarget conversionTarget(*context, typeConverter);
            RewritePatternSet patterns(context);
            
            populateCliffGPUToLLVMPatterns(typeConverter, patterns);
            
            if (failed(applyPartialConversion(getOperation(), conversionTarget, std::move(patterns)))) 
            return signalPassFailure();
            
        }

    }

};

}