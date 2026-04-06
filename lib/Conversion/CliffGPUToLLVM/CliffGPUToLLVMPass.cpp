#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

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

struct ProductTerm {
    int resultBit;
    int lhsBit;
    int rhsBit;
    int sign;
};



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
        uint64_t lhsMask = lhsTy.getMask();
        uint64_t rhsMask = rhsTy.getMask();
        uint64_t outMask = outTy.getMask();

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

        Type returnType = newResultTypes.empty()
            ? LLVM::LLVMVoidType::get(getContext())
            : newResultTypes[0];

        // scalar case
        if (!lhsMask || !rhsMask) {

            auto cstMultivector = !lhsMask ? lhsMultivector : rhsMultivector;
            auto dynMultivector = !lhsMask ? rhsMultivector : lhsMultivector;
            auto dynMask = !lhsMask ? rhsMask : lhsMask;
            auto cstScalar = LLVM::ExtractValueOp::create(rewriter, loc, cstMultivector, 0);
                
            while (dynMask) {
                int dynIdx = __builtin_ctz(dynMask);
                    
                auto dynBasis = LLVM::ExtractValueOp::create(rewriter, loc, dynMultivector, dynIdx);
                    
                Value ret = arith::MulFOp::create(rewriter, loc, cstScalar, dynBasis);
                    
                LLVM::InsertValueOp::create(rewriter, loc, result, dynIdx )
                dynMask &= dynMask - 1;
            }
            rewriter.eraseOp(op);
            return success();
        }


        // SmallVector<ProductTerm> productTerms;
        // while (lhsMask) {
        //     int lhsBasis = __builtin_ctz(lhsMask);
        //     while (rhsMask) {
        //         int rhsBasis = __builtin_ctz(rhsMask);

        //         int newBasis = lhsBasis ^ rhsBasis;
        //         int newSign = reorderSign(lhsBasis, rhsBasis) * metricSign(lhsBasis, rhsBasis, p, q, r);
        //         if (newSign != 0) {
        //             ProductTerm term = {newBasis, lhsBasis, rhsBasis, newSign};
        //             productTerms.push_back(term);

        //         }
                
        //         rhsMask &= rhsMask - 1;
        //     }

        //     lhsMask &= lhsMask - 1;
        // }

        

        // for (auto &term : productTerms) {
        //     auto [newBasis, lhsBasis, rhsBasis, newSign] = term;
        //     auto fpTy = rewriter.getF32Type();
        //     auto lhs_i = rewriter.create<LLVM::ExtractValueOp>(loc, fpTy, adaptorLHSMask, ArrayRef<int64_t>{lhsBasis});
        //     auto rhs_j = rewriter.create<LLVM::ExtractValueOp>(loc, fpTy, adaptorRHSMask, ArrayRef<int64_t>{rhsBasis});

        //     Value sign_i = rewriter.create<arith::ConstantOp>(loc, newSign);
        //     Value lhs_rhs = rewriter.create<arith::MulFOp>(loc, lhs_i, rhs_i);
        //     Value result = rewriter.create<arith::MulFOp>(loc, lhs_rhs, sign_i);

        //     rewriter.create<LLVM::InsertValueOp>(loc, result, component, )


        // }


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
            llvm::errs() << "arg " << i << " converted to " << converted.size() << " types\n";
            for (auto t : converted) llvm::errs() << "  -> " << t << "\n";
            sigConversion.addInputs(i, converted);
        }
            
        SmallVector<Type> newResultTypes;
        if (failed(converter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return rewriter.notifyMatchFailure(op, "Issue converting function return types");
        llvm::errs() << "Converted result type : " << newResultTypes[0] << "\n";
        
        SmallVector<Type> newArgTypes;
        for (const auto &arg : sigConversion.getConvertedTypes()) 
            newArgTypes.push_back(arg);
        llvm::errs() << "Converted arg type : " << newArgTypes[0] << "\n";

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

void populateCliffGPUToLLVMPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext* context = patterns.getContext();
    patterns.insert<CliffReturnOpPattern, CliffFuncOpPattern, CliffGeoProdOpPattern>(typeConverter, context);
}


class ConvertCliffGPUToLLVM : public clg::impl::ConvertCliffGPUToLLVMBase<ConvertCliffGPUToLLVM> {
public:
using ConvertCliffGPUToLLVMBase::ConvertCliffGPUToLLVMBase;

    void runOnOperation() override {

        MLIRContext *context = &getContext();
        ModuleOp mod = getOperation();
        CliffGPUToLLVMTypeConverter typeConverter(context);
        CliffGPUToLLVMConversionTarget conversionTarget(*context, typeConverter);
        RewritePatternSet patterns(context);

        populateCliffGPUToLLVMPatterns(typeConverter, patterns);

        if (failed(applyPartialConversion(mod, conversionTarget, std::move(patterns)))) 
            return signalPassFailure();
        

    }

};

}