#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/IR/Dialect.h"

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


// template <Class Op>
// struct GenericOpPattern : public OpConversionPattern<FuncOp> {
// public:
//     using OpConversionPattern<Op>::OpConversionPattern;

//     LogicalResult matchAndRewrite(Op op, Op::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {
//         ...
//     }
// }

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
    patterns.insert<CliffReturnOpPattern, CliffFuncOpPattern>(typeConverter, context);
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