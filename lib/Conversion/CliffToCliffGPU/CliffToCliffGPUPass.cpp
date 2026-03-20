#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "clifford/Conversion/CliffToCliffGPU/Passes.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/Transforms/CliffGPUConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"




namespace mlir::cliff {
#define GEN_PASS_DEF_CONVERTCLIFFTOCLIFFGPU
#include "clifford/Conversion/CliffToCliffGPU/Passes.h.inc"


} // namespace mlir::cliff 


namespace {

using namespace mlir;
using namespace mlir::cliff;
using namespace mlir::clg;


template <class Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
    using OpConversionPattern<Op>::OpConversionPattern;

    LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        
        SmallVector<Type> retTypes;
        if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), retTypes))) 
            return failure();
        
        rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(), op->getAttrs());

        return success();
    }
};


class CliffFuncOpPattern : public OpConversionPattern<FuncOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FuncOp op, FuncOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        auto converter = getTypeConverter();
        TypeConverter::SignatureConversion result(op.getNumArguments());
        auto newOp = rewriter.replaceOpWithNewOp<FuncOp>(op, op.getName(), op.getFunctionType(), nullptr, nullptr, nullptr);
        newOp->setAttrs(adaptor.getAttributes());
        rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.getBody().end());

        if (!newOp.getBody().empty())
            rewriter.applySignatureConversion(&newOp.getBody().front(), result, converter);
        
        return success();
    }
};

void populateCliffPatterns(CliffGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext *context = patterns.getContext();
    patterns.insert<
        GenericOpPattern<Return>,
        CliffFuncOpPattern
    >(typeConverter, context);

}


class ConvertCliffToCliffGPU : public cliff::impl::ConvertCliffToCliffGPUBase<ConvertCliffToCliffGPU> {

public:

using ConvertCliffToCliffGPUBase::ConvertCliffToCliffGPUBase;


void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    CliffGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp);
    CliffGPUConversionTarget conversionTarget(*context, typeConverter);
    RewritePatternSet patterns(context);

    populateCliffPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, conversionTarget, std::move(patterns)))) 
        return signalPassFailure();
    
}

};


} // namespace