#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "clifford/Conversion/CliffToCliffGPU/Passes.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "clifford/Dialect/CliffGPU/Transforms/CliffGPUConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"




namespace mlir::cliff {
    #define GEN_PASS_DEF_CONVERTCLIFFTOCLIFFGPU
    #include "clifford/Conversion/CliffToCliffGPU/Passes.h.inc"

}


namespace {

using namespace mlir;
using namespace mlir::cliff;
using namespace mlir::clg;



class ConvertCliffToCliffGPU : public cliff::impl::ConvertCliffToCliffGPUBase<ConvertCliffToCliffGPU> {

public:

using ConvertCliffToCliffGPUBase::ConvertCliffToCliffGPUBase;


class CliffReturnPattern : public OpConversionPattern<Return> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(Return op, Return::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        
        rewriter.replaceOpWithNewOp<Return>(op, adaptor.getOperands());
        return success();
    } 
};

class CliffFuncOpPatter : public OpConversionPattern<FuncOp> {
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

void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    CliffGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp);
    CliffGPUConversionTarget conversionTarget(*context, typeConverter);
}

};


}