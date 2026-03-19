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

void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    CliffGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp);
    CliffGPUConversionTarget conversionTarget(*context, typeConverter);
}

};


}