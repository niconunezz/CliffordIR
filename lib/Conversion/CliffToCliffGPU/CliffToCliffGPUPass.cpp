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

static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

class CliffFuncOpPattern : public OpConversionPattern<FuncOp> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FuncOp op, FuncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
        auto converter = getTypeConverter();

        TypeConverter::SignatureConversion sigConversion(op.getNumArguments());
        SmallVector<Type> newResultTypes;

        for (unsigned i = 0; i < op.getNumArguments(); ++i) {
            SmallVector<Type> converted;
            if (failed(converter->convertType(op.getArgument(i).getType(), converted)))
                return failure();
            sigConversion.addInputs(i, converted);
        }

        if (failed(converter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return failure();

        SmallVector<Type> newArgTypes;
        for (auto &input : sigConversion.getConvertedTypes())
            newArgTypes.push_back(input);

        auto newFuncType = FunctionType::get(getContext(), newArgTypes, newResultTypes);

        // Crear la nueva FuncOp SIN mover la región todavía
       auto newFunc = FuncOp::create(rewriter, op.getLoc(), op.getName(), newFuncType, 
                               /*sym_visibility=*/nullptr, 
                               /*arg_attrs=*/nullptr, 
                               /*res_attrs=*/nullptr);
        addNamedAttrs(newFunc, adaptor.getAttributes());

        rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.getBody().end());

        if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *converter, &sigConversion)))
            return failure();

        rewriter.replaceOp(op, newFunc);
        return success();
    }
};

void populateCliffPatterns(CliffGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext *context = patterns.getContext();
    patterns.insert<
        CliffFuncOpPattern,
        GenericOpPattern<ReturnOp>,
        GenericOpPattern<GeoProd>,
        GenericOpPattern<Exp>,
        GenericOpPattern<Sandwich>

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