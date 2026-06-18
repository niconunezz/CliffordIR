#include "clifford/Dialect/Clifford/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"


#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"

namespace mlir::cliff {
    #define GEN_PASS_DEF_GEOMETRICTYPECONVERSIONPASS
    #include "clifford/Dialect/Clifford/Transforms/CliffPasses.h.inc"

} // end namespace mlir::cliff

using namespace mlir;
using namespace mlir::cliff;


CliffTypeConverter::CliffTypeConverter(MLIRContext *ctx) : context(ctx) {

    addConversion([](Type type) { return type; });

    addConversion([&](RankedTensorType tensor) {
        if (auto point = dyn_cast<Cliff_PointType>(tensor.getElementType())) 
            return RankedTensorType::get(tensor.getShape(), convertType(point));
        if (auto line = dyn_cast<Cliff_LineType>(tensor.getElementType())) 
            return RankedTensorType::get(tensor.getShape(), convertType(line));
        if (auto motor = dyn_cast<Cliff_MotorType>(tensor.getElementType())) 
            return RankedTensorType::get(tensor.getShape(), convertType(motor));
        if (auto scalar = dyn_cast<Cliff_ScalarType>(tensor.getElementType())) 
            return RankedTensorType::get(tensor.getShape(), convertType(scalar));
        
        return tensor;
    });

    addConversion([&](Cliff_LineType obj) {
        return obj.asMultivector();
    });
    
    addConversion([&](Cliff_PointType obj) {
        return obj.asMultivector();
    });
    
    addConversion([&](Cliff_MotorType obj) {
        return obj.asMultivector();
    });

    addConversion([&](Cliff_ScalarType obj) {
       return obj.asMultivector();
    });
    
}
CliffConversionTarget::CliffConversionTarget(MLIRContext &context, TypeConverter &typeConverter) : ConversionTarget(context) {
    
    addDynamicallyLegalOp<FuncOp>([&](FuncOp op){
        for (auto arg : op.getArguments()) {
            if (!typeConverter.isLegal(arg))
                return false;
        }
        for (auto res : op.getResultTypes()) {
            if (!typeConverter.isLegal(res))
                return false;
        }

        return true;
    });

    addDynamicallyLegalDialect<cliff::CliffDialect>([&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });
    
};

bool CliffConversionTarget::isDynamicallyLegal(Operation *op, TypeConverter &typeConverter) {
    return typeConverter.isLegal(op->getOperandTypes());
};

namespace {    

static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}


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



// note this is reused code from cliffToCliffGPU, could be generalized
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
                return rewriter.notifyMatchFailure(op, "Issue converting argument types");
            sigConversion.addInputs(i, converted);
        }

        if (failed(converter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return rewriter.notifyMatchFailure(op, "Issue converting function return types");;

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
            return rewriter.notifyMatchFailure(op, "Error occurred while converting region types");

        rewriter.replaceOp(op, newFunc);
        return success();
    }
};


void populateGeometricTypeConversionPattern(TypeConverter &typeConverter, RewritePatternSet &patterns) {
    MLIRContext* context = patterns.getContext();
    patterns.insert<CliffFuncOpPattern,
                    GenericOpPattern<ReturnOp>,
                    GenericOpPattern<GeoProd>,
                    GenericOpPattern<Sandwich>,
                    GenericOpPattern<Rotate>,
                    GenericOpPattern<Translate>,
                    GenericOpPattern<Reverse>,
                    GenericOpPattern<Exp>
                    >(typeConverter, context);
}


class GeometricTypeConversionPass : public cliff::impl::GeometricTypeConversionPassBase<GeometricTypeConversionPass> {
public:
    using GeometricTypeConversionPassBase::GeometricTypeConversionPassBase;
        
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        CliffTypeConverter typeConverter(context);
        RewritePatternSet patterns(context);
            
        CliffConversionTarget conversionTarget(*context, typeConverter);
        populateGeometricTypeConversionPattern(typeConverter, patterns);

        if (failed(applyPartialConversion(getOperation(), conversionTarget, std::move(patterns)))) 
            return signalPassFailure();
            
    }

};
} // end namespace