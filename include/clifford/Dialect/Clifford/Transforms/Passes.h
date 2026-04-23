#ifndef CLIFFORD_DIALECT_CLIFFORD_TRANSFORMS_PASSES_H
#define CLIFFORD_DIALECT_CLIFFORD_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "clifford/Dialect/Clifford/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cliff {
    
    #define GEN_PASS_DECL
    #include "clifford/Dialect/Clifford/Transforms/CliffPasses.h.inc"
    
    #define GEN_PASS_REGISTRATION
    #include "clifford/Dialect/Clifford/Transforms/CliffPasses.h.inc"

} // end namespace mlir::cliff

namespace mlir {
class CliffTypeConverter : public TypeConverter {

    public:
        CliffTypeConverter(MLIRContext *context);
        
    private:
        MLIRContext *context;

};


class CliffConversionTarget : public ConversionTarget {
public:

    explicit CliffConversionTarget(MLIRContext &context, TypeConverter &typeConverter);

    bool isDynamicallyLegal(Operation *op, TypeConverter &typeConverter);
};

} // end namespace mlir


#endif //CLIFFORD_DIALECT_CLIFFORD_TRANSFORMS_PASSES_H