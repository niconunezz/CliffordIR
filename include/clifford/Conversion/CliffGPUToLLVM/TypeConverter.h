#ifndef CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_TYPECONVERTER_H
#define CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_TYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class CliffGPUToLLVMTypeConverter : public TypeConverter {

public:
    CliffGPUToLLVMTypeConverter(MLIRContext *context);
    
private:
    MLIRContext *context;

};


class CliffGPUToLLVMConversionTarget : public ConversionTarget {

public:
    explicit CliffGPUToLLVMConversionTarget(MLIRContext &ctx, const TypeConverter &typeConverter);
};

} // namespace mlir

#endif // CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_TYPECONVERTER_H