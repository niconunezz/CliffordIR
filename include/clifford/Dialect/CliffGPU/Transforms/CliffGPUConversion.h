#ifndef CLIFFORD_DIALECT_CLIFFGPU_TRANSFORMS_CONVERSION_H
#define CLIFFORD_DIALECT_CLIFFGPU_TRANSFORMS_CONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class CliffGPUTypeConverter : public TypeConverter {

public:
    CliffGPUTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp);
    int getNumWarps() const { return numWarps; }
    int getThreadsPerWarp() const { return threadsPerWarp; }

private:
    MLIRContext *context;
    int numWarps;
    int threadsPerWarp;
};


class CliffGPUConversionTarget : public ConversionTarget {

public:
    explicit CliffGPUConversionTarget(MLIRContext &ctx, const TypeConverter &typeConverter);
};

} // namespace mlir

#endif