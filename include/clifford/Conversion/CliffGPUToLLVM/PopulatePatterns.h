#include "clifford/Conversion/CliffGPUToLLVM/TypeConverter.h"
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"

namespace mlir {
namespace cliff {

void populateLoadStoreOpPatterns(CliffGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns);
void populateGeometricToLLVMPatterns(CliffGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns);
void populateGeometricRewritePatterns(RewritePatternSet &patterns);



}

}