#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clifford/Dialect/Clifford/IR/Dialect.h"

#include "clifford/Dialect/CliffGPU/IR/Dialect.h"
#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"

#include "clifford/Conversion/CliffGPUToLLVM/Passes.h"
#include "clifford/Conversion/CliffGPUToLLVM/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"


namespace mlir::clg {
#define GEN_PASS_DEF_CONVERTCLIFFGPUTOLLVM
#include "clifford/Conversion/CliffGPUToLLVM/Passes.h.inc"

} // namespace mlir::clg
#include "clifford/Conversion/CliffGPUToLLVM/PopulatePatterns.h"

using namespace mlir;
using namespace mlir::cliff;


class ConvertCliffGPUToLLVM : public clg::impl::ConvertCliffGPUToLLVMBase<ConvertCliffGPUToLLVM> {
public:
using ConvertCliffGPUToLLVMBase::ConvertCliffGPUToLLVMBase;

    void runOnOperation() override {

        {
            // prologue
            RewritePatternSet patterns(&getContext());
            populateGeometricRewritePatterns(patterns);
            
            if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
                signalPassFailure();
        }
        {

            MLIRContext *context = &getContext();
            CliffGPUToLLVMTypeConverter typeConverter(context);
            CliffGPUToLLVMConversionTarget conversionTarget(*context, typeConverter);
            RewritePatternSet patterns(context);
            populateGeometricToLLVMPatterns(typeConverter, patterns);
            populateLoadStoreOpPatterns(typeConverter, patterns);

            
            if (failed(applyPartialConversion(getOperation(), conversionTarget, std::move(patterns)))) 
            return signalPassFailure();
            
        }

    }

};

