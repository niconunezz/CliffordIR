#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"

namespace mlir::clg {
    
    SmallVector<Value> unpackElements(Location loc, Value llvmStruct, RewriterBase &rewriter) {
        LLVM::LLVMStructType types = cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
        
        auto b = CliffordLLVMOpBuilder(rewriter, loc);
        SmallVector<Value> ret;
        for (unsigned i = 0; i < types.size(); ++i) {
            Type type = types[i];
            ret.push_back(b.extract(type, llvmStruct, i));
        }
        return result;
    }

    Value packElements(Location loc, ValueRange resultVals,
                       const LLVMTypeConverter *typeConverter,
                       RewriterBase &rewriter, Type type) {
        
        auto structType = dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
        auto b = CliffordLLVMOpBuilder(rewriter, loc);
        
        Value result = LLVM::UndefOp::create(rewriter, loc, structType);
        auto structTypes = structType.getBody();
        for (auto [i, value] : llvm::enumerate(resultVals)) {
            Type currType = structTypes[i];
            b.insert(structType, result, resultVals[i], i);
        }

        return result;
    }
}