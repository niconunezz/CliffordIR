#include "clifford/Conversion/CliffGPUToLLVM/Utility.h"

namespace mlir::clg {
    
    SmallVector<Value> unpackElements(Location loc, Value llvmStruct, RewriterBase &rewriter) { 

        ArrayRef<mlir::Type> types = cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
        
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        SmallVector<Value> ret;
        for (unsigned i = 0; i < types.size(); ++i) {
            Type type = types[i];
            ret.push_back(b.extract(type, llvmStruct, i));
        }
        return ret;
    }

    Value packElements(Location loc, ValueRange resultVals,
                       const TypeConverter *typeConverter,
                       RewriterBase &rewriter, Type type) {
        auto structType = cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
        auto b = CliffordLLVMOpBuilder(loc, rewriter);
        
        Value result = LLVM::UndefOp::create(rewriter, loc, structType);
        auto structTypes = structType.getBody();

        for (auto [i, value] : llvm::enumerate(resultVals)) {
            Type currType = structTypes[i];
            result = b.insert(structType, result, resultVals[i], i);
        }
        return result;
    }
}