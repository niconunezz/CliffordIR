#ifndef CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_UTILITY_H
#define CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_UTILITY_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::clg {

    struct CliffordLLVMOpBuilder {
        CliffordLLVMOpBuilder(Location loc, OpBuilder &builder) : loc(loc), builder(&builder) {}

        template <typename ... Args> LLVM::AddOp add(Args &&...args) {
            return LLVM::AddOp::create(*builder, loc, std::forward<Args>(args)...);
        }

        template <typename ... Args> LLVM::SubOp sub(Args &&...args) {
            return LLVM::SubOp::create(*builder, loc, std::forward<Args>(args)...);
        }

        template <typename ... Args> LLVM::ExtractValueOp extract(Args &&...args) {
            return LLVM::ExtractValueOp::create(*builder, loc, std::forward<Args>(args)...);
        }

        template <typename ... Args> LLVM::InsertValueOp insert(Args &&...args) {
            return LLVM::InsertValueOp::create(*builder, loc, std::forward<Args>(args)...);
        }

        template <typename... Args> LLVM::FNegOp neg(Args &&...args) {
            return LLVM::FNegOp::create(*builder, loc, std::forward<Args>(args)...);
        }

        template <typename... Args> LLVM::FAddOp fadd(Args &&...args) {
            return LLVM::FAddOp::create(*builder, loc, std::forward<Args>(args)...);
        }
        template <typename... Args> LLVM::MulOp mul(Args &&...args) {
            return LLVM::MulOp::create(*builder, loc, std::forward<Args>(args)...);
        }
        template <typename... Args> LLVM::FMulOp fmul(Args &&...args) {
            return LLVM::FMulOp::create(*builder, loc, std::forward<Args>(args)...);
        }
        template <typename... Args> LLVM::FMAOp fma(Args &&...args) {
            return LLVM::FMAOp::create(*builder, loc, std::forward<Args>(args)...);
        }

        Location loc;
        OpBuilder *builder;
    };

    SmallVector<Value> unpackElements(Location loc, Value llvmStruct, RewriterBase &rewriter);
    Value packElements(Location loc, ValueRange resultVals,
                       const TypeConverter *typeConverter,
                       RewriterBase &rewriter, Type type);

}

#endif // CLIFFORD_CONVERSION_CLIFFGPUTOLLVM_UTILITY_H
