#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>
#include "llvm/IR/Module.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm-c/Target.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/IR/Verifier.h"

namespace py = nanobind;
using namespace llvm;


namespace {
    // taken from triton/python/src/llvm.cc line: 860
    std::unique_ptr<Module> getModuleFromString(std::string &llvmIR, llvm::LLVMContext &context) {
        std::unique_ptr<llvm::MemoryBuffer> buffer =
            llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
        llvm::SMDiagnostic error;
        std::unique_ptr<Module> mod =
            llvm::parseIR(buffer->getMemBufferRef(), error, context);
        if (!mod) {
        llvm::report_fatal_error(
            "failed to parse IR: " + error.getMessage() +
            "lineno: " + std::to_string(error.getLineNo()));
        }
        return mod;
    }
    
}

void init_llvm_ir(py::module_ &m) {

    py::class_<Module>(m, "Module")
    .def("dump", &Module::dump)
    .def("__str__", [](Module *self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        os << *self;
        return os.str();
    }, py::rv_policy::take_ownership);
    py::class_<LLVMContext>(m, "context")
    .def(py::new_([]() {return new LLVMContext();}));

    m.def("translate_mod", [](mlir::ModuleOp &mod, LLVMContext &context) {
        
        std::unique_ptr<Module> llvmMod = mlir::translateModuleToLLVMIR(mod, context);
        if (!llvmMod)
            throw std::runtime_error("failed to translate module to llvmir:\n");
        return llvmMod;
        }, py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>());
    m.def("optimize_mod", [](Module &mod) {

        LoopAnalysisManager LAM;
        FunctionAnalysisManager FAM;
        CGSCCAnalysisManager CGAM;
        ModuleAnalysisManager MAM;
        PassBuilder PB;

        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);

        MPM.run(mod, MAM);
    });
    m.def("init_targets", []() {
        static std::once_flag init_flag;
        std::call_once(init_flag, []() {
            LLVMInitializeNVPTXTargetInfo();
            LLVMInitializeNVPTXTarget();
            LLVMInitializeNVPTXTargetMC();
            LLVMInitializeNVPTXAsmPrinter();
        });
    });
    
    m.def("llvm_to_ptx", [](std::string &llvmIR, std::string cpu, std::string features) {
        Triple triple = Triple("nvptx64-nvidia-cuda");
        std::string err;
        llvm::LLVMContext context;
        std::unique_ptr<llvm::Module> mod = getModuleFromString(llvmIR, context);

        const Target *target = TargetRegistry::lookupTarget(triple, err);
        if (!target) throw std::runtime_error("lookupTarget: " + err);

        TargetOptions opts;
        std::unique_ptr<TargetMachine> tm(target->createTargetMachine(
            triple, cpu, features, opts, Reloc::PIC_, std::nullopt, CodeGenOptLevel::Aggressive
        ));

        mod->setTargetTriple(triple);
        mod->setDataLayout(tm->createDataLayout());

        SmallString<0> buf;
        raw_svector_ostream os(buf);

        legacy::PassManager pm;
        if (tm->addPassesToEmitFile(pm, os, nullptr, CodeGenFileType::AssemblyFile))
            throw std::runtime_error("NVPTX can't emit PTX code");
        
        if (!mod)
            throw std::runtime_error("Module cannot be runned");


        if (llvm::verifyModule(*mod, &llvm::errs()))
            throw std::runtime_error("invalid LLVM module");

        pm.run(*mod);
        llvm::errs() << "Here!\n";

        return std::string(buf.str());
    });
}