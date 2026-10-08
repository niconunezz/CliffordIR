#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>
#include "llvm/IR/Module.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

#include "llvm-c/Target.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace py = nanobind;
using namespace llvm;

void init_llvm_ir(py::module_ &m) {

    py::class_<Module>(m, "Module")
    .def("dump", &Module::dump);
    py::class_<LLVMContext>(m, "context")
    .def(py::new_([]() {return new LLVMContext();}));

    m.def("translate_mod", [](mlir::ModuleOp &mod, LLVMContext &context) {
        
        std::unique_ptr<Module> llvmMod = mlir::translateModuleToLLVMIR(mod, context);
        if (!llvmMod)
            throw std::runtime_error("failed to translate module to llvmir:\n");
        return llvmMod;
        }, py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>());
    
    m.def("init_targets", []() {
        static std::once_flag init_flag;
        std::call_once(init_flag, []() {
            LLVMInitializeNVPTXTargetInfo();
            LLVMInitializeNVPTXTarget();
            LLVMInitializeNVPTXTargetMC();
            LLVMInitializeNVPTXAsmPrinter();
        });
    });
}