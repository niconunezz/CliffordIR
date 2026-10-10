#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "cuda.h"
#include "llvm/Support/raw_ostream.h"
namespace py = nanobind;


#define CU_CHECK(call)                                                        \
  do {                                                                        \
    CUresult _r = (call);                                                     \
    if (_r != CUDA_SUCCESS) {                                                 \
      const char *n = "?", *s = "?";                                          \
      cuGetErrorName(_r, &n);                                                 \
      cuGetErrorString(_r, &s);                                               \
      throw std::runtime_error(std::string(#call) + " -> " + n + ": " + s);   \
    }                                                                         \
  } while (0)


static void ensure_context() {
    CU_CHECK(cuInit(0));
    CUcontext ctx;
    CU_CHECK(cuCtxGetCurrent(&ctx));
    if (!ctx) {
        CUdevice dev;
        CU_CHECK(cuDeviceGet(&dev, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
        CU_CHECK(cuCtxSetCurrent(ctx));
    }
}

struct Kernel { CUmodule module; CUfunction fn; };

static Kernel getKernel(const std::string &ptx, const std::string &name) {
    ensure_context();

    char errLog[8192] = {0};
    CUjit_option opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *vals[] = {errLog, (void *)(uintptr_t)sizeof(errLog)};

    Kernel k{};
    CUresult r = cuModuleLoadDataEx(&k.module, ptx.c_str(), 2, opts, vals);

    if (r != CUDA_SUCCESS)
        throw std::runtime_error(std::string("cuModuleLoadDataEx failed:\n") + errLog);
    CU_CHECK(cuModuleGetFunction(&k.fn, k.module, name.c_str()));
    return k;
}

void init_driver(py::module_ &m) {


    union Arg {CUdeviceptr p; int32_t i32; int64_t i64; float f32; double f64; };
    m.def("launch_kernel", [](std::vector<unsigned int> grid,
                              std::vector<unsigned int> blockDim,
                              const std::string &ptx, const std::string &name,
                              unsigned smemBytes, 
                              const std::vector<std::string> &sig, const std::vector<py::object> &args,
                              uintptr_t stream) {
        
        
        Kernel k = getKernel(ptx, name);
        
        std::vector<Arg> store(args.size());
        std::vector<void *> params(args.size());
        
        for (size_t i=0; i < args.size(); ++i) {

            const std::string &t = sig[i];
            if (t[0] == '*') store[i].p = py::cast<uint64_t>(args[i]);
            else if (t == "i32") store[i].i32 = py::cast<int32_t>(args[i]);
            else if (t == "fp32") store[i].f32 = py::cast<float>(args[i]);
            else throw std::runtime_error("unkown type : " + t);
            params[i] = &store[i];
        }        
        py::gil_scoped_release nogil;
        CU_CHECK(cuLaunchKernel(k.fn, grid[0], grid[1], grid[2], 
                       blockDim[0], blockDim[1], blockDim[2], smemBytes,
                       (CUstream)stream, params.data(), nullptr));
    });
}
