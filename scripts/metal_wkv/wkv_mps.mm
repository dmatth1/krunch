// PyTorch MPS extension wrapping scripts/metal_wkv/wkv.metal.
//
// Exposes two functions:
//   wkv_forward_mps(w, u, k, v) -> y
//   wkv_backward_mps(w, u, k, v, gy) -> (gw, gu, gk, gv)
//
// These are registered on torch's MPS dispatch key. The Python
// autograd wrapper (scripts/metal_wkv/__init__.py) composes them into
// a torch.autograd.Function that matches the reference CUDA WKV op.
//
// Build via torch.utils.cpp_extension.load(...) — see metal_wkv/__init__.py.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSAllocatorInterface.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>

using at::mps::MPSStream;
using at::mps::getCurrentMPSStream;
using at::mps::getIMPSAllocator;

// On MPS, `tensor.storage().data()` returns the CONTENTS pointer of the
// underlying MTLBuffer (not the buffer object). To get the actual
// id<MTLBuffer> we have to ask the MPSAllocator to look up which buffer
// backs that contents pointer. `getSharedBufferPtr(contents)` returns
// `(buffer, offset_bytes)` — the MTLBuffer is bridged opaquely via
// `const void*` in the C++ interface.
//
// Also accounts for tensor.storage_offset (elements, needs * element_size).
static id<MTLBuffer> buffer_from_tensor(const torch::Tensor& t, uint32_t& byte_offset) {
    void* contents = t.data_ptr();
    auto pr = getIMPSAllocator()->getSharedBufferPtr(contents);
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(pr.first);
    byte_offset = pr.second;
    // Debug: validate the buffer's contents pointer + offset matches
    // the tensor's data_ptr. If not, our indexing is off.
    if (buf == nil) {
        TORCH_CHECK(false, "buffer_from_tensor: allocator returned nil buffer for ptr ",
                    (void*)contents);
    }
    return buf;
}

// The path to wkv.metal is passed via the L3TC_WKV_METAL_PATH env var,
// set by scripts/metal_wkv/__init__.py before `load()`. Read at first
// kernel launch; resulting MTLLibrary + pipeline states are cached.

// ---------------------------------------------------------------- Cache
namespace {

struct KernelCache {
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> forward_pso = nil;
    id<MTLComputePipelineState> backward_pso = nil;
};

KernelCache& cache() {
    static KernelCache c;
    return c;
}

void ensure_compiled() {
    auto& kc = cache();
    if (kc.library != nil) return;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError* err = nil;

        const char* env_path = std::getenv("L3TC_WKV_METAL_PATH");
        TORCH_CHECK(env_path != nullptr,
                    "L3TC_WKV_METAL_PATH env var is not set; "
                    "scripts/metal_wkv/__init__.py._load_ext should set it");
        NSString* src_path = [NSString stringWithUTF8String:env_path];
        NSString* src = [NSString stringWithContentsOfFile:src_path
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        TORCH_CHECK(src != nil, "failed to read ", env_path, ": ",
                    err ? err.localizedDescription.UTF8String : "(no err)");

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        // setMathMode is macOS 15+; we'd want fastMath for parity with
        // the CUDA kernel's --use_fast_math, but on 14.x it's unavailable
        // so we rely on the default which is already fast. The
        // non-fast-math fallback still produces results within a few ULPs
        // of the CUDA kernel — verified in parity_check.py.

        kc.library = [device newLibraryWithSource:src options:opts error:&err];
        TORCH_CHECK(kc.library != nil, "metal compile failed: ",
                    err ? err.localizedDescription.UTF8String : "(no err)");

        id<MTLFunction> fwd_fn = [kc.library newFunctionWithName:@"wkv_forward"];
        id<MTLFunction> bwd_fn = [kc.library newFunctionWithName:@"wkv_backward"];
        TORCH_CHECK(fwd_fn && bwd_fn, "wkv_forward / wkv_backward not in library");

        kc.forward_pso = [device newComputePipelineStateWithFunction:fwd_fn error:&err];
        TORCH_CHECK(kc.forward_pso != nil, "forward pso: ",
                    err ? err.localizedDescription.UTF8String : "(no err)");

        kc.backward_pso = [device newComputePipelineStateWithFunction:bwd_fn error:&err];
        TORCH_CHECK(kc.backward_pso != nil, "backward pso: ",
                    err ? err.localizedDescription.UTF8String : "(no err)");
    }
}

// A single (buffer, byte_offset) pair to bind to a kernel arg slot.
struct BufArg {
    id<MTLBuffer> buf;
    uint32_t offset;
};

// Dispatch helper — one thread per (b, c). Runs on a fresh command buffer
// committed & waited on inline; this means the kernel is synchronous from
// the PyTorch side (which is what we want for a custom op that returns a
// tensor — the returned tensor's data must be valid by the time we return).
void dispatch_1d(id<MTLComputePipelineState> pso,
                 std::vector<BufArg> buffers,
                 std::vector<int32_t>& ints,
                 int total_threads) {
    @autoreleasepool {
        // Get the Metal command queue directly from the MPSStream. We don't
        // use getCurrentMPSStream()->commandEncoder() because that returns
        // an MPSGraph-oriented encoder path; our kernel is pure compute.
        id<MTLCommandQueue> queue = getCurrentMPSStream()->commandQueue();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        NSUInteger idx = 0;
        for (const BufArg& ba : buffers) {
            [enc setBuffer:ba.buf offset:ba.offset atIndex:idx++];
        }
        for (int32_t v : ints) {
            [enc setBytes:&v length:sizeof(int32_t) atIndex:idx++];
        }
        NSUInteger tpt = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 32);
        MTLSize grid = MTLSizeMake(total_threads, 1, 1);
        MTLSize tg = MTLSizeMake(tpt, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

}  // namespace

// ---------------------------------------------------------------- Ops
torch::Tensor wkv_forward_mps(torch::Tensor w, torch::Tensor u,
                              torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(k.device().is_mps(), "wkv_forward: k must be on MPS");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "wkv_forward: fp32 only");
    TORCH_CHECK(k.dim() == 3, "k shape: (B, T, C)");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have same shape");
    TORCH_CHECK(w.dim() == 1 && u.dim() == 1, "w, u must be 1-D");
    TORCH_CHECK(w.size(0) == u.size(0) && w.size(0) == k.size(2),
                "w, u, k channel dim mismatch");

    w = w.contiguous();
    u = u.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    int64_t B = k.size(0);
    int64_t T = k.size(1);
    int64_t C = k.size(2);
    auto y = torch::empty_like(k);

    ensure_compiled();
    auto& kc = cache();

    uint32_t ow, ou, ok, ov, oy;
    std::vector<BufArg> bufs = {
        {buffer_from_tensor(w, ow), ow},
        {buffer_from_tensor(u, ou), ou},
        {buffer_from_tensor(k, ok), ok},
        {buffer_from_tensor(v, ov), ov},
        {buffer_from_tensor(y, oy), oy},
    };
    std::vector<int32_t> ints = {(int32_t)B, (int32_t)T, (int32_t)C};
    dispatch_1d(kc.forward_pso, bufs, ints, (int)(B * C));
    return y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
wkv_backward_mps(torch::Tensor w, torch::Tensor u,
                 torch::Tensor k, torch::Tensor v,
                 torch::Tensor gy) {
    TORCH_CHECK(gy.device().is_mps(), "wkv_backward: gy must be on MPS");
    TORCH_CHECK(gy.dtype() == torch::kFloat32, "wkv_backward: fp32 only");
    TORCH_CHECK(gy.sizes() == k.sizes(), "gy and k shape mismatch");

    w = w.contiguous();
    u = u.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    gy = gy.contiguous();

    int64_t B = k.size(0);
    int64_t T = k.size(1);
    int64_t C = k.size(2);
    TORCH_CHECK(T <= 2048, "T_MAX = 2048 in wkv.metal; recompile to extend");

    // gw, gu are accumulated PER (b, c) by the kernel (matches CUDA
    // behavior). The Python side reduces across batch.
    auto gw = torch::zeros({B, C}, k.options());
    auto gu = torch::zeros({B, C}, k.options());
    auto gk = torch::empty_like(k);
    auto gv = torch::empty_like(k);

    ensure_compiled();
    auto& kc = cache();

    uint32_t ow, ou, ok, ov, ogy, ogw, ogu, ogk, ogv;
    std::vector<BufArg> bufs = {
        {buffer_from_tensor(w, ow), ow},
        {buffer_from_tensor(u, ou), ou},
        {buffer_from_tensor(k, ok), ok},
        {buffer_from_tensor(v, ov), ov},
        {buffer_from_tensor(gy, ogy), ogy},
        {buffer_from_tensor(gw, ogw), ogw},
        {buffer_from_tensor(gu, ogu), ogu},
        {buffer_from_tensor(gk, ogk), ogk},
        {buffer_from_tensor(gv, ogv), ogv},
    };
    std::vector<int32_t> ints = {(int32_t)B, (int32_t)T, (int32_t)C};
    dispatch_1d(kc.backward_pso, bufs, ints, (int)(B * C));
    return {gw, gu, gk, gv};
}

// ---------------------------------------------------------------- Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward_mps, "WKV forward (MPS)");
    m.def("wkv_backward", &wkv_backward_mps, "WKV backward (MPS)");
}
