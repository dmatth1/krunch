// Pybind11 glue for the GPU range-coder kernels.
//
// Public Python API surface (matches krunch_ac.cpu_reference where possible):
//
//   krunch_ac_cuda.encode_step(cdf, symbols, output_buf, state) -> None
//       cdf:        (N, V+1) int32 CUDA tensor, contiguous, cdf[i,V] == 65536
//       symbols:    (N,)     int32 CUDA tensor, contiguous
//       output_buf: (cap,)   uint8 CUDA tensor, ZERO-INITIALIZED, contiguous
//       state:      (4,)     uint32 CUDA tensor [low, high, pending, bit_offset]
//
//   krunch_ac_cuda.encode_finalize(output_buf, state) -> None
//
// `state` is mutated in place across calls — pass the same tensor for
// every batch within a chunk, finalize once at the end.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "range_coder.cuh"

// Defined in encode_kernel.cu / decode_kernel.cu — kernel launches via <<<>>> live there.
void launch_encode_step(
    const int32_t* cdf, int cdf_stride,
    const int32_t* symbols, int N,
    uint8_t* output_buf, RangeState* state,
    cudaStream_t stream);
void launch_encode_finalize(
    uint8_t* output_buf, RangeState* state,
    cudaStream_t stream);
void launch_decode_init(
    const uint8_t* input_buf, DecodeState* state,
    cudaStream_t stream);
void launch_decode_step(
    const int32_t* cdf, int V,
    const uint8_t* input_buf,
    DecodeState* state,
    int32_t* out_sym,
    cudaStream_t stream);
void launch_decode_init_batched(
    const uint8_t* input_buf, const int32_t* base_byte_offsets,
    DecodeState* states, int B, cudaStream_t stream);
void launch_decode_step_batched(
    const int32_t* cdfs, int V,
    const uint8_t* input_buf, const int32_t* base_byte_offsets,
    DecodeState* states, int32_t* out_syms,
    int B, cudaStream_t stream);
void launch_encode_step_batched(
    const int32_t* cdfs, int V,
    const int32_t* symbols,
    uint8_t* output_buf, const int32_t* base_byte_offsets,
    RangeState* states, int B, cudaStream_t stream);
void launch_encode_finalize_batched(
    uint8_t* output_buf, const int32_t* base_byte_offsets,
    RangeState* states, int B, cudaStream_t stream);

// mb_gemv.cu — multi-block GEMV experiment kernels.
extern "C" {
void launch_mb_gemv_768x768(const void*, const void*, void*, cudaStream_t);
void launch_mb_gemv_768x3072(const void*, const void*, void*, cudaStream_t);
void launch_mb_gemv_3072x768(const void*, const void*, void*, cudaStream_t);
}

// layer_cpp.cpp — C++ orchestration of one RWKV-4 layer at B=1, T=1.
void register_layer_cpp(pybind11::module& m);

// rwkv_step.cu — fused single-step RWKV-4 layer kernel.
// Uses void* for __half to avoid pulling cuda_fp16.h into main.cpp; the
// .cu side reinterpret_casts to __half*.
void launch_rwkv4_layer_step(
    const void* x_in, void* x_out,
    void* att_xx, float* aa, float* bb, float* pp, void* ffn_xx,
    const void* ln1_w, const void* ln1_b,
    const void* tm_k, const void* tm_v, const void* tm_r,
    const float* time_decay, const float* time_first,
    const void* Kw, const void* Vw, const void* Rw, const void* Ow,
    const void* ln2_w, const void* ln2_b,
    const void* ffn_tm_k, const void* ffn_tm_r,
    const void* ffn_Kw, const void* ffn_Vw, const void* ffn_Rw,
    cudaStream_t stream);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

void encode_step(
    at::Tensor cdf,
    at::Tensor symbols,
    at::Tensor output_buf,
    at::Tensor state)
{
    CHECK_CUDA(cdf); CHECK_CONTIG(cdf);
    CHECK_CUDA(symbols); CHECK_CONTIG(symbols);
    CHECK_CUDA(output_buf); CHECK_CONTIG(output_buf);
    CHECK_CUDA(state); CHECK_CONTIG(state);

    TORCH_CHECK(cdf.dtype() == at::kInt, "cdf must be int32 (value 65536 = T doesn't fit uint16)");
    TORCH_CHECK(symbols.dtype() == at::kInt, "symbols must be int32");
    TORCH_CHECK(output_buf.dtype() == at::kByte, "output_buf must be uint8");
    TORCH_CHECK(state.dtype() == at::kUInt32, "state must be uint32");
    TORCH_CHECK(state.numel() == 4, "state must have 4 elements (low,high,pending,bit_offset)");
    TORCH_CHECK(cdf.dim() == 2, "cdf must be 2D");

    const int N = (int)cdf.size(0);
    const int cdf_stride = (int)cdf.stride(0);
    TORCH_CHECK((int)symbols.size(0) == N, "symbols length must match cdf rows");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_encode_step(
        cdf.data_ptr<int32_t>(), cdf_stride,
        symbols.data_ptr<int32_t>(), N,
        output_buf.data_ptr<uint8_t>(),
        reinterpret_cast<RangeState*>(state.data_ptr<uint32_t>()),
        stream);
}

void encode_finalize(at::Tensor output_buf, at::Tensor state)
{
    CHECK_CUDA(output_buf); CHECK_CONTIG(output_buf);
    CHECK_CUDA(state); CHECK_CONTIG(state);
    TORCH_CHECK(state.numel() == 4, "state must have 4 elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_encode_finalize(
        output_buf.data_ptr<uint8_t>(),
        reinterpret_cast<RangeState*>(state.data_ptr<uint32_t>()),
        stream);
}

void decode_init(at::Tensor input_buf, at::Tensor state)
{
    CHECK_CUDA(input_buf); CHECK_CONTIG(input_buf);
    CHECK_CUDA(state); CHECK_CONTIG(state);
    TORCH_CHECK(input_buf.dtype() == at::kByte, "input_buf must be uint8");
    TORCH_CHECK(state.dtype() == at::kUInt32, "state must be uint32");
    TORCH_CHECK(state.numel() == 4, "state must have 4 elements (low,high,value,bit_offset)");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_decode_init(
        input_buf.data_ptr<uint8_t>(),
        reinterpret_cast<DecodeState*>(state.data_ptr<uint32_t>()),
        stream);
}

void decode_step(at::Tensor cdf, at::Tensor input_buf,
                 at::Tensor state, at::Tensor out_sym)
{
    CHECK_CUDA(cdf); CHECK_CONTIG(cdf);
    CHECK_CUDA(input_buf); CHECK_CONTIG(input_buf);
    CHECK_CUDA(state); CHECK_CONTIG(state);
    CHECK_CUDA(out_sym); CHECK_CONTIG(out_sym);

    TORCH_CHECK(cdf.dtype() == at::kInt, "cdf must be int32");
    TORCH_CHECK(input_buf.dtype() == at::kByte, "input_buf must be uint8");
    TORCH_CHECK(state.dtype() == at::kUInt32, "state must be uint32");
    TORCH_CHECK(state.numel() == 4, "state must have 4 elements");
    TORCH_CHECK(out_sym.dtype() == at::kInt, "out_sym must be int32");
    TORCH_CHECK(out_sym.numel() == 1, "out_sym must be a single int");
    TORCH_CHECK(cdf.dim() == 1, "cdf must be 1D (single row, V+1 entries)");

    const int V = (int)cdf.size(0) - 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_decode_step(
        cdf.data_ptr<int32_t>(), V,
        input_buf.data_ptr<uint8_t>(),
        reinterpret_cast<DecodeState*>(state.data_ptr<uint32_t>()),
        out_sym.data_ptr<int32_t>(),
        stream);
}

void decode_init_batched(at::Tensor input_buf, at::Tensor base_byte_offsets,
                         at::Tensor states)
{
    CHECK_CUDA(input_buf); CHECK_CONTIG(input_buf);
    CHECK_CUDA(base_byte_offsets); CHECK_CONTIG(base_byte_offsets);
    CHECK_CUDA(states); CHECK_CONTIG(states);
    TORCH_CHECK(input_buf.dtype() == at::kByte, "input_buf must be uint8");
    TORCH_CHECK(base_byte_offsets.dtype() == at::kInt, "base_byte_offsets must be int32");
    TORCH_CHECK(states.dtype() == at::kUInt32, "states must be uint32");
    TORCH_CHECK(states.numel() % 4 == 0, "states must be 4 * B uint32s");

    const int B = (int)base_byte_offsets.numel();
    TORCH_CHECK((int)(states.numel() / 4) == B, "states size must match B");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_decode_init_batched(
        input_buf.data_ptr<uint8_t>(),
        base_byte_offsets.data_ptr<int32_t>(),
        reinterpret_cast<DecodeState*>(states.data_ptr<uint32_t>()),
        B, stream);
}

void decode_step_batched(at::Tensor cdfs, at::Tensor input_buf,
                         at::Tensor base_byte_offsets,
                         at::Tensor states, at::Tensor out_syms)
{
    CHECK_CUDA(cdfs); CHECK_CONTIG(cdfs);
    CHECK_CUDA(input_buf); CHECK_CONTIG(input_buf);
    CHECK_CUDA(base_byte_offsets); CHECK_CONTIG(base_byte_offsets);
    CHECK_CUDA(states); CHECK_CONTIG(states);
    CHECK_CUDA(out_syms); CHECK_CONTIG(out_syms);

    TORCH_CHECK(cdfs.dtype() == at::kInt, "cdfs must be int32");
    TORCH_CHECK(cdfs.dim() == 2, "cdfs must be [B, V+1]");
    TORCH_CHECK(input_buf.dtype() == at::kByte, "input_buf must be uint8");
    TORCH_CHECK(base_byte_offsets.dtype() == at::kInt, "base_byte_offsets must be int32");
    TORCH_CHECK(states.dtype() == at::kUInt32, "states must be uint32");
    TORCH_CHECK(out_syms.dtype() == at::kInt, "out_syms must be int32");

    const int B = (int)cdfs.size(0);
    const int V = (int)cdfs.size(1) - 1;
    TORCH_CHECK((int)base_byte_offsets.numel() == B, "base_byte_offsets size mismatch");
    TORCH_CHECK((int)(states.numel() / 4) == B, "states size mismatch");
    TORCH_CHECK((int)out_syms.numel() == B, "out_syms size mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_decode_step_batched(
        cdfs.data_ptr<int32_t>(), V,
        input_buf.data_ptr<uint8_t>(),
        base_byte_offsets.data_ptr<int32_t>(),
        reinterpret_cast<DecodeState*>(states.data_ptr<uint32_t>()),
        out_syms.data_ptr<int32_t>(),
        B, stream);
}

void encode_step_batched(at::Tensor cdfs, at::Tensor symbols, at::Tensor output_buf,
                         at::Tensor base_byte_offsets, at::Tensor states)
{
    CHECK_CUDA(cdfs); CHECK_CONTIG(cdfs);
    CHECK_CUDA(symbols); CHECK_CONTIG(symbols);
    CHECK_CUDA(output_buf); CHECK_CONTIG(output_buf);
    CHECK_CUDA(base_byte_offsets); CHECK_CONTIG(base_byte_offsets);
    CHECK_CUDA(states); CHECK_CONTIG(states);

    TORCH_CHECK(cdfs.dtype() == at::kInt, "cdfs must be int32");
    TORCH_CHECK(cdfs.dim() == 2, "cdfs must be [B, V+1]");
    TORCH_CHECK(symbols.dtype() == at::kInt, "symbols must be int32");
    TORCH_CHECK(output_buf.dtype() == at::kByte, "output_buf must be uint8");
    TORCH_CHECK(base_byte_offsets.dtype() == at::kInt, "base_byte_offsets must be int32");
    TORCH_CHECK(states.dtype() == at::kUInt32, "states must be uint32");

    const int B = (int)cdfs.size(0);
    const int V = (int)cdfs.size(1) - 1;
    TORCH_CHECK((int)symbols.numel() == B, "symbols size mismatch");
    TORCH_CHECK((int)base_byte_offsets.numel() == B, "base_byte_offsets size mismatch");
    TORCH_CHECK((int)(states.numel() / 4) == B, "states size mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_encode_step_batched(
        cdfs.data_ptr<int32_t>(), V,
        symbols.data_ptr<int32_t>(),
        output_buf.data_ptr<uint8_t>(),
        base_byte_offsets.data_ptr<int32_t>(),
        reinterpret_cast<RangeState*>(states.data_ptr<uint32_t>()),
        B, stream);
}

void encode_finalize_batched(at::Tensor output_buf, at::Tensor base_byte_offsets,
                             at::Tensor states)
{
    CHECK_CUDA(output_buf); CHECK_CONTIG(output_buf);
    CHECK_CUDA(base_byte_offsets); CHECK_CONTIG(base_byte_offsets);
    CHECK_CUDA(states); CHECK_CONTIG(states);
    TORCH_CHECK(states.dtype() == at::kUInt32, "states must be uint32");

    const int B = (int)base_byte_offsets.numel();
    TORCH_CHECK((int)(states.numel() / 4) == B, "states size mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_encode_finalize_batched(
        output_buf.data_ptr<uint8_t>(),
        base_byte_offsets.data_ptr<int32_t>(),
        reinterpret_cast<RangeState*>(states.data_ptr<uint32_t>()),
        B, stream);
}

// rwkv_step binding: takes torch tensors, dispatches to launch_rwkv4_layer_step.
// All tensors must be CUDA + contiguous; weights fp16 except {aa,bb,pp,
// time_decay, time_first} which are fp32.
void rwkv4_layer_step(
    at::Tensor x_in, at::Tensor x_out,
    at::Tensor att_xx, at::Tensor aa, at::Tensor bb, at::Tensor pp,
    at::Tensor ffn_xx,
    at::Tensor ln1_w, at::Tensor ln1_b,
    at::Tensor tm_k, at::Tensor tm_v, at::Tensor tm_r,
    at::Tensor time_decay, at::Tensor time_first,
    at::Tensor Kw, at::Tensor Vw, at::Tensor Rw, at::Tensor Ow,
    at::Tensor ln2_w, at::Tensor ln2_b,
    at::Tensor ffn_tm_k, at::Tensor ffn_tm_r,
    at::Tensor ffn_Kw, at::Tensor ffn_Vw, at::Tensor ffn_Rw)
{
    auto check_fp16 = [](at::Tensor t, const char* name) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be CUDA contiguous");
        TORCH_CHECK(t.dtype() == at::kHalf, name, " must be fp16");
    };
    auto check_fp32 = [](at::Tensor t, const char* name) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be CUDA contiguous");
        TORCH_CHECK(t.dtype() == at::kFloat, name, " must be fp32");
    };
    check_fp16(x_in, "x_in"); check_fp16(x_out, "x_out");
    check_fp16(att_xx, "att_xx"); check_fp16(ffn_xx, "ffn_xx");
    check_fp32(aa, "aa"); check_fp32(bb, "bb"); check_fp32(pp, "pp");
    check_fp32(time_decay, "time_decay"); check_fp32(time_first, "time_first");
    check_fp16(ln1_w, "ln1_w"); check_fp16(ln1_b, "ln1_b");
    check_fp16(tm_k, "tm_k"); check_fp16(tm_v, "tm_v"); check_fp16(tm_r, "tm_r");
    check_fp16(Kw, "Kw"); check_fp16(Vw, "Vw"); check_fp16(Rw, "Rw"); check_fp16(Ow, "Ow");
    check_fp16(ln2_w, "ln2_w"); check_fp16(ln2_b, "ln2_b");
    check_fp16(ffn_tm_k, "ffn_tm_k"); check_fp16(ffn_tm_r, "ffn_tm_r");
    check_fp16(ffn_Kw, "ffn_Kw"); check_fp16(ffn_Vw, "ffn_Vw"); check_fp16(ffn_Rw, "ffn_Rw");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_rwkv4_layer_step(
        x_in.data_ptr(), x_out.data_ptr(),
        att_xx.data_ptr(), aa.data_ptr<float>(), bb.data_ptr<float>(),
        pp.data_ptr<float>(), ffn_xx.data_ptr(),
        ln1_w.data_ptr(), ln1_b.data_ptr(),
        tm_k.data_ptr(), tm_v.data_ptr(), tm_r.data_ptr(),
        time_decay.data_ptr<float>(), time_first.data_ptr<float>(),
        Kw.data_ptr(), Vw.data_ptr(), Rw.data_ptr(), Ow.data_ptr(),
        ln2_w.data_ptr(), ln2_b.data_ptr(),
        ffn_tm_k.data_ptr(), ffn_tm_r.data_ptr(),
        ffn_Kw.data_ptr(), ffn_Vw.data_ptr(), ffn_Rw.data_ptr(),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_step", &encode_step,
          "Range-encode a batch of symbols on GPU; mutates state in place.");
    m.def("encode_finalize", &encode_finalize,
          "Final-flush the range coder; mutates state.bit_offset.");
    m.def("decode_init", &decode_init,
          "Initialize decoder state by reading first 32 bits.");
    m.def("decode_step", &decode_step,
          "Decode one symbol given current CDF; mutates state, writes out_sym.");
    m.def("decode_init_batched", &decode_init_batched,
          "Init B decoder states; one launch over B streams.");
    m.def("decode_step_batched", &decode_step_batched,
          "Decode one symbol per stream for B streams; one launch.");
    m.def("encode_step_batched", &encode_step_batched,
          "Encode one symbol per stream for B streams; one launch.");
    m.def("encode_finalize_batched", &encode_finalize_batched,
          "Final-flush B encoder states; one launch.");
    m.def("rwkv4_layer_step", &rwkv4_layer_step,
          "Fused RWKV-4 single-layer forward (B=1, T=1).");
    m.def("mb_gemv_768x768", [](at::Tensor x, at::Tensor W, at::Tensor y) {
        cudaStream_t s = at::cuda::getCurrentCUDAStream();
        launch_mb_gemv_768x768(x.data_ptr(), W.data_ptr(), y.data_ptr(), s);
    }, "Multi-block GEMV: y[768] = x[768] @ W[768,768]");
    m.def("mb_gemv_768x3072", [](at::Tensor x, at::Tensor W, at::Tensor y) {
        cudaStream_t s = at::cuda::getCurrentCUDAStream();
        launch_mb_gemv_768x3072(x.data_ptr(), W.data_ptr(), y.data_ptr(), s);
    }, "Multi-block GEMV: y[3072] = x[768] @ W[768,3072]");
    m.def("mb_gemv_3072x768", [](at::Tensor x, at::Tensor W, at::Tensor y) {
        cudaStream_t s = at::cuda::getCurrentCUDAStream();
        launch_mb_gemv_3072x768(x.data_ptr(), W.data_ptr(), y.data_ptr(), s);
    }, "Multi-block GEMV: y[768] = x[3072] @ W[3072,768]");

    register_layer_cpp(m);
}
