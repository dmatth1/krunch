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

// layer_cpp.cpp — C++ orchestration of one RWKV-4 layer at B=1, T=1.
void register_layer_cpp(pybind11::module& m);

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

    register_layer_cpp(m);
}
