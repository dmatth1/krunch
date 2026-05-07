// GPU range-coder encode kernel. One thread per stream.
//
// Why single-thread-per-stream: range coding is fundamentally serial
// within a stream — each token's emitted bits depend on the AC state
// (low/high/pending) left by the previous token. We get speedup vs the
// CPU path NOT from intra-stream parallelism but by avoiding the
// 200 MB/batch (1024 × 50K × 4) GPU→CPU prob transfer entirely. The
// CDF lives in GPU memory; the kernel reads cdf[sym] and cdf[sym+1]
// directly.
//
// Multiple chunks per worker → grid_dim.x parallel streams (each
// block has its own state, output buffer, CDF slice). v1.1 ships
// with a single stream per kernel launch (one chunk at a time);
// the multi-stream path is a v1.2 follow-up.

#include "range_coder.cuh"
#include <cuda_runtime.h>

// Bit-write helper: writes `bit` MSB-first into output_buf at
// `bit_offset`, advances bit_offset. Bytes pre-zeroed by caller.
__device__ __forceinline__ void write_bit(
    uint8_t* output_buf,
    uint32_t& bit_offset,
    uint32_t bit)
{
    const uint32_t byte_idx = bit_offset >> 3;
    const uint32_t bit_in_byte = 7u - (bit_offset & 7u);  // MSB-first
    output_buf[byte_idx] |= ((bit & 1u) << bit_in_byte);
    bit_offset++;
}

__device__ __forceinline__ void emit_bit_with_pending(
    uint8_t* output_buf,
    uint32_t& bit_offset,
    uint32_t& pending,
    uint32_t bit)
{
    write_bit(output_buf, bit_offset, bit);
    const uint32_t inv = 1u - bit;
    for (uint32_t i = 0; i < pending; i++) {
        write_bit(output_buf, bit_offset, inv);
    }
    pending = 0;
}

// Encode `N` symbols using `cdf` (shape (N, V+1) uint16, row-major).
// Updates *state in place. cdf row stride is `cdf_stride` uint16
// elements (= V+1 typically).
//
// Single-threaded by construction (grid_dim = 1, block_dim = 1) —
// see file header. We could template this for N streams via grid.x
// but v1.1 stays simple.
extern "C" __global__ void encode_step_kernel(
    const int32_t* __restrict__ cdf,         // (N, V+1) int32 — value 65536 doesn't fit uint16
    int cdf_stride,
    const int32_t* __restrict__ symbols,
    int N,
    uint8_t* __restrict__ output_buf,
    RangeState* state)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t low = state->low;
    uint32_t high = state->high;
    uint32_t pending = state->pending;
    uint32_t bit_offset = state->bit_offset;

    for (int i = 0; i < N; i++) {
        const int sym = symbols[i];
        const uint32_t sym_lo = (uint32_t)cdf[i * cdf_stride + sym];
        const uint32_t sym_hi = (uint32_t)cdf[i * cdf_stride + sym + 1];
        const uint64_t rng = (uint64_t)(high - low) + 1ULL;

        // 32-bit range × 16-bit cdf = 48-bit product, fits uint64.
        high = low + (uint32_t)((rng * (uint64_t)sym_hi) >> CDF_PRECISION) - 1u;
        low  = low + (uint32_t)((rng * (uint64_t)sym_lo) >> CDF_PRECISION);

        // Renormalize.
        while (true) {
            if (high < HALF) {
                emit_bit_with_pending(output_buf, bit_offset, pending, 0u);
                low <<= 1;
                high = (high << 1) | 1u;
            } else if (low >= HALF) {
                emit_bit_with_pending(output_buf, bit_offset, pending, 1u);
                low = (low - HALF) << 1;
                high = ((high - HALF) << 1) | 1u;
            } else if (low >= QTR && high < THREE_QTR) {
                pending++;
                low = (low - QTR) << 1;
                high = ((high - QTR) << 1) | 1u;
            } else {
                break;
            }
            // (low/high are uint32; shift wraparound matches & TOP_MASK)
        }
    }

    state->low = low;
    state->high = high;
    state->pending = pending;
    state->bit_offset = bit_offset;
}

// Final flush — call once after all encode_step_kernel calls for a stream.
extern "C" __global__ void encode_finalize_kernel(
    uint8_t* __restrict__ output_buf,
    RangeState* state)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t low = state->low;
    uint32_t pending = state->pending + 1u;
    uint32_t bit_offset = state->bit_offset;

    const uint32_t bit = (low < QTR) ? 0u : 1u;
    emit_bit_with_pending(output_buf, bit_offset, pending, bit);
    state->bit_offset = bit_offset;
}

// Launch wrappers — `<<<>>>` only valid in .cu files compiled by nvcc.
// main.cpp calls these.

void launch_encode_step(
    const int32_t* cdf, int cdf_stride,
    const int32_t* symbols, int N,
    uint8_t* output_buf, RangeState* state,
    cudaStream_t stream)
{
    encode_step_kernel<<<1, 1, 0, stream>>>(cdf, cdf_stride, symbols, N, output_buf, state);
}

void launch_encode_finalize(
    uint8_t* output_buf, RangeState* state,
    cudaStream_t stream)
{
    encode_finalize_kernel<<<1, 1, 0, stream>>>(output_buf, state);
}

// ---------------------------------------------------------------------------
// Batched encode — B independent AC streams, one symbol per stream per call.
// Each stream owns its own (low, high, pending, bit_offset) state and writes
// into its own output sub-buffer at base_byte_offsets[stream]. Used by
// `compress_chunks_batched` to encode N chunks in lockstep symmetric to
// `decompress_chunks_batched`.
// ---------------------------------------------------------------------------

extern "C" __global__ void encode_step_batched_kernel(
    const int32_t* __restrict__ cdfs,         // [B, V+1]
    int V,
    const int32_t* __restrict__ symbols,      // [B]
    uint8_t* __restrict__ output_buf,
    const int32_t* __restrict__ base_byte_offsets,  // [B]
    RangeState* states,                       // [B]
    int B)
{
    const int stream = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= B) return;

    const int32_t* cdf = cdfs + (size_t)stream * (size_t)(V + 1);
    uint8_t* out = output_buf + base_byte_offsets[stream];

    uint32_t low = states[stream].low;
    uint32_t high = states[stream].high;
    uint32_t pending = states[stream].pending;
    uint32_t bit_offset = states[stream].bit_offset;

    const int sym = symbols[stream];
    const uint32_t sym_lo = (uint32_t)cdf[sym];
    const uint32_t sym_hi = (uint32_t)cdf[sym + 1];
    const uint64_t rng = (uint64_t)(high - low) + 1ULL;
    high = low + (uint32_t)((rng * (uint64_t)sym_hi) >> CDF_PRECISION) - 1u;
    low  = low + (uint32_t)((rng * (uint64_t)sym_lo) >> CDF_PRECISION);

    while (true) {
        if (high < HALF) {
            emit_bit_with_pending(out, bit_offset, pending, 0u);
            low <<= 1;
            high = (high << 1) | 1u;
        } else if (low >= HALF) {
            emit_bit_with_pending(out, bit_offset, pending, 1u);
            low = (low - HALF) << 1;
            high = ((high - HALF) << 1) | 1u;
        } else if (low >= QTR && high < THREE_QTR) {
            pending++;
            low = (low - QTR) << 1;
            high = ((high - QTR) << 1) | 1u;
        } else {
            break;
        }
    }

    states[stream].low = low;
    states[stream].high = high;
    states[stream].pending = pending;
    states[stream].bit_offset = bit_offset;
}

extern "C" __global__ void encode_finalize_batched_kernel(
    uint8_t* __restrict__ output_buf,
    const int32_t* __restrict__ base_byte_offsets,  // [B]
    RangeState* states,                             // [B]
    int B)
{
    const int stream = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= B) return;

    uint8_t* out = output_buf + base_byte_offsets[stream];
    uint32_t low = states[stream].low;
    uint32_t pending = states[stream].pending + 1u;
    uint32_t bit_offset = states[stream].bit_offset;
    const uint32_t bit = (low < QTR) ? 0u : 1u;
    emit_bit_with_pending(out, bit_offset, pending, bit);
    states[stream].bit_offset = bit_offset;
}

void launch_encode_step_batched(
    const int32_t* cdfs, int V,
    const int32_t* symbols,
    uint8_t* output_buf, const int32_t* base_byte_offsets,
    RangeState* states, int B, cudaStream_t stream)
{
    const int threads = 32;
    const int blocks = (B + threads - 1) / threads;
    encode_step_batched_kernel<<<blocks, threads, 0, stream>>>(
        cdfs, V, symbols, output_buf, base_byte_offsets, states, B);
}

void launch_encode_finalize_batched(
    uint8_t* output_buf, const int32_t* base_byte_offsets,
    RangeState* states, int B, cudaStream_t stream)
{
    const int threads = 32;
    const int blocks = (B + threads - 1) / threads;
    encode_finalize_batched_kernel<<<blocks, threads, 0, stream>>>(
        output_buf, base_byte_offsets, states, B);
}
