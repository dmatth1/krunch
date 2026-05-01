// GPU range-coder decode kernel. One thread per stream — like encode,
// the work is fundamentally serial within a stream (next interval depends
// on previous). The win comes from keeping the (V+1)-entry CDF on the GPU:
// decompress's autoregressive `m.forward([last_input])` already needs to
// pay one Python-int sync per step; that latency is what bounds us, not
// the CDF transfer.
//
// Per token:
//   1. target = ((value - low + 1) * T - 1) / rng
//   2. binary search CDF for sym such that cdf[sym] <= target < cdf[sym+1]
//   3. update low/high from sym's interval
//   4. renormalize (E1/E2/E3), reading bits from input_buf

#include "range_coder.cuh"
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t read_bit(
    const uint8_t* input_buf,
    uint32_t& bit_offset)
{
    const uint32_t byte_idx = bit_offset >> 3;
    const uint32_t bit_in_byte = 7u - (bit_offset & 7u);  // MSB-first
    bit_offset++;
    return (input_buf[byte_idx] >> bit_in_byte) & 1u;
}

// Initialize decoder state by reading the first PRECISION bits.
extern "C" __global__ void decode_init_kernel(
    const uint8_t* __restrict__ input_buf,
    DecodeState* state)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFu;
    uint32_t value = 0;
    uint32_t bit_offset = 0;
    for (int i = 0; i < PRECISION; i++) {
        value = (value << 1) | read_bit(input_buf, bit_offset);
    }
    state->low = low;
    state->high = high;
    state->value = value;
    state->bit_offset = bit_offset;
}

// Decode one symbol given the per-step CDF. Mutates state in place.
// Writes the symbol to *out_sym. CPU pulls *out_sym.item() to use as
// next forward-pass input.
extern "C" __global__ void decode_step_kernel(
    const int32_t* __restrict__ cdf,         // (V+1,) int32, cdf[V] == CDF_T
    int V,
    const uint8_t* __restrict__ input_buf,
    DecodeState* state,
    int32_t* out_sym)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t low = state->low;
    uint32_t high = state->high;
    uint32_t value = state->value;
    uint32_t bit_offset = state->bit_offset;

    const uint64_t rng = (uint64_t)(high - low) + 1ULL;
    // target = floor( ((value - low + 1) * T - 1) / rng )
    // arithmetic in uint64 to avoid overflow: (32+24)-bit product fits.
    const uint64_t value_off = (uint64_t)(value - low) + 1ULL;
    const uint64_t target = (value_off * (uint64_t)CDF_T - 1ULL) / rng;

    // Binary search: largest sym in [0, V) with cdf[sym+1] <= target.
    // Equivalent: smallest sym with cdf[sym+1] > target.
    int lo = 0, hi = V;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if ((uint32_t)cdf[mid + 1] <= (uint32_t)target) lo = mid + 1;
        else hi = mid;
    }
    const int sym = lo;
    *out_sym = sym;

    const uint32_t sym_lo = (uint32_t)cdf[sym];
    const uint32_t sym_hi = (uint32_t)cdf[sym + 1];
    high = low + (uint32_t)((rng * (uint64_t)sym_hi) >> CDF_PRECISION) - 1u;
    low  = low + (uint32_t)((rng * (uint64_t)sym_lo) >> CDF_PRECISION);

    // Renormalize. Mirror cpu_reference.RangeDecoder.decode_symbol.
    while (true) {
        if (high < HALF) {
            // E1 — low half. No state shift needed before reading bit.
        } else if (low >= HALF) {
            low -= HALF;
            high -= HALF;
            value -= HALF;
        } else if (low >= QTR && high < THREE_QTR) {
            low -= QTR;
            high -= QTR;
            value -= QTR;
        } else {
            break;
        }
        low = (low << 1) & 0xFFFFFFFFu;
        high = ((high << 1) | 1u) & 0xFFFFFFFFu;
        value = ((value << 1) | read_bit(input_buf, bit_offset)) & 0xFFFFFFFFu;
    }

    state->low = low;
    state->high = high;
    state->value = value;
    state->bit_offset = bit_offset;
}

// Launch wrappers — only `<<<>>>` syntax in .cu files.
void launch_decode_init(
    const uint8_t* input_buf, DecodeState* state, cudaStream_t stream)
{
    decode_init_kernel<<<1, 1, 0, stream>>>(input_buf, state);
}

void launch_decode_step(
    const int32_t* cdf, int V,
    const uint8_t* input_buf,
    DecodeState* state,
    int32_t* out_sym,
    cudaStream_t stream)
{
    decode_step_kernel<<<1, 1, 0, stream>>>(cdf, V, input_buf, state, out_sym);
}

// ---------------------------------------------------------------------------
// Batched variants — B independent streams in one launch. Each stream owns
// a (low, high, value, bit_offset) tuple and reads from `input_buf` starting
// at `base_byte_offsets[stream]`. CDFs are laid out as `[B, V+1]` row-major
// (each step uses the same V across streams). One thread per stream; work
// within a stream is still serial (range-coder dependency chain), so the
// win comes from issuing one launch per timestep instead of B launches.
// ---------------------------------------------------------------------------

extern "C" __global__ void decode_init_batched_kernel(
    const uint8_t* __restrict__ input_buf,
    const int32_t* __restrict__ base_byte_offsets,  // [B]
    DecodeState* states,                            // [B]
    int B)
{
    const int stream = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= B) return;

    const uint8_t* in = input_buf + base_byte_offsets[stream];
    uint32_t bit_offset = 0;
    uint32_t value = 0;
    for (int i = 0; i < PRECISION; i++) {
        value = (value << 1) | read_bit(in, bit_offset);
    }
    states[stream].low = 0;
    states[stream].high = 0xFFFFFFFFu;
    states[stream].value = value;
    states[stream].bit_offset = bit_offset;
}

extern "C" __global__ void decode_step_batched_kernel(
    const int32_t* __restrict__ cdfs,   // [B, V+1]
    int V,
    const uint8_t* __restrict__ input_buf,
    const int32_t* __restrict__ base_byte_offsets,  // [B]
    DecodeState* states,                // [B]
    int32_t* out_syms,                  // [B]
    int B)
{
    const int stream = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= B) return;

    const int32_t* cdf = cdfs + (size_t)stream * (size_t)(V + 1);
    const uint8_t* in = input_buf + base_byte_offsets[stream];

    uint32_t low = states[stream].low;
    uint32_t high = states[stream].high;
    uint32_t value = states[stream].value;
    uint32_t bit_offset = states[stream].bit_offset;

    const uint64_t rng = (uint64_t)(high - low) + 1ULL;
    const uint64_t value_off = (uint64_t)(value - low) + 1ULL;
    const uint64_t target = (value_off * (uint64_t)CDF_T - 1ULL) / rng;

    int lo = 0, hi = V;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if ((uint32_t)cdf[mid + 1] <= (uint32_t)target) lo = mid + 1;
        else hi = mid;
    }
    const int sym = lo;
    out_syms[stream] = sym;

    const uint32_t sym_lo = (uint32_t)cdf[sym];
    const uint32_t sym_hi = (uint32_t)cdf[sym + 1];
    high = low + (uint32_t)((rng * (uint64_t)sym_hi) >> CDF_PRECISION) - 1u;
    low  = low + (uint32_t)((rng * (uint64_t)sym_lo) >> CDF_PRECISION);

    while (true) {
        if (high < HALF) {
            // E1
        } else if (low >= HALF) {
            low -= HALF; high -= HALF; value -= HALF;
        } else if (low >= QTR && high < THREE_QTR) {
            low -= QTR; high -= QTR; value -= QTR;
        } else {
            break;
        }
        low = (low << 1) & 0xFFFFFFFFu;
        high = ((high << 1) | 1u) & 0xFFFFFFFFu;
        value = ((value << 1) | read_bit(in, bit_offset)) & 0xFFFFFFFFu;
    }

    states[stream].low = low;
    states[stream].high = high;
    states[stream].value = value;
    states[stream].bit_offset = bit_offset;
}

void launch_decode_init_batched(
    const uint8_t* input_buf, const int32_t* base_byte_offsets,
    DecodeState* states, int B, cudaStream_t stream)
{
    const int threads = 32;
    const int blocks = (B + threads - 1) / threads;
    decode_init_batched_kernel<<<blocks, threads, 0, stream>>>(
        input_buf, base_byte_offsets, states, B);
}

void launch_decode_step_batched(
    const int32_t* cdfs, int V,
    const uint8_t* input_buf, const int32_t* base_byte_offsets,
    DecodeState* states, int32_t* out_syms,
    int B, cudaStream_t stream)
{
    const int threads = 32;
    const int blocks = (B + threads - 1) / threads;
    decode_step_batched_kernel<<<blocks, threads, 0, stream>>>(
        cdfs, V, input_buf, base_byte_offsets, states, out_syms, B);
}
