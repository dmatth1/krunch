//! Minimal f32 linear algebra for the RWKV forward pass.
//!
//! We deliberately avoid pulling in a tensor framework (candle,
//! ndarray, burn, etc.) because the L3TC model is tiny (2-3 layers,
//! 96-256 hidden dim) and the ops we need are a small fixed set.
//! Writing them directly gives us full control over allocation,
//! determinism, and numerics, and the total code is smaller than
//! any framework's setup boilerplate.
//!
//! # Conventions
//!
//! - All tensors are `f32` and row-major.
//! - Matrix shapes are `(rows, cols)` a.k.a. `(out_dim, in_dim)` for
//!   weight matrices, matching PyTorch's `nn.Linear` layout. The
//!   L3TC checkpoint stores weights this way.
//! - Vector ops assume 1-D inputs of matching length; the caller is
//!   responsible for shape correctness. Debug builds assert; release
//!   builds are unchecked for speed.
//! - In-place ops take `&mut [f32]` and return nothing. Out-of-place
//!   ops take an output slice as the last argument. No allocation
//!   happens in the hot path.

/// Matrix-vector multiply: `out[i] = sum_j mat[i, j] * x[j]`.
///
/// `mat` is a row-major `(rows, cols)` matrix stored as `rows * cols`
/// contiguous floats.
///
/// For small matrices (row count below `MATVEC_BLAS_THRESHOLD`) this
/// uses a hand-written scalar inner loop that the compiler can
/// autovectorize. For larger matrices — specifically the
/// `(16384, 96)` embedding and head projections — it delegates to
/// `matrixmultiply::sgemm`, which ships hand-tuned SIMD kernels for
/// x86 AVX/AVX2/AVX512 and ARM NEON. The head projection is the
/// dominant cost per token, so this single swap gives us roughly
/// an order-of-magnitude speedup on realistic models.
#[inline]
pub fn matvec(mat: &[f32], x: &[f32], out: &mut [f32]) {
    let rows = out.len();
    let cols = x.len();
    debug_assert_eq!(
        mat.len(),
        rows * cols,
        "matvec: mat shape mismatch (expected {rows}*{cols}, got {})",
        mat.len()
    );
    if rows >= MATVEC_BLAS_THRESHOLD {
        matvec_sgemm(mat, x, out, rows, cols);
    } else {
        matvec_scalar(mat, x, out, rows, cols);
    }
}

/// Minimum row count at which we switch from the scalar matvec
/// implementation to the `matrixmultiply` SIMD kernel.
///
/// Small matrices (the per-layer 96×96 projections) pay more in
/// call overhead than they save in SIMD throughput, so we use the
/// scalar path for them. The embedding and head are much larger
/// and benefit immediately.
const MATVEC_BLAS_THRESHOLD: usize = 512;

/// Tall matrix-vector multiply where the matrix is stored **column-major**.
///
/// This is a different layout convention from [`matvec`]: the
/// matrix data is `(cols, rows)` instead of `(rows, cols)` in
/// memory, such that column `j` of the logical matrix is
/// `mat_col_major[j * rows .. (j + 1) * rows]`.
///
/// The algorithm is `out[i] = sum_j mat[i, j] * x[j]` computed as:
///
/// ```text
/// out = 0
/// for j in 0..cols:
///     axpy: out += x[j] * mat_col_major[j * rows .. (j + 1) * rows]
/// ```
///
/// This is the AXPY (`alpha * x + y`) form that vectorizes
/// beautifully: the inner loop is a constant-stride walk through
/// contiguous memory with a scalar broadcast multiply. LLVM
/// autovectorizes this cleanly into NEON / AVX SIMD without any
/// external crate.
///
/// For very tall matrices like the L3TC head (16384 × 96), this
/// form is dramatically faster than the row-major dot-product form
/// because:
///
/// 1. **Cache behavior**: memory is accessed in streaming order
///    with a stride of 1 across a full column (64 KB for a 16384-row
///    column), hitting the L1 cache predictor perfectly.
/// 2. **SIMD utilisation**: each AXPY is a pure elementwise op
///    that vectorizes 4 or 8 floats per instruction on ARM NEON
///    / x86 AVX with no reduction step.
/// 3. **Register pressure**: the inner loop holds one scalar
///    (`x[j]`) and streams through memory, leaving all the vector
///    registers free for the accumulation.
pub fn matvec_col_major(
    mat_col_major: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(mat_col_major.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);

    // Zero the output
    for v in out.iter_mut() {
        *v = 0.0;
    }

    // For each column, AXPY the column into the output
    for j in 0..cols {
        let col = &mat_col_major[j * rows..(j + 1) * rows];
        let xj = x[j];
        for i in 0..rows {
            out[i] += xj * col[i];
        }
    }
}

/// Transpose a row-major `(rows, cols)` matrix into a column-major
/// buffer of length `rows * cols`.
///
/// The output layout is such that column `j` is at offset
/// `j * rows`, suitable for passing to [`matvec_col_major`].
pub fn transpose(src_row_major: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src_row_major.len(), rows * cols);
    let mut dst = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src_row_major[i * cols + j];
        }
    }
    dst
}

#[inline]
fn matvec_scalar(mat: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mat[i * cols..(i + 1) * cols];
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += row[j] * x[j];
        }
        out[i] = acc;
    }
}

#[inline]
fn matvec_sgemm(mat: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // matrixmultiply::sgemm computes C = alpha*A*B + beta*C for
    // general (m, k, n) shapes. We want:
    //     out (rows,)   = mat (rows, cols) * x (cols,)
    // Viewed as matmul: m = rows, k = cols, n = 1.
    //
    // The safety requirements for the sgemm FFI wrapper are:
    //     - pointers must be valid for the described dimensions
    //     - strides describe the layout correctly (row-stride, col-stride)
    //     - no aliasing between A, B, and C
    //
    // mat is row-major, so row stride = cols, col stride = 1.
    // x is a single column vector, row stride = 1, col stride = 1
    // (there is only one column so col stride is never used).
    // out is a single column, row stride = 1, col stride = 1.
    //
    // matrixmultiply is safe-Rust internally but exposes an unsafe
    // wrapper because raw pointers are needed to express the strides.
    // We only do this in this one function; the unsafe block is
    // scoped to the FFI call.
    #[allow(unsafe_code)]
    unsafe {
        matrixmultiply::sgemm(
            rows,       // m
            cols,       // k
            1,          // n
            1.0,        // alpha
            mat.as_ptr(),
            cols as isize, // row stride of A
            1,             // col stride of A
            x.as_ptr(),
            1, // row stride of B (irrelevant for n=1, but must be valid)
            1, // col stride of B
            0.0, // beta
            out.as_mut_ptr(),
            1, // row stride of C
            1, // col stride of C
        );
    }
}

/// Lookup the `idx`-th row of `mat` (shape `(rows, cols)`) into `out`.
///
/// Used for embedding lookup: `out = emb[idx]`.
#[inline]
pub fn row_lookup(mat: &[f32], cols: usize, idx: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), cols, "row_lookup: output length mismatch");
    let start = idx * cols;
    let end = start + cols;
    debug_assert!(end <= mat.len(), "row_lookup: row index out of range");
    out.copy_from_slice(&mat[start..end]);
}

/// Layer normalization with learned weight and bias.
///
/// Computes: `out = weight * (x - mean) / sqrt(var + eps) + bias`.
/// The mean and variance are taken across the entire `x` slice.
#[inline]
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(weight.len(), n);
    debug_assert_eq!(bias.len(), n);
    debug_assert_eq!(out.len(), n);

    let mean = x.iter().sum::<f32>() / n as f32;
    let mut var = 0.0f32;
    for &v in x {
        let d = v - mean;
        var += d * d;
    }
    var /= n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Element-wise sigmoid: `out = 1 / (1 + exp(-x))`, in place.
#[inline]
pub fn sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Element-wise ReLU: `out = max(0, x)`, in place.
#[inline]
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Element-wise square: `out = x * x`, in place.
#[inline]
pub fn square_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= *v;
    }
}

/// Element-wise exp, in place.
#[inline]
pub fn exp_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.exp();
    }
}

/// Element-wise negation of exp: `out = -exp(x)`, in place.
#[inline]
pub fn neg_exp_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = -v.exp();
    }
}

/// Element-wise addition, in place: `a += b`.
#[inline]
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Element-wise multiplication, in place: `a *= b`.
#[inline]
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// Element-wise max: `out[i] = max(a[i], b[i])`.
#[inline]
pub fn max_elem(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].max(b[i]);
    }
}

/// Time-mix blend used in every RWKV block:
///
/// `out[i] = x[i] * mix[i] + prev[i] * (1 - mix[i])`
#[inline]
pub fn time_mix(x: &[f32], prev: &[f32], mix: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), prev.len());
    debug_assert_eq!(x.len(), mix.len());
    debug_assert_eq!(x.len(), out.len());
    for i in 0..x.len() {
        let m = mix[i];
        out[i] = x[i] * m + prev[i] * (1.0 - m);
    }
}

/// Copy src to dst (wrapper for explicit intent).
#[inline]
pub fn copy_into(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    dst.copy_from_slice(src);
}

/// Find the argmax of a vector. Returns 0 for empty input.
///
/// Used for sanity tests, not the hot path.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Softmax a vector in place. Numerically stable (subtracts max first).
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut m = x[0];
    for &v in x.iter().skip(1) {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_identity() {
        // 3x3 identity
        let m = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ];
        let x = vec![2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 3];
        matvec(&m, &x, &mut out);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn matvec_rectangular() {
        // 2x4 matrix × 4-vec -> 2-vec
        let m = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ];
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 2];
        matvec(&m, &x, &mut out);
        assert_eq!(out, vec![10.0, 26.0]);
    }

    #[test]
    fn layer_norm_basic() {
        // Input [1,2,3,4]; mean=2.5, var=1.25, std≈1.118
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let b = [0.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        layer_norm(&x, &w, &b, 1e-5, &mut out);
        // After normalization, mean should be ~0 and std ~1
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((mean).abs() < 1e-5, "mean after LN should be 0, got {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var after LN should be 1, got {var}");
    }

    #[test]
    fn sigmoid_basic() {
        let mut x = [0.0f32, 1.0, -1.0, 100.0, -100.0];
        sigmoid_inplace(&mut x);
        assert!((x[0] - 0.5).abs() < 1e-6);
        assert!(x[1] > 0.7 && x[1] < 0.74);
        assert!(x[2] > 0.26 && x[2] < 0.3);
        assert!(x[3] > 0.999);
        assert!(x[4] < 0.001);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Monotonic: larger input -> larger output
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1]);
        }
    }

    #[test]
    fn time_mix_extremes() {
        let x = [10.0f32, 20.0];
        let prev = [1.0f32, 2.0];
        let mix_all_one = [1.0f32, 1.0];
        let mix_all_zero = [0.0f32, 0.0];
        let mut out = [0.0f32; 2];
        time_mix(&x, &prev, &mix_all_one, &mut out);
        assert_eq!(out, [10.0, 20.0]);
        time_mix(&x, &prev, &mix_all_zero, &mut out);
        assert_eq!(out, [1.0, 2.0]);
    }

    #[test]
    fn row_lookup_basic() {
        let m = vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ];
        let mut out = [0.0f32; 3];
        row_lookup(&m, 3, 1, &mut out);
        assert_eq!(out, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 2.0]), 2);
        assert_eq!(argmax(&[-1.0, -2.0, -3.0]), 0);
        assert_eq!(argmax(&[]), 0);
    }
}
