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
    for i in 0..rows {
        let row = &mat[i * cols..(i + 1) * cols];
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += row[j] * x[j];
        }
        out[i] = acc;
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
