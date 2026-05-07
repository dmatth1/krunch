"""
Build the CUDA range-coder extension. Run on a GPU host:

    pip install torch  # if not already
    cd krunch_ac/cuda
    python setup.py build_ext --inplace

Produces krunch_ac_cuda*.so next to this file. Installation requires
nvcc + matching torch wheel.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="krunch_ac_cuda",
    ext_modules=[
        CUDAExtension(
            name="krunch_ac_cuda",
            sources=[
                # pybind + orchestration
                "main.cpp", "layer_cpp.cpp",
                # Arithmetic coder (encode + decode + softmax/CDF)
                "encode_kernel.cu", "decode_kernel.cu", "det_softmax_cdf.cu",
                # RWKV layer kernels (single-step + WKV recurrence + premix + LN)
                "rwkv_step.cu", "wkv_kernel.cu",
                "premix_kernels.cu", "layer_norm.cu",
                # Deterministic matmul variants — production routing in
                # layer_cpp.cpp's gemm_fp16:
                #   - det_matmul_tc_async: primary path on sm_80+ (cp.async + WMMA)
                #   - det_matmul_tc_3way_async: KVR fusion at M >= 256
                #   - det_matmul_tc_3way: T=1 single-batch KVR fusion
                #   - det_matmul_tc_mw: head matmul (large N) default
                #   - det_matmul_tc, det_matmul: single-warp / scalar fallbacks
                #   - det_matmul_cublas: KRUNCH_CUBLAS_PINNED=1 opt-in
                #   - det_matmul_tc_bf16: KRUNCH_BF16=1 opt-in (v2 codec, WIP)
                "det_matmul.cu", "det_matmul_tc.cu",
                "det_matmul_tc_async.cu", "det_matmul_tc_3way.cu",
                "det_matmul_tc_3way_async.cu", "det_matmul_tc_mw.cu",
                "det_matmul_cublas.cu", "det_matmul_tc_bf16.cu",
                # int8 weight × fp16 act WMMA — Phase 2 of int8 sprint
                # (per-input-channel uint8, inline dequant in K-loop).
                "det_matmul_int8_tc.cu",
                # W8A8 int8-Tensor-Core matmul + per-row int8 activation
                # quantization — Phase 2A of W8A8 sprint (per-output-channel
                # symmetric weight, per-row symmetric activation, native
                # int8 WMMA fragments). 2× compute throughput on Ampere
                # (250 TOPS int8 vs 125 TFLOPS fp16 TC).
                "det_matmul_w8a8_tc.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
