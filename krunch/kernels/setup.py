"""
Build the CUDA kernels extension. Run on a GPU host:

    pip install torch  # if not already
    cd krunch/kernels
    python setup.py build_ext --inplace

Produces krunch_ac_cuda*.so next to this file. Installation requires
nvcc + matching torch wheel.

(Binary module is still named `krunch_ac_cuda` for backwards compat —
the source-tree fold from krunch_ac/ into krunch/codec/+kernels/ kept
the .so name to avoid touching the pybind11 module / every Python
import. Renaming is a v1.x cleanup.)
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="krunch_ac_cuda",
    ext_modules=[
        CUDAExtension(
            name="krunch_ac_cuda",
            sources=[
                # pybind + orchestration
                "main.cpp",
                "rwkv/layer_cpp.cpp",
                # Range coder + softmax/CDF
                "range_coder/encode_kernel.cu",
                "range_coder/decode_kernel.cu",
                "range_coder/det_softmax_cdf.cu",
                # RWKV layer kernels
                "rwkv/rwkv_step.cu",
                "rwkv/wkv_kernel.cu",
                "rwkv/premix_kernels.cu",
                # Deterministic matmul variants — production routing in
                # rwkv/layer_cpp.cpp's gemm_fp16:
                #   - det_matmul_tc_async: primary path on sm_80+ (cp.async + WMMA)
                #   - det_matmul_tc_3way_async: KVR fusion at M >= 256
                #   - det_matmul_tc_3way: T=1 single-batch KVR fusion
                #   - det_matmul_tc_mw: head matmul (large N) default
                #   - det_matmul_tc, det_matmul: single-warp / scalar fallbacks
                #   - det_matmul_cublas: KRUNCH_CUBLAS_PINNED=1 opt-in
                #   - det_matmul_tc_bf16: KRUNCH_BF16=1 opt-in (v2 codec, WIP)
                "matmul/det_matmul.cu",
                "matmul/det_matmul_tc.cu",
                "matmul/det_matmul_tc_async.cu",
                "matmul/det_matmul_tc_3way.cu",
                "matmul/det_matmul_tc_3way_async.cu",
                "matmul/det_matmul_tc_mw.cu",
                "matmul/det_matmul_cublas.cu",
                "matmul/det_matmul_tc_bf16.cu",
                "matmul/layer_norm.cu",
                # int8 weight × fp16 act WMMA
                "matmul/det_matmul_int8_tc.cu",
                # W8A8 (per-output-channel symmetric weight, per-row activation)
                "matmul/det_matmul_w8a8_tc.cu",
                "matmul/det_matmul_w8a8_tc_async.cu",
            ],
            include_dirs=[
                # main.cpp + matmul/*.cu + rwkv/*.cu #include "range_coder.cuh"
                # without a path prefix — add range_coder/ to the search
                # path so the layout doesn't leak into source.
                os.path.join(THIS_DIR, "range_coder"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
