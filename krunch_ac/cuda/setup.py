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
            sources=["main.cpp", "encode_kernel.cu", "decode_kernel.cu",
                     "rwkv_step.cu", "mb_gemv.cu", "layer_cpp.cpp",
                     "premix_kernels.cu", "det_matmul.cu",
                     "det_softmax_cdf.cu", "det_matmul_tc.cu",
                     "det_matmul_tc_3way.cu", "det_matmul_cublas.cu",
                     "det_matmul_tc_v2.cu", "layer_norm.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
