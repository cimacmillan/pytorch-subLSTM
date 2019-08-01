from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sublstm',
    ext_modules=[
        CUDAExtension('sublstm_cuda', [
            'src/sublstm_cuda.cpp',
            'src/sublstm_cuda_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })