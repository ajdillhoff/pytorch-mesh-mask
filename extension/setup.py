from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(name='compute_mesh_mask',
      ext_modules=[
          CUDAExtension('compute_mesh_mask_cuda', [
              'gpu/compute_mesh_mask_cuda.cpp',
              'gpu/compute_mesh_mask.cu'
          ],
                        extra_compile_args=['-std=c++14']),
          CppExtension('compute_mesh_mask', [
              'cpp/compute_mesh_mask.cpp'
          ],
                       extra_compile_args=['-std=c++14'])
      ],
      cmdclass={'build_ext': BuildExtension})
