from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hashed_embedding_bag',
    ext_modules=[CUDAExtension(
        'hashed_embedding_bag', 
        [#'hashed_embedding_bag1.cpp',
        'hashed_embedding_bag_kernel.cu'])],
    py_modules=['hashedEmbeddingBag', 'hashedEmbeddingCPU'],
    cmdclass={'build_ext': BuildExtension}
)
