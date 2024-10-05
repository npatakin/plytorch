from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='plytorch',
    packages=['plytorch'],
    ext_modules=[
        CppExtension('_plytorch_extension', ['main.cpp', 'miniply.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version="0.1.0"
)