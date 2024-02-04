from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("ssim_batch_cython", sources=["ssim_batch_cython.pyx"], include_dirs=['.', get_include()])
setup(name="ssim_batch_cython", ext_modules=cythonize([ext]))
