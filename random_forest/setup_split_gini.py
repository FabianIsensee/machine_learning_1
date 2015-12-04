from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("split_gini_cython.pyx")
)