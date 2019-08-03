from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np


np_include = np.get_include()

ext_modules = [
    Extension(
        "pictorial",
        ["pictorial.pyx"],
        include_dirs=[np_include],
        libraries=["m"],
        extra_compile_args=["-ffast-math", "-O3"],
    )
]

setup(name="pictorial", cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
