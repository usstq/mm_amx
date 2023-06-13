# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sys

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

'''
using intel compiler:
source ~/intel/oneapi/setvars.sh
export CXX=icx CC=icx
'''
ext_modules = [
    # https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
   
    Pybind11Extension("mmbench.dnnl",
        ["mmbench/dnnl.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        # it can locate the site-specific include folder(for calling mkl/oneDNN/...)
        # even for virtualenv environment
        include_dirs=[
            f'{sys.prefix}/include',
            '../include',
        ],
        library_dirs=[ f'{sys.prefix}/lib', ],
        runtime_library_dirs=[ f'{sys.prefix}/lib', ],
        libraries=[
            'pthread',
            'stdc++',
            'gomp',
            'dnnl',
        ],
        extra_compile_args=[ '-fopenmp',],
    ),

    Pybind11Extension("mmbench.mkl",
        ["mmbench/mkl.cpp"],
        # it can locate the site-specific include folder(for calling mkl/oneDNN/...)
        # even for virtualenv environment
        include_dirs=[
            f'{sys.prefix}/include',
            '../include',
        ],
        library_dirs=[ f'{sys.prefix}/lib', ],
        runtime_library_dirs=[ f'{sys.prefix}/lib', ],
        libraries=[
            'pthread',
            'stdc++',
            'mkl_intel_ilp64',
            'mkl_gnu_thread',
            'mkl_core',
            'gomp',
        ],
        extra_compile_args=[ '-fopenmp',],
    ),

    Pybind11Extension("mmbench.mmamx",
        ["mmbench/mmamx.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        # it can locate the site-specific include folder(for calling mkl/oneDNN/...)
        # even for virtualenv environment
        include_dirs=[
            f'{sys.prefix}/include',
            '../include',
        ],
        library_dirs=[ f'{sys.prefix}/lib', ],
        runtime_library_dirs=[ f'{sys.prefix}/lib', ],
        libraries=[
            'pthread',
            'stdc++',
            'iomp5'
        ],
        extra_compile_args=['-fopenmp', '-march=native'],
    ),
]
    
setup(
    name="mmbench",
    version=__version__,
    packages=['mmbench',],
    ext_modules=ext_modules,
    install_requires = [
        "onednn-cpu-gomp"
    ],
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)