from setuptools import setup, Extension
from torch.utils import cpp_extension

'''
setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
'''

setup(name='mlp_opt',
      ext_modules=[
        cpp_extension.CppExtension(
                    'mlp_opt', ['mlp_opt.cpp'],
                    extra_compile_args=[ '-fopenmp',
                                        '-mno-avx256-split-unaligned-load',
                                        '-mno-avx256-split-unaligned-store',
                                        '-march=native',
                                        "-DOV_CPU_WITH_PROFILER"
                                        #'-g'
                                        ],
                    include_dirs=["../thirdparty/xbyak/xbyak"],
                    extra_link_args=['-lgomp'])
      ],
      include_dirs=['../include'],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )
