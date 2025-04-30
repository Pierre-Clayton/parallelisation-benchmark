import sys
import os
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# Détection de la plateforme pour OpenMP et AVX
extra_compile_args = ["-O3"]
extra_link_args = []
include_dirs = [numpy.get_include()]
library_dirs = []
libraries = []

if sys.platform == 'darwin':
    # Utiliser libomp installé via Homebrew: brew install libomp
    # Détection du préfixe Homebrew (ARM: /opt/homebrew, Intel: /usr/local)
    prefix = None
    for p in ['/opt/homebrew', '/usr/local']:
        if os.path.exists(os.path.join(p, 'include', 'omp.h')):
            prefix = p
            break
    if prefix is None:
        raise RuntimeError(
            'omp.h introuvable, installez libomp (brew install libomp) '
            'et assurez-vous que $(brew --prefix)/include contient omp.h'
        )
    # Ajouter folders pour headers et libs
    include_dirs.append(os.path.join(prefix, 'include'))
    library_dirs.append(os.path.join(prefix, 'lib'))
    libraries.append('omp')
    extra_compile_args += ['-Xpreprocessor', '-fopenmp']
    extra_link_args += ['-fopenmp']
else:
    # Linux/GCC
    extra_compile_args += ['-fopenmp', '-mavx2']
    extra_link_args += ['-fopenmp']
    libraries.append('gomp')  # ou libomp

ext = Extension(
    name='attention',
    sources=['attention.pyx', 'attention_impl.cpp'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    language='c++',
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='attention',
    ext_modules=cythonize([ext], language_level=3),
)