import sys
import os
import platform
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# Détection de la plateforme pour OpenMP et AVX
extra_compile_args = ["-O3"]
extra_link_args = []
include_dirs = [numpy.get_include()]
library_dirs = []
libraries = []

# Détection de l'architecture
is_arm = platform.machine() == 'arm64' or platform.machine() == 'aarch64'
print(f"Architecture détectée: {platform.machine()}")

if sys.platform == 'darwin':
    # Chemins spécifiques pour libomp sur ce système
    omp_include = '/opt/homebrew/Cellar/libomp/20.1.3/include'
    omp_lib = '/opt/homebrew/Cellar/libomp/20.1.3/lib'
    
    # Vérifier que les chemins existent
    if not os.path.exists(os.path.join(omp_include, 'omp.h')):
        raise RuntimeError(f"omp.h non trouvé dans {omp_include}")
    
    # Configurer les chemins pour la compilation
    include_dirs.append(omp_include)
    library_dirs.append(omp_lib)
    libraries.append('omp')
    
    # Drapeaux pour le compilateur
    extra_compile_args += ['-Xpreprocessor', '-fopenmp']
    extra_link_args += ['-fopenmp']
    
    # Pour la vectorisation, on utilise uniquement AVX sur les architectures x86
    if not is_arm:
        extra_compile_args.append('-mavx2')
    else:
        print("Architecture ARM détectée: pas d'utilisation des instructions AVX")
    
    # Afficher les configurations pour le débogage
    print(f"Using omp.h from: {omp_include}")
    print(f"Using omp library from: {omp_lib}")
    
else:
    # Linux/GCC
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
    
    # AVX seulement sur x86
    if not is_arm:
        extra_compile_args.append('-mavx2')
    
    libraries.append('gomp')  # ou libomp

# Afficher les configurations finales pour le débogage
print(f"Include dirs: {include_dirs}")
print(f"Library dirs: {library_dirs}")
print(f"Libraries: {libraries}")
print(f"Compile args: {extra_compile_args}")
print(f"Link args: {extra_link_args}")

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