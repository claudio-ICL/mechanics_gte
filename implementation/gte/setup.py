import os
import sys
path_gte=os.path.abspath('./')
n=0
while (not os.path.basename(path_gte)=='gte') and (n<4):
    path_gte=os.path.dirname(path_gte)
    n+=1
if not os.path.basename(path_gte)=='gte':
    print("path_ gte not found. Instead: {}".format(path_gte))
    raise ValueError()
path_resources=path_gte+'/resources'
sys.path.append(path_gte+'/')
sys.path.append(path_resources+'/')
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
#Setup resources
os.chdir(path_resources+'/')
cwd=os.getcwd()
print("\n\n I am performing setup of resources. Current working directory is "+cwd+"\n")
ext_modules=[
        Extension("*",
            ["*.pyx"],
            libraries=["m"],
            extra_compile_args=["-O3", "-ffast-math","-fopenmp"],
            extra_link_args=['-fopenmp']
            )
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
