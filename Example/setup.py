import sys
import os
import subprocess as sp
import numpy
from distutils.core import setup, Extension

libraries = []

libraries.extend(['boost_numpy3', 'boost_python3'])

mace = Extension('cMace',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y'], 
        include_dirs = ['/usr/local/include', 'affy/sdk/'],
        libraries = libraries,
        library_dirs = ['/usr/local/lib'],
        sources = ['python-api.cpp', 'affy/sdk/file/FileIO.cpp', 'affy/sdk/file/CELFileData.cpp', 'affy/sdk/file/CDFFileData.cpp'] 
        )

setup (name = 'cMace',
       version = '0.2.2',
       url = 'https://github.com/aaalgo/mace',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [mace], #, picpac_legacy],
       )

