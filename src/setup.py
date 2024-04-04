from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'matrix_processing',
        ['matrix_processing.cpp', 'image_handler.cpp', 'IQR.cpp', 'kernels.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            '/usr/include/opencv4',

            # Path to Eigen headers
            '/usr/include/eigen3',
            get_pybind_include(),

        ],
        libraries=["opencv_core","opencv_highgui","opencv_imgproc"],
        language='c++'
    ),
]

setup(
    name='matrix_processing',
    version='0.0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python module for processing matrices using C++',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
