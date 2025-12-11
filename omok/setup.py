# setup.py
from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++17', '-O3']

ext_modules = [
    Extension(
        'mcts_core',
        ['mcts_core.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='mcts_core',
    version='0.1',
    ext_modules=ext_modules,
)