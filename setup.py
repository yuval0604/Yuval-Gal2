from setuptools import setup, Extension

module = Extension("mykmeanssp", sources=["kmeansmodule.c"])

setup(
    name="mykmeanssp",
    version="1.0",
    ext_modules=[module]
)
