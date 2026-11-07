from setuptools import Extension, setup

# Marker only: a platform-tagged wheel must not be installed on the wrong OS.
setup(ext_modules=[Extension("esmc_li._runtime_tag",
                             ["src/runtime_tag.c"], py_limited_api=True)])
