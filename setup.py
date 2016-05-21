from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

numpy_inc = numpy.get_include()

extensions = [
    Extension("redescendl.redescend_likelihood",
			["redescendl/redescend_likelihood.pyx"],
        include_dirs = [numpy_inc],
		)
	]

setup(
    name = "Redescend Likelihood Influence",
    ext_modules = cythonize(extensions),
	packages=['redescendl'],
)
