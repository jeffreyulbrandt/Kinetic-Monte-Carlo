
"""
Setup.py for KMC rev 3.4

Command Line: python setup.py build_ext --inplace

"""

# import modules
import numpy

from distutils.core import setup
from Cython.Build import cythonize

#setup(name = 'KMC_Simulation_rev3p4',
#      include_dirs = [numpy.get_include()],
#      ext_modules=cythonize(["KMC_Solver_v3p4.pyx",
#                             "KMC_Lattice_2D_v3p4.pyx",
#                             "KMC_Lattice_3D_v3p4.pyx",
#                             "KMC_Miscellaneous_Cy_v3p4.pyx"],annotate=True))


#setup(name = 'KMC_Simulation_rev3p4',
#      include_dirs = [numpy.get_include()],
#      ext_modules=cythonize(["KMC_Engine_v3p4.pyx",
#                             "KMC_Lattice_2D_v3p4.pyx",
#                             "KMC_Lattice_3D_v3p4.pyx",
#                             "KMC_Solver_v3p4.pyx",
#                             "KMC_Miscellaneous_Cy_v3p4.pyx"],annotate=True))


#setup(name = 'KMC_Simulation_rev3p4',
#      include_dirs = [numpy.get_include()],
#      ext_modules=cythonize(["KMC_Model_Pulse_v3p4.pyx",
#                             "KMC_Engine_v3p4.pyx",
#                             "KMC_Lattice_2D_v3p4.pyx",
#                             "KMC_Lattice_3D_v3p4.pyx",
#                             "KMC_Solver_v3p4.pyx",
#                             "KMC_Miscellaneous_Cy_v3p4.pyx"],annotate=True))


setup(name = 'KMC_Simulation_rev3p4',
      include_dirs = [numpy.get_include()],
      ext_modules=cythonize(["KMC_Engine_v3p4p1.pyx",
                             "KMC_Lattice_3D_v3p4p1.pyx",
                             "KMC_Lattice_2D_v3p4p1.pyx",
                             "KMC_Solver_v3p4p1.pyx",
                             "KMC_Miscellaneous_Cy_v3p4p1.pyx",
                             "KMC_Analysis_v3p4p1.pyx"],annotate=True))


#----------------------------------------
# Test out new version based on the docs
#----------------------------------------
#import numpy
#from setuptools import Extension, setup
#from Cython.Build import cythonize
#
#
#extensions = [Extension("Solver", ["src/cy_files/KMC_Solver_v3p4.pyx"],include_dirs = [numpy.get_include()]),
#              ]
#
#setup(
#    ext_modules = cythonize(extensions)
#)