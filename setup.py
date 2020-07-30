
"""
Setup.py for KMC rev 3.3

"""

# import modeuls
import numpy

from distutils.core import setup
from Cython.Build import cythonize

setup(name = 'KMC_Simulation_rev3p3',
      include_dirs = [numpy.get_include()],
      ext_modules=cythonize(["KMC_Engine_v3p3.pyx",
                             "KMC_Processes_3D_v3p3.pyx",
                             "KMC_Processes_2D_v3p3.pyx",
                             "KMC_Solver_v3p3.pyx",
                             "KMC_Miscellaneous_Cy_v3p3.pyx",
                             "KMC_Analysis_v3p3.pyx"],annotate=True))


