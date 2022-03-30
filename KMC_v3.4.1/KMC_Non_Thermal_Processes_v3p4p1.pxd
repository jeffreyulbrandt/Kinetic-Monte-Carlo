#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False


"""
Header File for KMC_Non_Thermal_Processes_3D

Version: 3.4.1a

"""

# import modules
import numpy as np

# cimport modules
cimport numpy as np
from libcpp.vector cimport vector

# KMC modules
from KMC_Solver_v3p4p1 cimport Solver as solver



# Processes_3D Base Class
cdef class Non_Thermal_Processes:
    
    cdef:
        np.int32_t lx, ly, length, num_proc, solver_type
        np.int32_t [:,:,::1] lattice
        np.int8_t [:,::1] process_table
        np.int8_t [::1] process_table_update
       
        solver Solver

    
    cdef void Update_Process_Table(self, np.int32_t atom_index)
    