#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False


"""
Header File for KMC_Lattice_2D

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
cdef class Lattice_2D:
    
    cdef:
        np.int32_t lx, length, num_proc, solver_type
        np.int32_t [:,::1] lattice
        np.int8_t [:,::1] process_table
        np.int8_t [::1] process_table_update
       
        solver Solver
        
    
    cdef void Update_Process_Table(self, np.int32_t atom_index)
    
    cdef void Check_Move(self,np.int32_t backward, np.int32_t left, np.int32_t right, np.int32_t forward_left, np.int32_t forward_right, np.int32_t num_NN, np.int32_t direction)

    cdef void Find_Elementary_Processes(self, np.int32_t x, np.int32_t z, np.int32_t atom_index)
    
    cdef void Check_Neighborhood(self, np.int32_t x, np.int32_t z, np.int32_t xx, np.int32_t zz)
    




















# End of Module