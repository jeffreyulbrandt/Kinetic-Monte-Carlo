#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False


"""
Header File for KMC_Processes_2D

Version: 3.3a

"""

# import modules
import numpy as np

# cimport modules
cimport numpy as np
from libcpp.vector cimport vector

# KMC modules
from KMC_Solver_v3p3 cimport Solver as solver



# Processes_3D Base Class
cdef class Processes_2D:
    
    cdef:
        np.int32_t lx, length, num_proc, save_events, solver_type
        double [::1] time
        double [::1] time_dep
        double [::1] time_diff
        np.int32_t [::1] atom_index
        np.int32_t [::1] dep_index
        np.int32_t [:,::1] lattice
        np.int32_t [:,::1] coordinate_table
        np.int32_t [:,::1] move_table
        np.int8_t [:,::1] process_table
        np.int8_t [::1] process_table_update
        np.int64_t [::1] process_counters
        np.int64_t [::1] event_counters
       
        solver Solver
        
        np.int32_t process, diff_index
        
        double [::1] dep_times
        np.int32_t [::1] dep_coordinates
        
        vector[np.int32_t] x_store
        vector[np.int32_t] z_store
        vector[np.int32_t] m_store
        vector[double] t_store
        
    cpdef void Return_Events(self, np.int32_t [::1] X,  np.int32_t [::1] Z, np.int32_t [::1] M, double [::1] T)
    
    cdef void Update_Process_Table(self, np.int32_t atom_index)
    
    cdef void Check_Move(self,np.int32_t backward, np.int32_t left, np.int32_t right, np.int32_t forward_left, np.int32_t forward_right, np.int32_t num_NN, np.int32_t direction)

    cdef void Find_Elementary_Processes(self, np.int32_t x, np.int32_t z, np.int32_t atom_index)
    
    cpdef void Check_Neighborhood(self, np.int32_t x, np.int32_t z, np.int32_t xx, np.int32_t zz)
    
    cpdef void Update_Dep_Coordinates(self, double [::1] dep_times, np.int32_t [::1] dep_coordinates)
    
    cpdef void Deposit(self)
    
    cpdef void Diffuse(self)



















# End of Module