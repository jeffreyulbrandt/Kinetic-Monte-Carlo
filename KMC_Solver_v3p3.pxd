#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False


"""
Header File for KMC_Solver

Version: 3.3a

"""

# import modules
import numpy as np

# cimport modules
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.set cimport set as Sset

# KMC modules
from KMC_Miscellaneous_Cy_v3p3 cimport Ran




# Solver Class
cdef class Solver:
    
    cdef:
        Py_ssize_t i
        np.int32_t solver_type
        double [::1] rates
        np.int32_t length
        double [::1] time
        double [::1] time_diff
        np.int32_t [::1] ni
        double [::1] ri
        double [::1] total_rate
        np.int32_t [::1] max_ni
        np.uint64_t seed
        
        vector[double] cummulative_rates
        
        vector[np.int32_t] store
        vector[vector[np.int32_t]] process_list
        
        Sset[np.int32_t] store_set
        vector[Sset[np.int32_t]] process_list_set
        
        Ran Rand
        
    cpdef void Return_Cummulative_Rates(self, double [::1] c_rates)
    
    cpdef list Return_Process_Lists(self, list store)
    
    cdef void Calc_tdiff(self)
    
    cdef np.int32_t Calc_Total_Processes(self)
    
    cdef void Calc_Partial_Rates(self)
    
    cdef void Calc_Cummulative_Rates(self)
    
    cdef np.int32_t Select_Process(self)
    
    cdef np.int32_t Select_Particle(self, np.int32_t process)
    
    cdef np.int32_t Select_Particle_Set(self, np.int32_t process)
    
    cdef void Update_Process_Linear(self, np.int32_t atom, np.int32_t process)
    
    cdef void Update_Process_Binary(self, np.int32_t atom, np.int32_t process)
    
    cdef void Update_Process_Set(self, np.int32_t atom, np.int32_t process)
    
        


# End of Module