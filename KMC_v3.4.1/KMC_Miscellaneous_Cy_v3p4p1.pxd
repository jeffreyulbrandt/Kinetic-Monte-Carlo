#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False


"""
Header File for KMC_Miscellaneous_Cy

Version: 3.4.1a

"""


# import modules
import numpy as np

# cimport modules
cimport numpy as np
cimport cython
from libc.time cimport time,time_t
from libcpp.vector cimport vector



# Cython Modulus Function (python like modulus)    
cdef np.int32_t modulus(np.int32_t,np.int32_t)


# Surface functions
cdef void Calc_Surface_3D(np.int32_t[:,:,::1] Lattice, np.int32_t[:,::1] Surface)
cdef void Calc_Surface_2D(np.int32_t[:,::1] Lattice, np.int32_t[::1] Surface)


# Cython Timer class
cdef class Cy_Timer:
    cdef:
        time_t timer_begin
        time_t start_time
        double avg,std
        vector[time_t] store_times
        
    cdef void Clear(self)
    
    cdef void Timer_Start(self)
    
    cdef time_t Record_Time(self)
    
    cdef double Average
    
    cdef double StDev



# Random Number Generator Class
cdef class Ran:

    cdef:
        np.uint64_t seed
        np.uint64_t u,v,w,x
        np.uint64_t a,b
    
    cdef np.uint64_t Int64(self)

    cdef np.uint64_t Int64_n(self, np.uint64_t n)
    
    cdef double Float64(self)

    cdef double Float64_n(self, double n)
    
    cdef np.uint32_t Int32(self)








# End of Module