#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

###########################################################################################################################
###########################################################################################################################
#########################                                                                         #########################
#########################            PLD Kinetic Monte Carlo Miscellaneous_Cy Module              #########################
#########################                                                                         #########################
###########################################################################################################################
###########################################################################################################################


"""
This module contains miscellaneous C functions and extension types used in the other modules.

Version: 3.4.1a

"""

# import modules
import numpy as np

# cimport modules
cimport numpy as np
cimport cython
from libc.time cimport time,time_t
from libc.math cimport sqrt
from libcpp.vector cimport vector





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


#############################
###   C-level Functions   ###
#############################



# Cython Modulus Function (python like modulus)    
@cython.cdivision(True)
cdef np.int32_t modulus(np.int32_t a, np.int32_t b):
    c = (a%b + b)%b 
    return c




# Surface Height calculation - 3D
cdef void Calc_Surface_3D(np.int32_t[:,:,::1] Lattice, np.int32_t[:,::1] Surface):
    """ Covert 3D Lattice to 2D Surface. Top down approach. """
    cdef: 
        Py_ssize_t i,j
        np.int32_t k,m
    
    for i in range(Lattice.shape[0]):
        for j in range(Lattice.shape[1]):    
                
            # top down algorithm
            m = 0
            k = Lattice.shape[2] - 1
            while m == 0:
                m = Lattice[i,j,k]
                k -= 1
                
            Surface[i,j] = k + 1
                    
                    
 
                   
# Surface Height calculation - 2D
cdef void Calc_Surface_2D(np.int32_t[:,::1] Lattice, np.int32_t[::1] Surface):
    """ Covert 2D Lattice to 1D Surface. Top down approach. """
    cdef:
        Py_ssize_t i
        np.int32_t k,m
    
    for i in range(Lattice.shape[0]):
        
        # top down algorithm
        m = 0
        k = Lattice.shape[1] - 1
        while m == 0:
            m = Lattice[i,k]
            k -= 1
            
        Surface[i] = k + 1





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


##########################
###   Cython Classes   ###
##########################
        
# Under development
    

# Cython Timer
    
cdef class Cy_Timer:
    """ 
    Data Structure to store timer data and calculate timer stats. CUrrently using Epcoh time which is in
    seconds, so is not very useful. Need a high-resoultion timer.
    """
    
    
    def __init__(self):
        self.timer_begin = time(NULL)
                
    
    cdef void Clear(self):
        """ clear and reset the vector """
        self.store.times.clear()
        
    
    cdef void Timer_Start(self):
        """ begin a timer """        
        self.start_time = time(NULL)

        
    cdef time_t Record_Time(self):
        """ Save delta time """
        cdef time_t t
        
        t = time(NULL)-self.start_time
        self.store_times.push_back(t)
        
        return t

       
    @cython.cdivision(True)     
    cdef double Average(self):
        """ cacluate average deviation of data in store_times """
        cdef Py_ssize_t i
        
        self.avg = 0
        for i in range(self.store_times.size()):
            self.avg += self.store_times[i]
        self.avg = self.avg/self.store_times.size()
        
        return self.avg   
    
    
    @cython.cdivision(True)     
    cdef double StDev(self):
        """ cacluate average std deviation of data in store_times """
        cdef Py_ssize_t i
        
        self.std = 0
        for i in range(self.store_times.size()):
            self.std += (self.store_times[i] - self.avg)**2
        self.std = self.std/self.store_times.size()
        self.std = sqrt(self.std)
        
        return self.std    





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

####################################
###   Random Number Generators   ###
####################################



cdef class Ran:
    """
    Pseudo Random Number Generator
    
    Inputs:
    -------
    seed:  Integer to seed generator
    
    Methods:
    --------
    Int64:      outputs an unsigned 64-bit intgeter
    Int64_n:    output an unsigned 64-bit integer in the range 0 <= x < n
    Float64:    outputs a 64-bit floating point number in the range 0.0 < x < 1.0
    Float64_n:  outputs a 64-bit floating point number in the range 0.0 < x < n
    Int32:      outputs an unsigned 32-bit integer
    
    """

    
    def __init__(self, np.uint64_t seed):      
        self.seed = seed
        
        self.a = 4294957665U
        self.b = 0xffffffff
        
        self.v = 4101842887655102017LL
        self.w = 1
        
        self.u = self.seed ^ self.v
        self.u = self.Int64()
        
        self.v = self.u
        self.v = self.Int64()
        
        self.w = self.v
        self.w = self.Int64()
        
        
    cdef np.uint64_t Int64(self):
        """ return a 64-bit unsigned random integer """
        
        self.u = self.u*2862933555777941757LL + 7046029254386353087LL
        
        self.v = self.v ^ (self.v >> 17)
        self.v = self.v ^ (self.v << 31)
        self.v = self.v ^ (self.v >> 8)
        
        self.w = self.a*(self.w & self.b) + (self.w >> 32)
        
        self.x = self.u ^ (self.u << 21)
        self.x = self.x ^ (self.x >> 35)
        self.x = self.x ^ (self.x << 4)
        
        return (self.x + self.v) ^ self.w
    
    
    @cython.cdivision(True)
    cdef np.uint64_t Int64_n(self, np.uint64_t n):
        """ return a 64-bit unsigned random integer in the range 0 <= x < n, not including n. n must be greater than 0. """
        return self.Int64()%n
    
    
    cdef double Float64(self):
        """ return a 64-bit floating point number in the range 0.0 < x < 1.0 """
        return 5.42101086242752217e-20*self.Int64()
    
    
    cdef double Float64_n(self, double n):
        """ return a 64-bit floating point number in the range 0.0 < x < n """
        return n*5.42101086242752217e-20*self.Int64()
    
    
    cdef np.uint32_t Int32(self):
        """ return a 32-bit unsigned integer """
        return self.Int64()






# End of Module