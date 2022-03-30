#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

################################################################################################################################
################################################################################################################################
#########################                                                                              #########################
#########################            PLD Kinetic Monte Carlo Non-Thermal Processes Module              #########################
#########################                                                                              #########################
################################################################################################################################
################################################################################################################################

"""
Kinetic Monte Carlo -  Non-Thermal Processes Module

Handles the fast non-thermal processes: Downward Funneling, Transient Mobility, and Island Break-up

Version: 3.4.1a

"""



# import python modules
import numpy as np
import time

# cimport cython modules
cimport numpy as np
cimport cython





#########################################
###   Cython Class: KMC_Non_Thermal   ###
#########################################
#----------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Non_Thermal_Processes_3D:
    """
    Class to perform any of the non-thermal or energetic processes. This is outside of the KMC algorithm. It essential takes the deposited
    particles and moves them to a new location. This is considered instantaneous, so does not update the time.
    
    """
    
    cdef:
        np.int32_t  lx, ly
        
        np.int32_t [:,:,::1] lattice
        np.int8_t [:,::1] df_moves
        
        
    
    def __init__(self, np.int32_t  lx, np.int32_t  ly, np.int32_t [:,:,::1] lattice, np.int8_t [:,::1] df_moves):
        pass

   
    
    cdef Down_Funnel(self):
        """ Calculate the local minimum and move particle. Return new coordinates. """
        pass



    cdef Return_Coodinates(self):
        """ """
        pass






#----------------------------------------------------------------------------------------------------------------------------------------------------

#cdef class Non_Thermal_Processes_2D(Non_Thermal_Processes_3D):











# End of Module