#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False


"""
Header File for KMC_Processes_3D

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
cdef class Processes_3D:
    
    cdef:
        np.int32_t lx, ly, length, num_proc, save_events, solver_type
        double [::1] time
        double [::1] time_dep
        double [::1] time_diff
        np.int32_t [::1] atom_index
        np.int32_t [::1] dep_index
        np.int32_t [:,:,::1] lattice
        np.int32_t [:,::1] coordinate_table
        np.int32_t [:,::1] move_table
        np.int8_t [:,::1] process_table
        np.int8_t [::1] process_table_update
        np.int64_t [::1] process_counters
        np.int64_t [::1] event_counters
       
        solver Solver
        
        np.int32_t process, diff_index
        
        double [::1] dep_times
        np.int32_t [:,::1] dep_coordinates
        
        vector[np.int32_t] x_store
        vector[np.int32_t] y_store
        vector[np.int32_t] z_store
        vector[np.int32_t] m_store
        vector[double] t_store
        
    cpdef void Return_Events(self, np.int32_t [::1] X, np.int32_t [::1] Y, np.int32_t [::1] Z, np.int32_t [::1] M, double [::1] T)
    
    cdef void Update_Process_Table(self, np.int32_t atom_index)
    
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)

    cdef void Find_Elementary_Processes(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t atom_index)
    
    cpdef void Check_Neighborhood(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t xx, np.int32_t yy, np.int32_t zz)
    
    cpdef void Update_Dep_Coordinates(self, double [::1] dep_times, np.int32_t [:,::1] dep_coordinates)
    
    cpdef void Deposit(self)
    
    cpdef void Diffuse(self)




# Processes_3D subclasse A
cdef class Processes_3D_A(Processes_3D):    
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)


# Processes_3D subclasse B
cdef class Processes_3D_B(Processes_3D):
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)


# Processes_3D subclasse C
cdef class Processes_3D_C(Processes_3D):
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)


# Processes_3D subclasse D
cdef class Processes_3D_D(Processes_3D):    
    cdef void Check_Move_alt(self, np.int32_t straight, np.int32_t down, np.int32_t numNN, np.int32_t direction)
    
    cdef void Find_Elementary_Processes(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t atom_index)
    
    cpdef void Deposit(self)
    
    cpdef void Diffuse(self)




















# End of Module