#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False


"""
Header File for KMC_Processes_3D

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
cdef class Lattice_3D:
    
    cdef:
        np.int32_t lx, ly, length, num_proc, solver_type
        np.int32_t [:,:,::1] lattice
        np.int8_t [:,::1] process_table
        np.int8_t [::1] process_table_update
       
        solver Solver

    
    cdef void Update_Process_Table(self, np.int32_t atom_index)
    
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)

    cdef void Find_Elementary_Processes(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t atom_index)
    
    cdef void Check_Neighborhood(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t xx, np.int32_t yy, np.int32_t zz)




# Processes_3D subclass A
cdef class Lattice_3D_A(Lattice_3D):    
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)


# Processes_3D subclass B
cdef class Lattice_3D_B(Lattice_3D):
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)


# Processes_3D subclass C
cdef class Lattice_3D_C(Lattice_3D):
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)


# Processes_3D subclass D
cdef class Lattice_3D_D(Lattice_3D):
    cdef void Check_Move(self, np.int32_t Straight, np.int32_t Left, np.int32_t Right, np.int32_t Down, np.int32_t Up, np.int32_t Corner_Left, np.int32_t Corner_Right, np.int32_t Corner_Left_Down, np.int32_t Corner_Right_Down, np.int32_t numNN, np.int32_t Direction)













# End of Module