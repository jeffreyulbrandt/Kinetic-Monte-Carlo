#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

#######################################################################################################################
#######################################################################################################################
#########################                                                                     #########################
#########################            PLD Kinetic Monte Carlo Processes_2D Module              #########################
#########################                                                                     #########################
#######################################################################################################################
#######################################################################################################################

"""
Kinetic Monte Carlo - Processes_2D Module

Low-Level Cython class implementing the 2D lattice search algorithms

Version: 3.3.1a

"""

# import modules
import numpy as np

# cimport modules
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

# KMC modules
from KMC_Miscellaneous_Cy_v3p4p1 cimport modulus
from KMC_Solver_v3p4p1 cimport Solver as solver





#######################################
###   Cython Class: Processes 23D    ###
#######################################
#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Lattice_2D:
    """
    Processes_2D Base Class: Searches lattice for all elementary processes. 
    
    
    Attributes
    ----------
    lx:           lattice dimensions
    length:       number of elementary processes
    num_Proc:     number of elementary process rates (doens't differentiate move)
    solver_type:  which colver type to use
    
    lattice:               lattice array
    process_table:         array of available moves for each particle
    process_table_update:  array to update available moves
  
    
    Cython Classes
    --------------
    Solver:      Instance of Solver object
    
    
    Methods
    -------
    Update_Process_Table:       Updates the process table
    Check_Move:                 Check all possible moves along primary direction
    Find_Elementary_Processes:  Find all possible moves for a specific particle
    Check_Neighborhood:         Define a local neigborhood and check all particle in it

    
    """
    

        
    def __init__(self, np.int32_t lx, np.int32_t length, np.int32_t num_proc, np.int32_t solver_type, np.int32_t[:,::1] lattice, np.int8_t[:,::1] process_table, np.int8_t[::1] process_table_update, solver Solver):

        # variables
        self.lx = lx
        self.length = length
        self.num_proc = num_proc
        self.solver_type = solver_type    # Solver Type: Vector_Linear (0), Vector_Binary (1), Set (2)
        
        # memoryviews
        self.lattice = lattice
        self.process_table = process_table
        self.process_table_update = process_table_update
    
        # Instance of Solver
        self.Solver = Solver
        
        


    cdef void Update_Process_Table(self, np.int32_t atom_index):
        """ Update the process table and create list update for Solver. """
        
        cdef np.int32_t i
        
        for i in range(self.length):
            # nothing changes
            if self.process_table_update[i] == self.process_table[atom_index,i]:  
                pass

            # new process added
            elif self.process_table_update[i] == 1 and self.process_table[atom_index,i] == 0:   
                self.process_table[atom_index,i] = 1
                if self.solver_type == 0:
                    self.Solver.Update_Process_Linear(atom_index,i)
                elif self.solver_type == 1:
                    self.Solver.Update_Process_Binary(atom_index,i)
                elif self.solver_type == 2:
                    self.Solver.Update_Process_Set(atom_index,i)
                
            # delete process from list
            elif self.process_table_update[i] == 0 and self.process_table[atom_index,i] == 1:  
                self.process_table[atom_index,i] = 0
                if self.solver_type == 0:
                    self.Solver.Update_Process_Linear(-atom_index,i)
                elif self.solver_type == 1:
                    self.Solver.Update_Process_Binary(-atom_index,i)
                elif self.solver_type == 2:
                    self.Solver.Update_Process_Set(-atom_index,i)
              
                
        
    cdef void Check_Move(self, np.int32_t backward, np.int32_t left, np.int32_t right, np.int32_t forward_left, np.int32_t forward_right, np.int32_t num_NN, np.int32_t direction):
        """ 
        Evaluate General Moves:
        Here, point of view is relative to primary move. So, for Left primary, a left move is down
            
        """
        cdef np.int32_t i
        
        # 0 = Left, 1 = Up, 2 = Right, 3 = Down
        if direction == 0: i = 0
        if direction == 1: i = self.num_proc
        if direction == 2: i = 2*self.num_proc
        if direction == 3: i = 3*self.num_proc
        
        # Check Primary move:
        if (left > 0 and forward_left > 0) or (right > 0 and forward_right > 0):
            if num_NN == 1:
                self.process_table_update[i+0] = 1    # Free Edge Diff
            elif num_NN == 2:
                self.process_table_update[i+1] = 1    # Free Edge Diff + Detach_1
            elif num_NN == 3:
                self.process_table_update[i+2] = 1    # Free Edge Diff + Detach_2
                
        if left > 0 and forward_left > 0 and right > 0 and forward_right > 0:
            if num_NN == 2:
                self.process_table_update[i+3] = 1    # Internal Edge Diff
            elif num_NN == 3:
                self.process_table_update[i+4] = 1    # Internal Edge Diff + Detach_1
                            
        # Check Secondary move to Left
        if left > 0 and forward_left == 0:                
            if num_NN == 1:
                self.process_table_update[i+5] = 1    # Corner Diffusion
            elif num_NN == 2:
                self.process_table_update[i+6] = 1    # Corner + Detach_1
            elif num_NN == 3:
                self.process_table_update[i+7] = 1    # Corner + Detach_2
               
        # Check Secondary move to Right
        if right > 0 and forward_right == 0:
            if num_NN == 1:
                self.process_table_update[i+8] = 1    # Corner Diffusion
            elif num_NN == 2:
                self.process_table_update[i+9] = 1    # Corner + Detach_1
            elif num_NN == 3:
                self.process_table_update[i+10] = 1    # Corner + Detach_2
        
        
              
    cdef void Find_Elementary_Processes(self,np.int32_t x,np.int32_t z,np.int32_t atom_index):
        """ Find all elementary processes for site i,k,atom """
        
        cdef:
            np.int32_t left, right, up, down
            np.int32_t left_down, right_down, left_up, right_up           
            np.int32_t nnn_left, nnn_right, nnn_up, nnn_down
            np.int32_t numNN_left, numNN_right, numNN_up, numNN_down
        
        # reset the updater array
        self.process_table_update[:] = 0
                
        # Calculate occupation of Nearest Neighbors and Next Nearest Neighbors
        left = self.lattice[modulus((x-1),self.lx),z]
        right = self.lattice[modulus((x+1),self.lx),z]
        up = self.lattice[x,z+1]
        down = self.lattice[x,z-1]
        
        left_down = self.lattice[modulus((x-1),self.lx),z-1]
        right_down = self.lattice[modulus((x+1),self.lx),z-1]        
        left_up = self.lattice[modulus((x-1),self.lx),z+1]
        right_up = self.lattice[modulus((x+1),self.lx),z+1]
        
        nnn_left = self.lattice[modulus((x-2),self.lx),z]
        nnn_right = self.lattice[modulus((x+2),self.lx),z]
        nnn_up = self.lattice[x,z+2]
        nnn_down = self.lattice[x,z-2]  # Note: substrate layer must be 2 layers thick for this to work
                
        # Number of NN not in the move direction
        numNN_left = 0
        if right > 0: numNN_left += 1
        if up > 0: numNN_left += 1
        if down > 0: numNN_left += 1
        
        numNN_right = 0
        if left > 0: numNN_right += 1
        if up > 0: numNN_right += 1
        if down > 0: numNN_right += 1
        
        numNN_up = 0
        if left > 0: numNN_up += 1
        if right > 0: numNN_up += 1
        if down > 0: numNN_up += 1
        
        numNN_down = 0
        if left > 0: numNN_down += 1
        if right > 0: numNN_down += 1
        if up > 0: numNN_down += 1
        
        # first check if any nearest neighbor is isolated - if so no moves available
        if left > 0 and nnn_left == 0 and left_up == 0 and left_down == 0:       # Left Isolated Particle
            pass
        elif up > 0 and nnn_up == 0 and left_up == 0 and right_up == 0:          # Up Isolated Particle
            pass
        elif right > 0 and nnn_right == 0 and right_down == 0 and right_up == 0: # Right Isolated Particle
            pass
        elif down > 0 and nnn_down == 0 and left_down == 0 and right_down == 0:  # Down Isolated Particle
            pass
        else:
            if left == 0:
                self.Check_Move(right,down,up,left_down,left_up,numNN_left,0)
            
            if up == 0:
                self.Check_Move(down,left,right,left_up,right_up,numNN_up,1)
            
            if right == 0 :
                self.Check_Move(left,up,down,right_up,right_down,numNN_right,2)
                
            if down == 0 :
                self.Check_Move(up,right,left,right_down,left_down,numNN_down,3)
                      
        # update the Process Table
        self.Update_Process_Table(atom_index)



    cdef void Check_Neighborhood(self, np.int32_t x, np.int32_t z, np.int32_t xx, np.int32_t zz):
        """ Search all nearest neighbors of sites i,k and ii,kk. Sets neighborhood lattice (x,z,atom) """
        
        cdef:
            Py_ssize_t i,k
            np.int32_t ii
            np.int32_t xs,xmin,zmin,zmax
            np.int32_t atom
        
        # determine range to check
        if x == xx:
            xs = 3; xmin = modulus(x-1,self.lx)
        else:
            xs = 4
            if xx-x > 10: 
                xmin = xx-1 
            elif xx-x < -10:
                xmin = x-1
            else:
                xmin = modulus(min(x,xx)-1,self.lx)
                
        if z == zz:
            zmin = z-1; zmax = z+1
        else:
            zmin = min(z,zz)-1; zmax = max(z,zz)+1
        
        for i in range(xs):
            ii = modulus(xmin+i,self.lx)
            for k in range(zmin,zmax+1):                    
                atom = self.lattice[ii,k]   # Check site for a particle                 
                if atom > 1:  # > 1 because substrate particle can never move
                    
                    # find all available processes
                    self.Find_Elementary_Processes(ii,k,atom)

        
        
        
#--------------------------------------------------------------------------------------------------------------------------------------------------


        
        
        
        
        
        
        
# End of Module