#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

#######################################################################################################################
#######################################################################################################################
#########################                                                                     #########################
#########################            PLD Kinetic Monte Carlo Lattice_3D Module              #########################
#########################                                                                     #########################
#######################################################################################################################
#######################################################################################################################

"""
Kinetic Monte Carlo - Lattice_3D Module

Low-Level Cython class implementing the 3D lattice search algorithms. Base class is a simple cubic (001) lattice. More 
complex lattice can be implemented by making a new child class.

Version: 3.4.1a

"""

# import modules
import numpy as np
import time

# cimport modules
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

# KMC modules
from KMC_Miscellaneous_Cy_v3p4p1 cimport modulus
from KMC_Solver_v3p4p1 cimport Solver as solver





#######################################
###   Cython Class: Processes 3D    ###
#######################################
#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Lattice_3D:
    """
    Lattice_3D Base Class: Searches Simple-Cubic (001) lattice for all elementary processes. 
    
    Attributes
    ----------
    lx,ly:        lattice dimensions
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
    

        
    def __init__(self, np.int32_t lx, np.int32_t ly, np.int32_t length, np.int32_t num_proc, np.int32_t solver_type, np.int32_t[:,:,::1] lattice, np.int8_t[:,::1] process_table, np.int8_t[::1] process_table_update, solver Solver):

        # variables
        self.lx = lx
        self.ly = ly
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
                
            # delete process
            elif self.process_table_update[i] == 0 and self.process_table[atom_index,i] == 1:  
                self.process_table[atom_index,i] = 0
                if self.solver_type == 0:
                    self.Solver.Update_Process_Linear(-atom_index,i)
                elif self.solver_type == 1:
                    self.Solver.Update_Process_Binary(-atom_index,i)
                elif self.solver_type == 2:
                    self.Solver.Update_Process_Set(-atom_index,i)
              
                
        
    cdef void Check_Move(self, np.int32_t straight, np.int32_t left, np.int32_t right, np.int32_t down, np.int32_t up, np.int32_t corner_left, 
                    np.int32_t corner_right, np.int32_t corner_left_down, np.int32_t corner_right_down, np.int32_t numNN, np.int32_t direction):
        """ Check all moves for a primary direction. """
        
        cdef np.int32_t i
        
        # 0 = North, 1 = East, 2 = South, 3 = West
        if direction == 0: i = 0
        if direction == 1: i = 1*self.num_proc
        if direction == 2: i = 2*self.num_proc
        if direction == 3: i = 3*self.num_proc
                
        # Check In-Plane Primary Moves
        if down > 0 and straight == 0:
            # Free Diffusion
            if numNN == 0:
                self.process_table_update[i+0] = 1  # Free
            # Detach from Islands        
            elif corner_right == 0 and corner_left == 0:                
                if numNN == 1:
                    self.process_table_update[i+1] = 1  # Detach_1 (from edge)
                elif numNN == 2:
                    self.process_table_update[i+2] = 1  # Detach_2 (from kink)
                elif numNN == 3:
                    self.process_table_update[i+3] = 1  # Detach_3 (from inside island)
            # Edge Diffusion        
            elif (left > 0 and corner_left > 0) or (right > 0 and corner_right > 0):
                if numNN == 1:
                    self.process_table_update[i+4] = 1  # Edge
                elif numNN == 2:
                    self.process_table_update[i+5] = 1  # Edge_Detach_1 (from kink)
                elif numNN == 3:
                    self.process_table_update[i+6] = 1  # Edge_Detach_2 (from inside island)
                
        # Check Out-of-Plane Downhill Moves
        if down == 0 and straight == 0:                
            if numNN == 0:
                self.process_table_update[i+7] = 1  # Down
            elif numNN == 1:
                self.process_table_update[i+8] = 1  # Down_Detach_1 (from edge)
            elif numNN == 2:
                self.process_table_update[i+9] = 1  # Down_Detach_2 (from kink)
            elif numNN == 3:
                self.process_table_update[i+10] = 1  # Down_Detach_3 (from inside island)
        
        # Check Out-of-Plane Uphill Moves        
        if up == 0 and straight > 0:
            if numNN == 0:
                self.process_table_update[i+11] = 1  # Up_Detach_1 (from straight edge, detach from surface)
            elif numNN == 1:
                self.process_table_update[i+12] = 1  # Up_Detach_2 (from kink, breaking 1 in-plane bond)
            elif numNN == 2:
                self.process_table_update[i+13] = 1  # Up_Detach_3 (from inside edge, breaking 2 in-plane bonds)
            elif numNN == 3:
                self.process_table_update[i+14] = 1  # Up_Detach_4 (from inside island, breaking 3 in-plane bonds) 
        
        # Check In-Plane Corner Left Move
        if left > 0 and down > 0 and straight == 0 and corner_left == 0 and corner_left_down > 0:
            if numNN == 1:
                self.process_table_update[i+15] = 1  # Corner_Straight_Left
            elif numNN == 2:
                self.process_table_update[i+16] = 1  # Corner_Straight_Left (from kink)
            elif numNN == 3:
                self.process_table_update[i+17] = 1  # Corner_Straight_Left (from inside island)
                    
        # Check In-Plane Corner Right Move
        if right > 0 and down > 0 and straight == 0 and corner_right == 0 and corner_right_down > 0:
            if numNN == 1:
                self.process_table_update[i+18] = 1  # Corner_Straight_Right
            elif numNN == 2:
                self.process_table_update[i+19] = 1  # Corner_Straight_Right (from kink)
            elif numNN == 3:
                self.process_table_update[i+20] = 1  # Corner_Straight_Right (from inside island)
                    
    
              
    cdef void Find_Elementary_Processes(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t atom_index):
        """ Find all elementary processes for site i,j,k,atom """

        cdef:
            np.int32_t north, east, south, west
            np.int32_t north_down, east_down, south_down, west_down
            np.int32_t north_up, east_up, south_up, west_up
            np.int32_t north_east, south_east, south_west, north_west
            np.int32_t north_east_down, north_west_down, south_east_down, south_west_down
            np.int32_t numNN_north, numNN_east, numNN_south, numNN_west
        
        # reset the updater array
        self.process_table_update[:] = 0
        
        # first check if neighbor is on top - if so no moves available
        if self.lattice[x,y,z+1] > 0:
            pass
        
        else:
            # Check for periodic boundary
            if x > 1 and x < self.lx-1 and y > 1 and y < self.ly-1:
                # direct lookup               
                north = self.lattice[x,y+1,z]
                east = self.lattice[x+1,y,z]
                south = self.lattice[x,y-1,z]
                west = self.lattice[x-1,y,z]
               
                north_down = self.lattice[x,y+1,z-1]
                east_down = self.lattice[x+1,y,z-1]
                south_down = self.lattice[x,y-1,z-1]
                west_down = self.lattice[x-1,y,z-1]
               
                north_up = self.lattice[x,y+1,z+1]
                east_up = self.lattice[x+1,y,z+1]
                south_up = self.lattice[x,y-1,z+1]
                west_up = self.lattice[x-1,y,z+1]
                     
                north_east = self.lattice[x+1,y+1,z]
                south_east = self.lattice[x+1,y-1,z]
                south_west = self.lattice[x-1,y-1,z]
                north_west = self.lattice[x-1,y+1,z]
               
                north_east_down = self.lattice[x+1,y+1,z-1]
                north_west_down = self.lattice[x-1,y+1,z-1]
                south_east_down = self.lattice[x+1,y-1,z-1]
                south_west_down = self.lattice[x-1,y-1,z-1]
                 
            else:            
                # use modulus to wrap around boundary
                north = self.lattice[modulus(x,self.lx),modulus(y+1,self.ly),z]
                east = self.lattice[modulus(x+1,self.lx),modulus(y,self.ly),z]
                south = self.lattice[modulus(x,self.lx),modulus(y-1,self.ly),z]
                west = self.lattice[modulus(x-1,self.lx),modulus(y,self.ly),z]
                
                north_down = self.lattice[modulus(x,self.lx),modulus(y+1,self.ly),z-1]
                east_down = self.lattice[modulus(x+1,self.lx),modulus(y,self.ly),z-1]
                south_down = self.lattice[modulus(x,self.lx),modulus(y-1,self.ly),z-1]
                west_down = self.lattice[modulus(x-1,self.lx),modulus(y,self.ly),z-1]
                
                north_up = self.lattice[modulus(x,self.lx),modulus(y+1,self.ly),z+1]
                east_up = self.lattice[modulus(x+1,self.lx),modulus(y,self.ly),z+1]
                south_up = self.lattice[modulus(x,self.lx),modulus(y-1,self.ly),z+1]
                west_up = self.lattice[modulus(x-1,self.lx),modulus(y,self.ly),z+1]
                      
                north_east = self.lattice[modulus(x+1,self.lx),modulus(y+1,self.ly),z]
                south_east = self.lattice[modulus(x+1,self.lx),modulus(y-1,self.ly),z]
                south_west = self.lattice[modulus(x-1,self.lx),modulus(y-1,self.ly),z]
                north_west = self.lattice[modulus(x-1,self.lx),modulus(y+1,self.ly),z]
                
                north_east_down = self.lattice[modulus(x+1,self.lx),modulus(y+1,self.ly),z-1]
                north_west_down = self.lattice[modulus(x-1,self.lx),modulus(y+1,self.ly),z-1]
                south_east_down = self.lattice[modulus(x+1,self.lx),modulus(y-1,self.ly),z-1]
                south_west_down = self.lattice[modulus(x-1,self.lx),modulus(y-1,self.ly),z-1]
            
            # Number of NN not in the move direction
            numNN_north = 0
            if east > 0: numNN_north += 1
            if south > 0: numNN_north += 1
            if west > 0: numNN_north += 1
            
            numNN_east = 0
            if north > 0: numNN_east += 1
            if south > 0: numNN_east += 1
            if west > 0: numNN_east += 1
            
            numNN_south = 0
            if east > 0: numNN_south += 1
            if north > 0: numNN_south += 1
            if west > 0: numNN_south += 1
            
            numNN_west = 0
            if east > 0: numNN_west += 1
            if south > 0: numNN_west += 1
            if north > 0: numNN_west += 1
            
            # Check along the Primary Moves
            self.Check_Move(north, west, east, north_down, north_up, north_west, north_east ,north_west_down ,north_east_down ,numNN_north ,0)            
            self.Check_Move(east, north, south, east_down, east_up, north_east, south_east, north_east_down, south_east_down, numNN_east, 1)            
            self.Check_Move(south, east, west, south_down, south_up, south_east, south_west, south_east_down, south_east_down, numNN_south, 2)
            self.Check_Move(west, south, north, west_down, west_up, south_west, north_west, south_west_down, north_west_down, numNN_west, 3)
            
        # Update the Process Table
        self.Update_Process_Table(atom_index)



    cdef void Check_Neighborhood(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t xx, np.int32_t yy, np.int32_t zz):
        """ Search all nearest neighbors of sites i,j,k and ii,jj,kk. Sets neighborhood lattice (x,y,z,atom) """
        
        cdef:
            np.int32_t i,j,ii,jj
            np.int32_t xs,ys,xmin,ymin,zmin,zmax
            np.int32_t atom_index
        
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
                
        if y == yy:
            ys = 3; ymin = modulus(y-1,self.ly)
        else:
            ys = 4
            if yy-y > 10: 
                ymin = yy-1 
            elif yy-y < -10:
                ymin = y-1
            else:
                ymin = modulus(min(y,yy)-1,self.ly)
                
        if z == zz:
            zmin = z-1; zmax = z+1
        else:
            zmin = min(z,zz)-1; zmax = max(z,zz)+1
        
        for i in range(xs):
            ii = modulus(xmin+i,self.lx)
            for j in range(ys):
                jj = modulus(ymin+j,self.ly) 
                for k in range(zmin,zmax+1):                    
                    atom_index = self.lattice[ii,jj,k]   # Check site for a particle                 
                    if atom_index > 1:  # > 1 because substrate particle can never move
                        
                        # find all available processes
                        self.Find_Elementary_Processes(ii,jj,k,atom_index)



    
        
        
        
        
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
                        



cdef class Lattice_3D_A(Lattice_3D):
    """
    Processes_3D Class Child Class A: Disables Uphill diffusion with 4 nearest neighbors 
    """
                                
    cdef void Check_Move(self, np.int32_t straight, np.int32_t left, np.int32_t right, np.int32_t down, np.int32_t up, np.int32_t corner_left, 
                    np.int32_t corner_right, np.int32_t corner_left_down, np.int32_t corner_right_down, np.int32_t numNN, np.int32_t direction):
        """ Check all moves for a primary direction. """
        
        cdef np.int32_t i
        
        # 0 = North, 1 = East, 2 = South, 3 = West
        if direction == 0: i = 0
        if direction == 1: i = 1*self.num_proc
        if direction == 2: i = 2*self.num_proc
        if direction == 3: i = 3*self.num_proc
                
        # Check In-Plane Primary Moves
        if down > 0 and straight == 0:
            # Free Diffusion
            if numNN == 0:
                self.process_table_update[i+0] = 1  # Free
            # Detach from Islands        
            elif corner_right == 0 and corner_left == 0:                
                if numNN == 1:
                    self.process_table_update[i+1] = 1  # Detach_1 (from edge)
                elif numNN == 2:
                    self.process_table_update[i+2] = 1  # Detach_2 (from kink)
                elif numNN == 3:
                    self.process_table_update[i+3] = 1  # Detach_3 (from inside island)
            # Edge Diffusion        
            elif (left > 0 and corner_left > 0) or (right > 0 and corner_right > 0):
                if numNN == 1:
                    self.process_table_update[i+4] = 1  # Edge
                elif numNN == 2:
                    self.process_table_update[i+5] = 1  # Edge_Detach_1 (from kink)
                elif numNN == 3:
                    self.process_table_update[i+6] = 1  # Edge_Detach_2 (from inside island)
                
        # Check Out-of-Plane Downhill Moves
        if down == 0 and straight == 0:                
            if numNN == 0:
                self.process_table_update[i+7] = 1  # Down
            elif numNN == 1:
                self.process_table_update[i+8] = 1  # Down_Detach_1 (from edge)
            elif numNN == 2:
                self.process_table_update[i+9] = 1  # Down_Detach_2 (from kink)
            elif numNN == 3:
                self.process_table_update[i+10] = 1  # Down_Detach_3 (from inside island)
        
        # Check Out-of-Plane Uphill Moves        
        if up == 0 and straight > 0:
            if numNN == 0:
                self.process_table_update[i+11] = 1  # Up_Detach_1 (from straight edge, detach from surface)
            elif numNN == 1:
                self.process_table_update[i+12] = 1  # Up_Detach_2 (from kink, breaking 1 in-plane bond)
            elif numNN == 2:
                self.process_table_update[i+13] = 1  # Up_Detach_3 (from inside edge, breaking 2 in-plane bonds)
        
        # Check In-Plane Corner Left Move
        if left > 0 and down > 0 and straight == 0 and corner_left == 0 and corner_left_down > 0:
            if numNN == 1:
                self.process_table_update[i+15] = 1  # Corner_Straight_Left
            elif numNN == 2:
                self.process_table_update[i+16] = 1  # Corner_Straight_Left (from kink)
            elif numNN == 3:
                self.process_table_update[i+17] = 1  # Corner_Straight_Left (from inside island)
                    
        # Check In-Plane Corner Right Move
        if right > 0 and down > 0 and straight == 0 and corner_right == 0 and corner_right_down > 0:
            if numNN == 1:
                self.process_table_update[i+18] = 1  # Corner_Straight_Right
            elif numNN == 2:
                self.process_table_update[i+19] = 1  # Corner_Straight_Right (from kink)
            elif numNN == 3:
                self.process_table_update[i+20] = 1  # Corner_Straight_Right (from inside island)
                



            
#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Lattice_3D_B(Lattice_3D):
    """
    Processes 3D Class Child Class B: Disables Detachment Processes. (1/11/22 - Modify to allow corner diffusion from kink site.)
    """
                
    cdef void Check_Move(self, np.int32_t straight, np.int32_t left, np.int32_t right, np.int32_t down, np.int32_t up, np.int32_t corner_left, 
                    np.int32_t corner_right, np.int32_t corner_left_down, np.int32_t corner_right_down, np.int32_t numNN, np.int32_t direction):
        """ Check all moves for a primary direction. """
        
        cdef np.int32_t i
        
        # 0 = North, 1 = East, 2 = South, 3 = West
        if direction == 0: i = 0
        if direction == 1: i = 1*self.num_proc
        if direction == 2: i = 2*self.num_proc
        if direction == 3: i = 3*self.num_proc
                
        # Check In-Plane Primary Moves
        if down > 0 and straight == 0:
            # Free Diffusion
            if numNN == 0:
                self.process_table_update[i+0] = 1  # Free

            # Edge Diffusion        
            elif (left > 0 and corner_left > 0) or (right > 0 and corner_right > 0):
                if numNN == 1:
                    self.process_table_update[i+4] = 1  # Edge
                
        # Check Out-of-Plane Downhill Moves
        if down == 0 and straight == 0:                
            if numNN == 0:
                self.process_table_update[i+7] = 1  # Down
        
        # Check In-Plane Corner Left Move
        if left > 0 and down > 0 and straight == 0 and corner_left == 0 and corner_left_down > 0:
            if numNN == 1:
                self.process_table_update[i+15] = 1  # Corner_Straight_Left
            elif numNN == 2:
                self.process_table_update[i+16] = 1  # Corner_Straight_Left (from kink)
                    
        # Check In-Plane Corner Right Move
        if right > 0 and down > 0 and straight == 0 and corner_right == 0 and corner_right_down > 0:
            if numNN == 1:
                self.process_table_update[i+18] = 1  # Corner_Straight_Right
            elif numNN == 2:
                self.process_table_update[i+19] = 1  # Corner_Straight_Right (from kink)





#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Lattice_3D_C(Lattice_3D):
    """
    Processes 3D Class Child Class C: Disables Edge Diffusion. Also doesn't allow 4NN uphill diffusion.
    """
                
    cdef void Check_Move(self, np.int32_t straight, np.int32_t left, np.int32_t right, np.int32_t down, np.int32_t up, np.int32_t corner_left, 
                    np.int32_t corner_right, np.int32_t corner_left_down, np.int32_t corner_right_down, np.int32_t numNN, np.int32_t direction):
        """ Check all moves for a primary direction. """
        
        cdef np.int32_t i
        
        # 0 = North, 1 = East, 2 = South, 3 = West
        if direction == 0: i = 0
        if direction == 1: i = 1*self.num_proc
        if direction == 2: i = 2*self.num_proc
        if direction == 3: i = 3*self.num_proc
                
        # Check In-Plane Primary Moves
        if down > 0 and straight == 0:
            # Free Diffusion
            if numNN == 0:
                self.process_table_update[i+0] = 1  # Free
            # Detach from Islands        
            elif corner_right == 0 and corner_left == 0:                
                if numNN == 1:
                    self.process_table_update[i+1] = 1  # Detach_1 (from edge)
                elif numNN == 2:
                    self.process_table_update[i+2] = 1  # Detach_2 (from kink)
                elif numNN == 3:
                    self.process_table_update[i+3] = 1  # Detach_3 (from inside island)
                
        # Check Out-of-Plane Downhill Moves
        if down == 0 and straight == 0:                
            if numNN == 0:
                self.process_table_update[i+7] = 1  # Down
            elif numNN == 1:
                self.process_table_update[i+8] = 1  # Down_Detach_1 (from edge)
            elif numNN == 2:
                self.process_table_update[i+9] = 1  # Down_Detach_2 (from kink)
            elif numNN == 3:
                self.process_table_update[i+10] = 1  # Down_Detach_3 (from inside island)
        
        # Check Out-of-Plane Uphill Moves        
        if up == 0 and straight > 0:
            if numNN == 0:
                self.process_table_update[i+11] = 1  # Up_Detach_1 (from straight edge, detach from surface)
            elif numNN == 1:
                self.process_table_update[i+12] = 1  # Up_Detach_2 (from kink, breaking 1 in-plane bond)
            elif numNN == 2:
                self.process_table_update[i+13] = 1  # Up_Detach_3 (from inside edge, breaking 2 in-plane bonds)
#            elif numNN == 3:
#                self.process_table_update[i+14] = 1  # Up_Detach_4 (from inside island, breaking 3 in-plane bonds) 


        


#--------------------------------------------------------------------------------------------------------------------------------------------------




cdef class Lattice_3D_D(Lattice_3D):
    """
    Processes_3D Class Child Class D: Disables Uphill diffusion 
    """
                                
    cdef void Check_Move(self, np.int32_t straight, np.int32_t left, np.int32_t right, np.int32_t down, np.int32_t up, np.int32_t corner_left, 
                    np.int32_t corner_right, np.int32_t corner_left_down, np.int32_t corner_right_down, np.int32_t numNN, np.int32_t direction):
        """ Check all moves for a primary direction. """
        
        cdef np.int32_t i
        
        # 0 = North, 1 = East, 2 = South, 3 = West
        if direction == 0: i = 0
        if direction == 1: i = 1*self.num_proc
        if direction == 2: i = 2*self.num_proc
        if direction == 3: i = 3*self.num_proc
                
        # Check In-Plane Primary Moves
        if down > 0 and straight == 0:
            # Free Diffusion
            if numNN == 0:
                self.process_table_update[i+0] = 1  # Free
            # Detach from Islands        
            elif corner_right == 0 and corner_left == 0:                
                if numNN == 1:
                    self.process_table_update[i+1] = 1  # Detach_1 (from edge)
                elif numNN == 2:
                    self.process_table_update[i+2] = 1  # Detach_2 (from kink)
                elif numNN == 3:
                    self.process_table_update[i+3] = 1  # Detach_3 (from inside island)
            # Edge Diffusion        
            elif (left > 0 and corner_left > 0) or (right > 0 and corner_right > 0):
                if numNN == 1:
                    self.process_table_update[i+4] = 1  # Edge
                elif numNN == 2:
                    self.process_table_update[i+5] = 1  # Edge_Detach_1 (from kink)
                elif numNN == 3:
                    self.process_table_update[i+6] = 1  # Edge_Detach_2 (from inside island)
                
        # Check Out-of-Plane Downhill Moves
        if down == 0 and straight == 0:                
            if numNN == 0:
                self.process_table_update[i+7] = 1  # Down
            elif numNN == 1:
                self.process_table_update[i+8] = 1  # Down_Detach_1 (from edge)
            elif numNN == 2:
                self.process_table_update[i+9] = 1  # Down_Detach_2 (from kink)
            elif numNN == 3:
                self.process_table_update[i+10] = 1  # Down_Detach_3 (from inside island)
        
        # Check In-Plane Corner Left Move
        if left > 0 and down > 0 and straight == 0 and corner_left == 0 and corner_left_down > 0:
            if numNN == 1:
                self.process_table_update[i+15] = 1  # Corner_Straight_Left
            elif numNN == 2:
                self.process_table_update[i+16] = 1  # Corner_Straight_Left (from kink)
            elif numNN == 3:
                self.process_table_update[i+17] = 1  # Corner_Straight_Left (from inside island)
                    
        # Check In-Plane Corner Right Move
        if right > 0 and down > 0 and straight == 0 and corner_right == 0 and corner_right_down > 0:
            if numNN == 1:
                self.process_table_update[i+18] = 1  # Corner_Straight_Right
            elif numNN == 2:
                self.process_table_update[i+19] = 1  # Corner_Straight_Right (from kink)
            elif numNN == 3:
                self.process_table_update[i+20] = 1  # Corner_Straight_Right (from inside island)        
        
        
        
        
# End of Module