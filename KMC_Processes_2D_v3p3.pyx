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

Version: 3.3a

"""

# import modules
import numpy as np

# cimport modules
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

# KMC modules
from KMC_Miscellaneous_Cy_v3p3 cimport modulus
from KMC_Solver_v3p3 cimport Solver as solver





#######################################
###   Cython Class: Processes 23D    ###
#######################################
#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Processes_2D:
    """
    Processes_2D Base Class: Searches lattice for all elementary processes. 
    
    Attributes
    ----------
    lx:           lattice dimensions
    length:       number of elementary processes
    num_Proc:     number of elementary process rates (doens't differentiate move)
    save_events:  eneable saving all events
    solver_type:  which colver type to use
    time:         simulation time
    time_dep:     time to next dep event
    time_diff:    time to next diffusion event
    atom_index:   index of particle in process and coordinate tables
    dep_index:    index of current particle to be deposited
    
    lattice:               lattice array
    coordinate_table:      array of particle coordinates in lattice
    move_table:            array of coordinate change for each type of move
    process_table:         array of available moves for each particle
    process_table_update:  array to update available moves
    process_counters:      count how many processes are chosen
    event_counters:        count total number of events
    
    process:     index of an elementary process
    diff_index:  index of a particle to undergo diffusion
    
    
    Cython Classes
    --------------
    Solver:      Instance of Solver object
    
    
    Methods
    -------
    Return_Events:              Return event data to python
    Update_Process_Table:       Updates the process table
    Check_Move:                 Check all possible moves along primary direction
    Find_Elementary_Processes:  Find all possible moves for a specific particle
    Check_Neighborhood:         Define a local neigborhood and check all particle in it
    Update_Dep_Coordinates:     Get deposition data from python
    Deposit:                    Deposit a new particle
    Diffuse:                    Diffuse a particle
    
    """
    

        
    def __init__(self, np.int32_t lx, np.int32_t length, np.int32_t num_proc, np.int32_t save_events, np.int32_t solver_type, double[::1] time, double[::1] time_dep, double[::1] time_diff, np.int32_t[::1] atom_index, np.int32_t[::1] dep_index, np.int32_t[:,::1] lattice, np.int32_t[:,::1] coordinate_table, np.int32_t[:,::1] move_table, np.int8_t[:,::1] process_table, np.int8_t[::1] process_table_update, np.int64_t[::1] process_counters, np.int64_t[::1] event_counters, solver Solver):

        # variables
        self.lx = lx
        self.length = length
        self.num_proc = num_proc
        self.save_events = save_events    # 0 = no save, 1 = save
        self.solver_type = solver_type    # Solver Type: Vector_Linear (0), Vector_Binary (1), Set (2)
        
        # times
        self.time = time              # current simulation time
        self.time_dep = time_dep      # delta t to next dep event
        self.time_diff = time_diff    # delta t to next diffusion event
        
        # deposition indices
        self.atom_index = atom_index      # index of atom in process and coordinate table
        self.dep_index = dep_index        # index of depositing atom in the dep_coordinates array
        
        # memoryviews
        self.lattice = lattice
        self.coordinate_table = coordinate_table
        self.move_table = move_table
        self.process_table = process_table
        self.process_table_update = process_table_update
        self.process_counters = process_counters
        self.event_counters = event_counters
        
        # define new variables
        self.process = 0        # index of an elementary process
        self.diff_index = 0     # index of a particle to undergo diffusion process
    
        # Instance of Solver
        self.Solver = Solver
        
        
        
    cpdef void Return_Events(self, np.int32_t[::1] X, np.int32_t[::1] Z, np.int32_t[::1] M, double[::1] T):
        """ Return event data to numpy arrays and clear store vectors. """
        
        cdef Py_ssize_t i
        
        # copy over the data
        for i in range(self.x_store.size()):
            X[i] = self.x_store[i]
            Z[i] = self.z_store[i]
            M[i] = self.m_store[i]
            T[i] = self.t_store[i]
            
        # clear vectors
        self.x_store.clear()
        self.z_store.clear()
        self.m_store.clear()
        self.t_store.clear()
        self.event_counters[3] = 0
        


    cdef void Update_Process_Table(self, np.int32_t atom_index):
        """ Update the process table and create list update for Solver. """
        
        # Note: Should be storing particle id which is 2 higher than atom
        
        cdef np.int32_t i
        cdef np.int32_t m = atom_index + 2
        
        for i in range(self.length):
            # nothing changes
            if self.process_table_update[i] == self.process_table[atom_index,i]:  
                pass

            # new process added
            elif self.process_table_update[i] == 1 and self.process_table[atom_index,i] == 0:   
                self.process_table[atom_index,i] = 1
                if self.solver_type == 0:
                    self.Solver.Update_Process_Linear(m,i)
                elif self.solver_type == 1:
                    self.Solver.Update_Process_Binary(m,i)
                elif self.solver_type == 2:
                    self.Solver.Update_Process_Set(m,i)
                
            # delete process from list
            elif self.process_table_update[i] == 0 and self.process_table[atom_index,i] == 1:  
                self.process_table[atom_index,i] = 0
                if self.solver_type == 0:
                    self.Solver.Update_Process_Linear(-m,i)
                elif self.solver_type == 1:
                    self.Solver.Update_Process_Binary(-m,i)
                elif self.solver_type == 2:
                    self.Solver.Update_Process_Set(-m,i)
              
                
        
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



    cpdef void Check_Neighborhood(self, np.int32_t x, np.int32_t z, np.int32_t xx, np.int32_t zz):
        """ Search all nearest neighbors of sites i,k and ii,kk. Sets neighborhood lattice (x,z,atom) """
        
        cdef:
            Py_ssize_t i,k
            np.int32_t ii
            np.int32_t xs,xmin,zmin,zmax
            np.int32_t atom,atom_index
        
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
                    atom_index = atom - 2    # particle ID index
                    
                    # find all available processes
                    self.Find_Elementary_Processes(ii,k,atom_index)



    cpdef void Update_Dep_Coordinates(self, double[::1] dep_times, np.int32_t[::1] dep_coordinates):
        """ Update Dep Coordinate Values at start of new pulse """
                
        self.dep_times = dep_times
        self.dep_coordinates = dep_coordinates
    
    
                            
    cpdef void Deposit(self):
        """ Deposit an atom into lattice and update time """
                                                      
        # Rename variables: atom->atom_index, atom_dep->dep_index. m==atom
        
        cdef np.int32_t test,x,z
        
        # Get Deposition Coordinates
        x = self.dep_coordinates[self.dep_index[0]]   # get the deposition x coordinate
        
        # Find a z coordinate to land
        z = 0
        test = 1
        while test != 0:
            z += 1
            test = self.lattice[x,z]
            
        # Future: Energetic Processes here
            
        # Deposit particle into lattice (atom_index + 2)
        self.lattice[x,z] = self.atom_index[0] + 2
        self.coordinate_table[self.atom_index[0],0] = x   # store x position
        self.coordinate_table[self.atom_index[0],1] = z   # store z position
        
        # Search the Neighborhood and update Process Tables and Lists
        self.Check_Neighborhood(x,z,x,z)
        
        # Advance simulation time
        self.time[0] = self.dep_times[self.dep_index[0]]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.z_store.push_back(z)
            self.m_store.push_back(self.atom_index[0]+2)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
            
        # Update Deposition Indices
        self.atom_index[0] += 1
        self.dep_index[0] += 1
        
        # Calculate time to next dep event
        self.time_dep[0] = self.dep_times[self.dep_index[0]] - self.time[0]
        
        # Call Solver to advance KMC algorithm
        self.Solver.Calc_Partial_Rates()              # calculate partial rates
        self.Solver.Calc_Cummulative_Rates()          # calculate cummulative rates    
        self.Solver.Calc_tdiff()                      # calculate time to next diffusion event
        self.process = self.Solver.Select_Process()   # select a process
        
        # Select a particle (note: index is particle_id - 2)
        if self.solver_type == 0 or self.solver_type == 1:
            self.diff_index = self.Solver.Select_Particle(self.process)-2
        elif self.solver_type == 2:
            self.diff_index = self.Solver.Select_Particle_Set(self.process)-2
        
        # Update Event Counters
        self.event_counters[0] += 1  # total events
        self.event_counters[1] += 1  # dep event
        
        
      
    cpdef void Diffuse(self):
        """ Diffuse an atom according to KMC algorithm """
                                                      
        # Note: Here diff_index is current index of particle selected to move
        
        cdef np.int32_t x,z,x_old,z_old
        
        # particles current positions
        x_old = self.coordinate_table[self.diff_index,0]
        z_old = self.coordinate_table[self.diff_index,1]
    
        # new position (Move Table Index: [(x,y,z),process])
        x = modulus(x_old+self.move_table[self.process,0],self.lx)
        z = z_old + self.move_table[self.process,1]
        
        # delete id from old position and add to new position
        self.lattice[x_old,z_old] = 0
        self.lattice[x,z] = self.diff_index + 2
        
        self.coordinate_table[self.diff_index,0] = x
        self.coordinate_table[self.diff_index,1] = z
    
        # Search the Neighborhood and update Process Tables and Lists
        self.Check_Neighborhood(x,z,x_old,z_old)
        
        # Advance simulation time
        self.time[0] += self.time_diff[0]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.z_store.push_back(z)
            self.m_store.push_back(self.diff_index+2)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
        
        # Update the Process Counter for selected process
        self.process_counters[self.process] += 1
        
        # Calculate time to next dep event
        self.time_dep[0] = self.dep_times[self.dep_index[0]] - self.time[0]

        # Call Solver to advance KMC algorithm
        self.Solver.Calc_Partial_Rates()              # calculate partial rates
        self.Solver.Calc_Cummulative_Rates()          # calculate cummulative rates    
        self.Solver.Calc_tdiff()                      # calculate time to next diffusion event
        self.process = self.Solver.Select_Process()   # select a process
        
        # Select a particle (note: index is particle_id - 2)
        if self.solver_type == 0 or self.solver_type == 1:
            self.diff_index = self.Solver.Select_Particle(self.process)-2
        elif self.solver_type == 2:
            self.diff_index = self.Solver.Select_Particle_Set(self.process)-2
        
        # Update Event Counters
        self.event_counters[0] += 1  # total events
        self.event_counters[2] += 1  # diffuse event
        
        
        
        
#--------------------------------------------------------------------------------------------------------------------------------------------------


        
        
        
        
        
        
        
# End of Module