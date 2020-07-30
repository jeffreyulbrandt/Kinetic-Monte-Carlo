#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

#######################################################################################################################
#######################################################################################################################
#########################                                                                     #########################
#########################            PLD Kinetic Monte Carlo Processes_3D Module              #########################
#########################                                                                     #########################
#######################################################################################################################
#######################################################################################################################

"""
Kinetic Monte Carlo - Processes_3D Module

Low-Level Cython class implementing the 3D lattice search algorithms

Version: 3.3a

"""

# import modules
import numpy as np
import time

# cimport modules
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

# KMC modules
from KMC_Miscellaneous_Cy_v3p3 cimport modulus
from KMC_Solver_v3p3 cimport Solver as solver





#######################################
###   Cython Class: Processes 3D    ###
#######################################
#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Processes_3D:
    """
    Processes_3D Base Class: Searches lattice for all elementary processes. 
    
    Attributes
    ----------
    lx,ly:        lattice dimensions
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
    

        
    def __init__(self, np.int32_t lx, np.int32_t ly, np.int32_t length, np.int32_t num_proc, np.int32_t save_events, np.int32_t solver_type, double[::1] time, double[::1] time_dep, double[::1] time_diff, np.int32_t[::1] atom_index, np.int32_t[::1] dep_index, np.int32_t[:,:,::1] lattice, np.int32_t[:,::1] coordinate_table, np.int32_t[:,::1] move_table, np.int8_t[:,::1] process_table, np.int8_t[::1] process_table_update, np.int64_t[::1] process_counters, np.int64_t[::1] event_counters, solver Solver):

        # variables
        self.lx = lx
        self.ly = ly
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
        
        
        
    cpdef void Return_Events(self, np.int32_t[::1] X, np.int32_t[::1] Y, np.int32_t[::1] Z, np.int32_t[::1] M, double[::1] T):
        """ Return event data to numpy arrays and clear store vectors. """
        
        cdef Py_ssize_t i
        
        # copy over the data
        for i in range(self.x_store.size()):
            X[i] = self.x_store[i]
            Y[i] = self.y_store[i]
            Z[i] = self.z_store[i]
            M[i] = self.m_store[i]
            T[i] = self.t_store[i]
            
        # clear vectors
        self.x_store.clear()
        self.y_store.clear()
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



    cpdef void Check_Neighborhood(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t xx, np.int32_t yy, np.int32_t zz):
        """ Search all nearest neighbors of sites i,j,k and ii,jj,kk. Sets neighborhood lattice (x,y,z,atom) """
        
        cdef:
            np.int32_t i,j,ii,jj
            np.int32_t xs,ys,xmin,ymin,zmin,zmax
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
                    atom = self.lattice[ii,jj,k]   # Check site for a particle                 
                    if atom > 1:  # > 1 because substrate particle can never move
                        atom_index = atom - 2    # particle ID index
                        
                        # find all available processes
                        self.Find_Elementary_Processes(ii,jj,k,atom_index)



    cpdef void Update_Dep_Coordinates(self, double[::1] dep_times, np.int32_t[:,::1] dep_coordinates):
        """ Update Dep Coordinate Values at start of new pulse """
                
        self.dep_times = dep_times
        self.dep_coordinates = dep_coordinates
    
    
                            
    cpdef void Deposit(self):
        """ Deposit an atom into lattice and update time """
                                                      
        # Rename variables: atom->atom_index, atom_dep->dep_index. m==atom
        
        cdef np.int32_t test,x,y,z
        
        # Get Deposition Coordinates
        x = self.dep_coordinates[0,self.dep_index[0]]   # get the deposition x coordinate
        y = self.dep_coordinates[1,self.dep_index[0]]   # get the deposition y coordinate
        
        # Find a z coordinate to land
        z = 0
        test = 1
        while test != 0:
            z += 1
            test = self.lattice[x,y,z]
            
        # Future: Energetic Processes here
            
        # Deposit particle into lattice (atom_index + 2)
        self.lattice[x,y,z] = self.atom_index[0] + 2
        self.coordinate_table[self.atom_index[0],0] = x   # store x position
        self.coordinate_table[self.atom_index[0],1] = y   # store y position
        self.coordinate_table[self.atom_index[0],2] = z   # store z position
        
        # Search the Neighborhood and update Process Tables and Lists
        self.Check_Neighborhood(x,y,z,x,y,z)
        
        # Advance simulation time
        self.time[0] = self.dep_times[self.dep_index[0]]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.y_store.push_back(y)
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
        
#        time.sleep(1)
#        print ('process selected')
        
        # Select a particle (note: index is particle_id - 2)
        if self.solver_type == 0 or self.solver_type == 1:
            self.diff_index = self.Solver.Select_Particle(self.process)-2
        elif self.solver_type == 2:
            self.diff_index = self.Solver.Select_Particle_Set(self.process)-2
        
#        time.sleep(1)
#        print ('particle selected')
        
        # Update Event Counters
        self.event_counters[0] += 1  # total events
        self.event_counters[1] += 1  # dep event
        
        
      
    cpdef void Diffuse(self):
        """ Diffuse an atom according to KMC algorithm """
                                                      
        # Note: Here diff_index is current index of particle selected to move
        
        cdef np.int32_t x,y,z,x_old,y_old,z_old
        
        # particles current positions
        x_old = self.coordinate_table[self.diff_index,0]
        y_old = self.coordinate_table[self.diff_index,1]
        z_old = self.coordinate_table[self.diff_index,2]
    
        # new position (Move Table Index: [(x,y,z),process])
        x = modulus(x_old+self.move_table[self.process,0],self.lx)
        y = modulus(y_old+self.move_table[self.process,1],self.ly)
        z = z_old + self.move_table[self.process,2]
        
        # delete id from old position and add to new position
        self.lattice[x_old,y_old,z_old] = 0
        self.lattice[x,y,z] = self.diff_index + 2
        
        self.coordinate_table[self.diff_index,0] = x
        self.coordinate_table[self.diff_index,1] = y
        self.coordinate_table[self.diff_index,2] = z
    
        # Search the Neighborhood and update Process Tables and Lists
        self.Check_Neighborhood(x,y,z,x_old,y_old,z_old)
        
        # Advance simulation time
        self.time[0] += self.time_diff[0]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.y_store.push_back(y)
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




cdef class Processes_3D_A(Processes_3D):
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


cdef class Processes_3D_B(Processes_3D):
    """
    Processes 3D Class Child Class B: Disables Detachment Processes
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
                    
        # Check In-Plane Corner Right Move
        if right > 0 and down > 0 and straight == 0 and corner_right == 0 and corner_right_down > 0:
            if numNN == 1:
                self.process_table_update[i+18] = 1  # Corner_Straight_Right





#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Processes_3D_C(Processes_3D):
    """
    Processes 3D Class Child Class C: Disables Edge Diffusion
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
            elif numNN == 3:
                self.process_table_update[i+14] = 1  # Up_Detach_4 (from inside island, breaking 3 in-plane bonds) 





#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Processes_3D_D(Processes_3D):
    """
    Processes 3D Class Child Class D: Disables Edge and Detachment Processes.
    
    Note: This raised a bug in the code where when no process were available, solver would crash. This has a band-aid fix
    by overiding the deposit and diffuse methods to check for this senario.
    
    """
                
    cdef void Check_Move_alt(self, np.int32_t straight, np.int32_t down, np.int32_t numNN, np.int32_t direction):
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
                
        # Check Out-of-Plane Downhill Moves
        if down == 0 and straight == 0:                
            if numNN == 0:
                self.process_table_update[i+7] = 1  # Down
                


    cdef void Find_Elementary_Processes(self, np.int32_t x, np.int32_t y, np.int32_t z, np.int32_t atom_index):
        """ Find all elementary processes for site i,j,k,atom """

        cdef:
            np.int32_t north, east, south, west
            np.int32_t north_down, east_down, south_down, west_down
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
            
            # Number of NN not in the move direction (fourth one is the move direction)
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
            self.Check_Move_alt(north, north_down, numNN_north ,0)            
            self.Check_Move_alt(east, east_down, numNN_east, 1)            
            self.Check_Move_alt(south, south_down, numNN_south, 2)
            self.Check_Move_alt(west, west_down, numNN_west, 3)
            
        # Update the Process Table
        self.Update_Process_Table(atom_index)



            
    cpdef void Deposit(self):
        """ Deposit an atom into lattice and update time """
                                                      
        # Rename variables: atom->atom_index, atom_dep->dep_index. m==atom
        
        cdef np.int32_t test,x,y,z,tot
        
        # Get Deposition Coordinates
        x = self.dep_coordinates[0,self.dep_index[0]]   # get the deposition x coordinate
        y = self.dep_coordinates[1,self.dep_index[0]]   # get the deposition y coordinate
        
        # Find a z coordinate to land
        z = 0
        test = 1
        while test != 0:
            z += 1
            test = self.lattice[x,y,z]
            
        # Future: Energetic Processes here
            
        # Deposit particle into lattice (atom_index + 2)
        self.lattice[x,y,z] = self.atom_index[0] + 2
        self.coordinate_table[self.atom_index[0],0] = x   # store x position
        self.coordinate_table[self.atom_index[0],1] = y   # store y position
        self.coordinate_table[self.atom_index[0],2] = z   # store z position
        
        # Search the Neighborhood and update Process Tables and Lists
        self.Check_Neighborhood(x,y,z,x,y,z)
        
        # Advance simulation time
        self.time[0] = self.dep_times[self.dep_index[0]]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.y_store.push_back(y)
            self.z_store.push_back(z)
            self.m_store.push_back(self.atom_index[0]+2)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
            
        # Update Deposition Indices
        self.atom_index[0] += 1
        self.dep_index[0] += 1
        
        # Calculate time to next dep event
        self.time_dep[0] = self.dep_times[self.dep_index[0]] - self.time[0]
        
        # Check if any process is possible, if so call Solver, else set long time_diff
        tot = self.Solver.Calc_Total_Processes()
        if tot == 0:
            self.time_diff[0] = 1e10
            
        elif tot > 0:        
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
        
        cdef np.int32_t x,y,z,x_old,y_old,z_old
        
        # particles current positions
        x_old = self.coordinate_table[self.diff_index,0]
        y_old = self.coordinate_table[self.diff_index,1]
        z_old = self.coordinate_table[self.diff_index,2]
    
        # new position (Move Table Index: [(x,y,z),process])
        x = modulus(x_old+self.move_table[self.process,0],self.lx)
        y = modulus(y_old+self.move_table[self.process,1],self.ly)
        z = z_old + self.move_table[self.process,2]
        
        # delete id from old position and add to new position
        self.lattice[x_old,y_old,z_old] = 0
        self.lattice[x,y,z] = self.diff_index + 2
        
        self.coordinate_table[self.diff_index,0] = x
        self.coordinate_table[self.diff_index,1] = y
        self.coordinate_table[self.diff_index,2] = z
    
        # Search the Neighborhood and update Process Tables and Lists
        self.Check_Neighborhood(x,y,z,x_old,y_old,z_old)
        
        # Advance simulation time
        self.time[0] += self.time_diff[0]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.y_store.push_back(y)
            self.z_store.push_back(z)
            self.m_store.push_back(self.diff_index+2)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
        
        # Update the Process Counter for selected process
        self.process_counters[self.process] += 1
        
        # Calculate time to next dep event
        self.time_dep[0] = self.dep_times[self.dep_index[0]] - self.time[0]

        # Check if any process is possible, if so call Solver, else set long time_diff
        tot = self.Solver.Calc_Total_Processes()
        if tot == 0:
            self.time_diff[0] = 1e10
            
        elif tot > 0:      
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
        
        
        
        
        
        
        
        
# End of Module