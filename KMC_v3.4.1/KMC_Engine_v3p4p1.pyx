#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

#################################################################################################################
#################################################################################################################
#########################                                                               #########################
#########################            PLD Kinetic Monte Carlo Engine Module              #########################
#########################                                                               #########################
#################################################################################################################
#################################################################################################################

"""
Kinetic Monte Carlo - Engine Module

Low-level Python Class and Cython Extension controling single step in the KMC algoithm. Contains code to decide 
between deposition and diffusion processes. 

Owns the Cython classes: Solver and Lattice_2D and Lattice_3D

Version: 3.4.1a

"""



# import python modules
import numpy as np
import time

# cimport cython modules
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

# impot KMC modules
import KMC_Processes_v3p4p1 as KMC_Proc
import KMC_Miscellaneous_v3p4p1 as KMC_Misc

# cimport KMC modules
cimport KMC_Solver_v3p4p1 as KMC_Solver
cimport KMC_Lattice_2D_v3p4p1 as KMC_Lattice_2D
cimport KMC_Lattice_3D_v3p4p1 as KMC_Lattice_3D

# cimport functions
from KMC_Miscellaneous_Cy_v3p4p1 cimport modulus,Calc_Surface_2D, Calc_Surface_3D



####################################
###   Python Class: KMC_Engine   ###
####################################
#----------------------------------------------------------------------------------------------------------------------------------------------------

class KMC_Engine:
    """
    KMC_Engine Base Class: Owns data arrays and cython extensions
    
    
    Attributes
    ----------
    solver_type:           Solver algorithm used
    save_events:           Salve all events
    time:                  Array for saving simulation time and dep and diff times
    
    rate_catalog:          Dictionary of elementary process rates
    rates:                 Array of elementary process rates (in order of process lists)
    move_table:            Array of moves for moving particle throughout Lattice    
    keys:                  List of names of each elementary process
    num_proc:              Number of processes per primary direction
    length:                Length of the rates array
    
    ni:                    Array of process list sizes
    ri:                    Partical sum of rates for each process list
    total_rate:            Total rate of all processes
    cummualitve_rates:     Array of cummulative partial rates
    max_ni                 Debug array which stores the largest size of each process list
    
    process_table:         Array of current available process for each particle
    process_table_update:  New status for selected particle
    
    lattice:               Array for storing particle postions and id
    surface:               Array for storing surface height of lattice   
    coordinate_table:      Stores coordinates for each deposited particle
    
    Sim_Stats:             Statistics object
    process_counters:      Keeps track of processes selected
    event_counters:        Kepps track of number of events
    
    
    Cython Classes
    --------------
    Lattice:                   KMC Lattice for spatial portions of the code
    Solver:                    KMC solver algorithm for temporal evolution of simulation
    Non_Thermal_Processes:     Code for executing non-thermal processes
    
    
    Methods
    -------
    Initialize:             Initialize the Engine object
    Add_Substrate:          Build a substrate into lattice
    Activate_Substrate:     Activate the substrate particles into live particles
    Run_Until_Time:         Run KMC algorithm until a specified time
    Run_Until_Step:         Run KMC algorithm until a specified step
    Run_One_Step:           Run KMC algorithm for one step
    Reset_Depsotion_Pulse:  Reset the dep coordinates
    Surface:                Calculate surface height from lattice
    Lattice_Coordinates:    Return table of lattice coordinates of current state
    Save_Events:            Extract and return all event data for save file
    
    """
    
    

    def __init__(self,Parameters,seed):
        self.Parameters = Parameters
        
        # extract useful variables from Parameters Dictionary
        self.enable_plots = self.Parameters['Enable_Plots']
        self.dim = self.Parameters['Dimension']
        
        self.save = self.Parameters['Save_All_Events']      # save all events
        self.stats_path = self.Parameters['Save_Path_Simulation']   # path for stats file
        
        self.lx = self.Parameters['Lx']          # length of substrate in x direction
        self.ly = self.Parameters['Ly']          # length of substrate in y direction (3D only)
        self.n = self.Parameters['n']            # number of particles deposited per pulse
        self.depth = self.Parameters['depth']    # number of incomplete layers (z coordinate)
        self.pulses = self.Parameters['Pulses']  # number of pulses
        
        self.substrate = self.Parameters['Substrate_Type']         # substrate type
        self.state = self.Parameters['Substrate_Particle_State']   # are they active or passive
        
        self.enable = self.Parameters['Enable_Processes']        # Which Thermal Processes are activated
        self.downward = self.Parameters['Downward_Funneling']    # Enable Downward Funneling Process
        self.transient = self.Parameters['Transient_Mobility']    # Enable Transient Mobility Process
        self.chipping = self.Parameters['Island_Chipping']    # Enable Island Break-up/Chipping Process
        
        # Solver Type: Vector_Linear = 0, Vector_Binary = 1, Set = 2
        if self.Parameters['Solver_Type'] == 'Linear':
            self.solver_type = 0
        elif self.Parameters['Solver_Type'] == 'Binary':
            self.solver_type = 1
        elif self.Parameters['Solver_Type'] == 'Set':
            self.solver_type = 2
            
        # seed for random number generator
        self.seed = seed
               
        self.Initialize()
        
        
    
    def Initialize(self):
        """ Initialize the KMC_Engine object """
        
        # save all events?
        if self.save == True:
            self.save_events = 1
        elif self.save == False: 
            self.save_events = 0
        
        # Non thermal processes?
        if self.downward == True or self.transient == True or self.chipping == True:
            self.non_therm = 1
        if self.downward == False and self.transient == False and self.chipping == False:
            self.non_therm = 0
            
        # simulation times
        self.time = np.zeros([3],dtype=np.double)      # current simulation times. [0] = simulation time, [1] = time to next dep, [2] = time to next diff event
            
        # Create the Rate and Process Catalogs and Arrays
        self.Process = KMC_Proc.Process_Catalog(self.Parameters)
        self.rate_catalog, self.rates, self.move_table, self.keys, self.num_proc = self.Process.Return_Rates()
        self.length = self.rates.shape[0]         # length of the rates array
        
        # Create Downward Funneling Moves Table
        self.df_moves = self.Process.Downward_Funneling_Moves()
        
        # Create the Rate Tables    
        self.ni = np.zeros([self.length],dtype=np.int32)      # Total moves possible per process
        self.ri = np.zeros([self.length],dtype=np.double)     # Partial rates for each elementary process        
        self.total_rate = np.array([0.0],dtype=np.double)     # Total Rate of all Processes (array to pass as memoryview to Cython Classes)
        
        # Debug Data strucutres (These aren't used in the KMC algorithm, they are here for the user)
        self.cummulative_rates = np.zeros([self.length],dtype=np.double)    # Cummulative partial rates
        self.max_ni = np.zeros([self.length],dtype=np.int32)                # Keep track of maximum number of particles in each list

        # Create Process Tables
        self.process_table = np.zeros([self.n*self.pulses+2,self.length],dtype=np.int8)    # Array to store available processes for each particle
        self.process_table_update = np.zeros([self.length],dtype=np.int8)                  # Array to test for change in process status
        
        # Create the Lattice, Surface, and Coordinate Table Arrays
        if self.dim == '2D':
            self.lattice = np.zeros([self.lx,self.depth+2],dtype=np.int32)               # adding +2 to depth for the substrate particle layers
            self.surface = np.zeros([self.lx],dtype=np.int32)                            # surface array. Change dtype to np.uint8
            self.coordinate_table = np.zeros([self.n*self.pulses+2,2],dtype=np.int32)    # +2 since 0,1 are not active particles. Could probably be np.uint16
            
        elif self.dim == '3D':
            self.lattice = np.zeros([self.lx,self.ly,self.depth+2],dtype=np.int32)
            self.surface = np.zeros([self.lx,self.ly],dtype=np.int32)    
            self.coordinate_table = np.zeros([self.n*self.pulses+2,3],dtype=np.int32)
            
        # Create Substrate
        self.Add_Substrate()
        
        # Initialize Simulation Statistics Instance
        self.Sim_Stats = KMC_Misc.Statistics(self.enable_plots, self.stats_path, self.keys, self.num_proc)
        self.process_counters = np.zeros([self.length],dtype=np.int64)     # array to store selected processes
        self.event_counters = np.zeros([4],dtype=np.int64)                 # array to store event counts: Total, Dep, Diffuse, Event Vector Size

        
        #####################################################
        ###  Create the KMC_Engine_Cy Extension objects   ###
        #####################################################
        
        # Solver object
        self.Solver = KMC_Solver.Solver(self.solver_type,self.rates,self.length,self.ni,self.ri,self.total_rate,self.max_ni,self.seed)
        
        # dimension specific objects
        if self.dim == '2D':
            self.Lattice = KMC_Lattice_2D.Lattice_2D(self.lx,self.length,self.num_proc,self.solver_type,self.lattice,self.process_table,self.process_table_update,self.Solver)

#            self.Non_Thermal_Processes = KMC_Non_Therm_2D(self.lx,self.lattice,self.df_moves)  

            self.CyEngine = KMC_CyEngine_2D(self.lx,self.length,self.num_proc,self.solver_type,self.save_events,self.non_therm,self.lattice,self.surface,self.time,self.coordinate_table,
                                            self.process_table,self.process_table_update,self.move_table,self.process_counters,self.event_counters,self.Solver,self.Lattice)
            
        elif self.dim == '3D':
            # Create Lattice Object
            if self.enable == 'All':
                self.Lattice = KMC_Lattice_3D.Lattice_3D(self.lx,self.ly,self.length,self.num_proc,self.solver_type,self.lattice,self.process_table,self.process_table_update,self.Solver)
            elif self.enable == 'No_Uphill_4NN':
                self.Lattice = KMC_Lattice_3D.Lattice_3D_A(self.lx,self.ly,self.length,self.num_proc,self.solver_type,self.lattice,self.process_table,self.process_table_update,self.Solver)
            elif self.enable == 'No_Detach':
                self.Lattice = KMC_Lattice_3D.Lattice_3D_B(self.lx,self.ly,self.length,self.num_proc,self.solver_type,self.lattice,self.process_table,self.process_table_update,self.Solver)
            elif self.enable == 'No_Edge':
                self.Lattice = KMC_Lattice_3D.Lattice_3D_C(self.lx,self.ly,self.length,self.num_proc,self.solver_type,self.lattice,self.process_table,self.process_table_update,self.Solver)
            elif self.enable == 'No_Uphill':
                self.Lattice = KMC_Lattice_3D.Lattice_3D_D(self.lx,self.ly,self.length,self.num_proc,self.solver_type,self.lattice,self.process_table,self.process_table_update,self.Solver)
            
            # Create Non Thermal Processes Object
#            self.Non_Thermal_Processes = KMC_Non_Therm_3D(self.lx,self.ly,self.lattice,self.df_moves)
            
            # Create the CyEngine Object
            self.CyEngine = KMC_CyEngine_3D(self.lx,self.ly,self.length,self.num_proc,self.solver_type,self.save_events,self.non_therm,self.lattice,self.surface,self.time,self.coordinate_table,
                                            self.process_table,self.process_table_update,self.move_table,self.process_counters,self.event_counters,self.Solver,self.Lattice)
                        
        # Activate Substrate
        if self.substrate != 'Flat':
            if self.state == 'Active':
                self.Activate_Substrate()
            
            
        
    #############################
    ###   Substrate Methods   ###
    #############################
        
    def Add_Substrate(self):
        """ Set-up a starting substrate.
        
        This function does the following:
            
        Set up the default flat substrate. This is 2 layers to account for next nearest neighbor interactions.
        Call Substrate Generator and request list substrate particle positions and ids
        Place all substrate particles into the lattice
        For active substrates, call Lattice.Check_Neighborhood to set up particles in the process table
        
        """
        
        # Set up the Default Starting Substrate (this is always set)
        if self.dim == '2D':
            self.lattice[:,0:2] = 1  # for 2D, we need two rows of substrate particles
        elif self.dim == '3D':
            self.lattice[:,:,0:2] = 1   # new: 2 rows of substrate atoms
            
        # Create any additional substrate features       
        if self.substrate == 'Flat':
            pass
        elif self.substrate == 'Islands' or self.substrate == 'Steps' or self.substrate == 'Custom':
            # Create instance of Substrate Generator
            self.Substrate = KMC_Misc.Substrate_Generator(self.Parameters)

            # Generate a substrate (substrate_count is 1 higher than last substrate particle)
            self.substrate_list,self.substrate_count = self.Substrate.Generate_Substrate()
            
            # If Passive, just place atoms in the lattice  
            if self.state == 'Passive':
                if self.dim == '2D':
                    for i,k,m in self.substrate_list:
                        self.lattice[i,k] = m
                elif self.dim == '3D':
                    for i,j,k,m in self.substrate_list:
                        self.lattice[i,j,k] = m
                    
            # If Active substrate, expand the Process Table and Coordinate table to make room for substrate particles
            if self.state == 'Active':                    
                self.process_table = np.append(self.process_table,np.zeros([self.substrate_count-2,self.length],dtype=np.int8),axis=0)
                if self.dim == '2D':
                    self.coordinate_table = np.append(self.coordinate_table,np.zeros([self.substrate_count-2,2],dtype=np.int32),axis=0)
                elif self.dim == '3D':
                    self.coordinate_table = np.append(self.coordinate_table,np.zeros([self.substrate_count-2,3],dtype=np.int32),axis=0)
                
        
        
    def Activate_Substrate(self):
        """ Activate the Substrate: For active substrate particles, deposit into the simulation. """

        # Convert substrate list to Dep_Coordinates
        dep_times = np.zeros([len(self.substrate_list)])
        dep_coordinates = np.zeros([2,len(self.substrate_list)],dtype=np.int32)
        
        for i in range(len(self.substrate_list)):
            if self.dim == '2D':
                dep_coordinates[0,i] = self.substrate_list[i][0]
            elif self.dim == '3D':
                dep_coordinates[0,i] = self.substrate_list[i][0]
                dep_coordinates[1,i] = self.substrate_list[i][1]
                
        # Send Dep Coordinates to CyEngine
        self.CyEngine.Update_Dep_Coordinates(dep_times,dep_coordinates)
        
        # Pre-deposit substrate particles
        for i in range(len(self.substrate_list)):            
            self.CyEngine.Step_Deposit()
        
    
                
    ###################################
    ###   KMC Algorithm Sequences   ###
    ###################################
    
    def Run_First_Step(self, dep_times, dep_coordinates):
        """ Run the first step in deposition simulation to get things going """
        
        # set new deposition pulse coordinates
        self.Reset_Deposition_Pulse(dep_times, dep_coordinates)
        
        # run the first deposition step
        self.CyEngine.Step_Deposit()
        
        
        
    def Run_until_Time(self,end_time,anneal):
        """ Run the KMC algorithm until set time """
        
        error_code = self.CyEngine.Run_until_Time(end_time,anneal)
    
        return (self.time[0],self.event_counters[0],error_code)
    
 
    
    def Run_until_Step(self,step,end_time,anneal):
        """ Run the KMC algorithm until set step """
        
        error_code = self.CyEngine.Run_until_Step(step,end_time,anneal)
    
        return (self.time[0],self.event_counters[0],error_code)
    


    def Run_One_Step(self,anneal):
        """ Run one step of the KMC simulation """
        
        if anneal == 0:
            self.CyEngine.Step()
        elif anneal == 1:
            self.CyEngine.Step_Anneal()
            
            
            
    def Reset_Deposition_Pulse(self, dep_times, dep_coordinates):
        """ Reset Deposition coordinates and zero the dep index variable """
        
        self.CyEngine.Update_Dep_Coordinates(dep_times, dep_coordinates)
    
    
    
    ###############################
    ###   Return Data Methods   ###
    ###############################
    
    def Surface(self):
        """ Return surface array """
        
        if self.dim == '2D':
            Calc_Surface_2D(self.lattice,self.surface)
        elif self.dim == '3D':
            Calc_Surface_3D(self.lattice,self.surface)
        
        return self.surface

    
    
    def Lattice_Coordinates(self):
        """ Return coordinates of particles in lattice. (For saving lattice data) """
        
        lattice_coord = self.coordinate_table[2:self.event_counters[1]+2,:]   # cut out the zeros
        
        return lattice_coord
        


    def Save_Events(self):
        """ extract event data for save file. """
        
        # length of event data vectors
        length = self.event_counters[3]
        
        # make numpy arrays
        x_store = np.zeros([length],dtype=np.int32) 
        y_store = np.zeros([length],dtype=np.int32) 
        z_store = np.zeros([length],dtype=np.int32)
        m_store = np.zeros([length],dtype=np.int32)
        t_store = np.zeros([length],dtype=np.double)
        
        # extract the data
        if self.dim == '2D':
            self.CyEngine.Return_All_Events(x_store,z_store,m_store,t_store)
        elif self.dim == '3D':
            self.CyEngine.Return_All_Events(x_store,y_store,z_store,m_store,t_store)
        
        return (x_store,y_store,z_store,m_store,t_store)
    
    
    
    def Return_Counters(self):
        """ Return event and process counters """
        
        return (self.event_counters,self.process_counters)
    
    
    
    def Return_Times(self):
        """ Return simulation times array """
        
        return self.time
    
    
    
    def Get_Cummulative_Rates(self):
        """ Extract cummulative rate data from Solver vector """
        self.Solver.Return_Cummulative_Rates(self.cummulative_rates)
        

       
    def Get_Process_Lists(self):
        """ Extract process list data from Solver vector """
        List = [[] for _ in range(self.length)]
        process_lists = self.Solver.Return_Process_Lists(List)
        
        return process_lists
    
    
    
    ###################################
    ###   Sim Statsistics Methods   ###
    ###################################
    
    def Save_Sim_Stats(self):
        """ Save current simulation statistics """
        
        self.Sim_Stats.Record_Data(self.time[0],self.process_counters,self.event_counters,self.ni)




#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
        
##############################################
###   Cython Extension: KMC_Engine_Cy_3D   ###
##############################################



cdef class KMC_CyEngine_3D:
    """
    KMC_Engine_Cy 3D Class: Central control over KMC algorithm, including deposition and diffusion processes
    
    
    Attributes
    ----------
    
    
    Methods
    -------
    Update_Dep_Coordinates:     Get deposition data from python
    Deposit:                    Deposit a new particle
    Diffuse:                    Diffuse a particle
    
    """      
    
    cdef:
        np.int32_t  lx, ly, length, num_proc, solver_type, save_events, non_therm_process
        np.int64_t  atom_index, dep_index, diff_index, process_index
        
        np.int32_t [:,:,::1] lattice
        np.int32_t [:,::1] lattice2d
        np.int32_t [:,::1] surface
        np.int32_t [::1] surface2d
        np.double_t [::1] time
        
        np.int32_t [:,::1] coordinate_table
        np.int8_t [:,::1] process_table
        np.int8_t [::1] process_table_update
        np.int32_t [:,::1] move_table
        
        np.int64_t [::1] process_counters
        np.int64_t [::1] event_counters
        
        KMC_Solver.Solver Solver
        KMC_Lattice_2D.Lattice_2D Lattice2d
        KMC_Lattice_3D.Lattice_3D Lattice
#        KMC_Non_Therm.Non_Thermal Non_Thermal_Processes
        
        double [::1] dep_times
        np.int32_t [:,::1] dep_coordinates
        
        vector[np.int32_t] x_store
        vector[np.int32_t] y_store
        vector[np.int32_t] z_store
        vector[np.int32_t] m_store
        vector[double] t_store
        
        

    def __init__(self, np.int32_t  lx, np.int32_t  ly, np.int32_t  length, np.int32_t  num_proc, np.int32_t  solver_type, np.int32_t  save_events, np.int32_t  non_therm_process, np.int32_t [:,:,::1] lattice, np.int32_t [:,::1] surface, np.double_t [::1] time, np.int32_t [:,::1] coordinate_table, np.int8_t [:,::1] process_table, np.int8_t [::1] process_table_update, np.int32_t [:,::1] move_table, np.int64_t [::1] process_counters, np.int64_t [::1] event_counters, KMC_Solver.Solver Solver, KMC_Lattice_3D.Lattice_3D Lattice):
        # variables
        self.lx = lx
        self.ly = ly
        self.length = length
        self.num_proc = num_proc
        self.solver_type = solver_type
        self.save_events = save_events
        self.non_therm_process = non_therm_process
        
        # lattice memoryviews
        self.lattice = lattice
        self.surface = surface
        
        # times memoryview
        self.time = time      # current simulation times. [0] = simulation time, [1] = time to next dep, [2] = time to next diff event
        
        # data table memoryviews
        self.coordinate_table = coordinate_table
        self.process_table = process_table
        self.process_table_update = process_table_update
        self.move_table = move_table
        
        # counter memoryviews
        self.process_counters = process_counters
        self.event_counters = event_counters
        
        # Cython extension object
        self.Solver = Solver
        self.Lattice = Lattice
                
        # create new variables and initialize to zero
        self.atom_index = 2          # particle id for current depositing particle (for lattice value and indexing tables)
        self.dep_index = 0            # index for depositing particle in current pulse
        self.diff_index = 0           # index for particle u next diffusion process
        self.process_index = 0        # index of an elementary process


    
    
    ###############################
    ###   Return Data Methods   ###
    ###############################
    
    cpdef void Return_All_Events(self, np.int32_t[::1] X, np.int32_t[::1] Y, np.int32_t[::1] Z, np.int32_t[::1] M, double[::1] T):
        """ Return complete event data to numpy arrays and clear store vectors. """
        
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
        
        
        
    
    ################################
    ###   KMC Sequence Methods   ###
    ################################
      
    cpdef void Step(self):
        """ Kinetic Monte Carlo Step - Execute one step of the KMC algorithm """
        
        # decide to deposit or diffuse (dep_time < diff_time)
        if self.time[1] < self.time[2]:     
            self.Deposit()
        else:
            self.Diffuse()
    

            
    cpdef void Step_Anneal(self):
        """ Kinetic Monte Carlo Step - Execute one step of the KMC algorithm - Diffusion Only. """
        
        self.Diffuse()
        
        
        
    cpdef void Step_Deposit(self):
        """ Force a deposition. Used to start a new pulse in deposition simulation """
        
        self.Deposit()
        
        
        
    cpdef np.int32_t Run_until_Time(self, double end_time, np.int8_t anneal):
        """ Kinetic Monte Carlo Sequence - Run until a specific simulation time, but do not pass it.  """
        cdef np.int32_t error_check
        cdef double delta_end_time, delta_min
        
        
        # error check: make sure end_time > current simulation time
        if end_time > self.time[0]:
            error_check = 0
            
            # calc the delta times
            delta_end_time = end_time - self.time[0]
            delta_min = min(self.time[1],self.time[2])
                    
            # run loop
            if anneal == 0:
                while delta_end_time > delta_min:
                    self.Step()
                    delta_end_time = end_time - self.time[0]
                    delta_min = min(self.time[1],self.time[2])

            elif anneal == 1:
                while delta_end_time > self.time[2]:
                    self.Step_Anneal()
                    delta_end_time = end_time - self.time[0]
        
        # return an error code to Engine
        elif end_time <= self.time[0]:
            error_check = 1
            
        return error_check
    
    
    
    cpdef np.int32_t Run_until_Step(self, np.int64_t step, double end_time, np.int8_t anneal):
        """ Kinetic Monte Carlo Sequence - Run until a specific simulation step, but do not go past end of simulation time  """
        cdef np.int32_t error_check
        cdef double delta_end_time, delta_min
        
        # error check: make sure step > current simulation step
        if step > self.event_counters[0]:
            error_check = 0
            
            # calc the delta times
            delta_end_time = end_time - self.time[0]
            delta_min = min(self.time[1],self.time[2])
            
            # run loop
            if anneal == 0:
                while self.event_counters[0] < step and delta_end_time > delta_min:
                    self.Step()
                    delta_end_time = end_time - self.time[0]
                    delta_min = min(self.time[1],self.time[2])
                    
            elif anneal == 1:
                while self.event_counters[0] < step and delta_end_time > self.time[2]:
                    self.Step_Anneal()
                    delta_end_time = end_time - self.time[0]
        
        # return an error code to Engine
        elif step <= self.event_counters[0]:
            error_check = 1
            
        return error_check
            


            
    #################################
    ###   KMC Algorithm Methods   ###
    #################################
    
    cpdef void Update_Dep_Coordinates(self, double[::1] dep_times, np.int32_t[:,::1] dep_coordinates):
        """ Update Dep Coordinate Values at start of new pulse """
                
        self.dep_times = dep_times
        self.dep_coordinates = dep_coordinates
        self.dep_index = 0  # reset the deposition index
    
    
    
    cdef void Deposit(self):
        """ Deposition step """
        
        cdef np.int32_t test,x,y,z
        
        # Get Deposition Coordinates
        x = self.dep_coordinates[0,self.dep_index]   # get the deposition x coordinate
        y = self.dep_coordinates[1,self.dep_index]   # get the deposition y coordinate
        
        # Find a z coordinate to land (switch to top down calculation, concerned about voids )
        z = 0
        test = 1
        while test != 0:
            z += 1
            test = self.lattice[x,y,z]
            
        # Energetic or Non-Thermal Processes here
        if self.non_therm_process == 0:     # none enabled
            pass
        elif self.non_therm_process == 1:   # do the non-thermal processes
            
            # Only Downward Funneling
            pass

        
        
        
                   
        # Deposit particle into lattice
        self.lattice[x,y,z] = self.atom_index
        self.coordinate_table[self.atom_index,0] = x   # store x position
        self.coordinate_table[self.atom_index,1] = y   # store y position
        self.coordinate_table[self.atom_index,2] = z   # store z position
        
        # Search the Neighborhood and update Process Tables and Lists
        self.Lattice.Check_Neighborhood(x,y,z,x,y,z)
        
        # Advance simulation time
        self.time[0] = self.dep_times[self.dep_index]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.y_store.push_back(y)
            self.z_store.push_back(z)
            self.m_store.push_back(self.atom_index)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
            
        # Update Event Counters
        self.event_counters[0] += 1  # total events
        self.event_counters[1] += 1  # dep event
            
        # Update Deposition Indices
        self.atom_index += 1
        self.dep_index += 1
        
        # Calculate time to next dep event
        self.time[1] = self.dep_times[self.dep_index] - self.time[0]
        
        # Choose next thermal process
        self.Update_Next_Thermal_Process()
        
            
    
    cdef void Diffuse(self):
        """ Execute the diffusion move """
        
        # Note: Here diff_index is current index of particle selected to move
        
        cdef np.int32_t x,y,z,x_old,y_old,z_old
        
        # particles current positions
        x_old = self.coordinate_table[self.diff_index,0]
        y_old = self.coordinate_table[self.diff_index,1]
        z_old = self.coordinate_table[self.diff_index,2]
    
        # new position (Move Table Index: [(x,y,z),process])
        x = modulus(x_old+self.move_table[self.process_index,0],self.lx)
        y = modulus(y_old+self.move_table[self.process_index,1],self.ly)
        z = z_old + self.move_table[self.process_index,2]
        
        # delete id from old position and add to new position
        self.lattice[x_old,y_old,z_old] = 0
        self.lattice[x,y,z] = self.diff_index
        
        self.coordinate_table[self.diff_index,0] = x
        self.coordinate_table[self.diff_index,1] = y
        self.coordinate_table[self.diff_index,2] = z
    
        # Search the Neighborhood and update Process Tables and Lists
        self.Lattice.Check_Neighborhood(x,y,z,x_old,y_old,z_old)
        
        # Advance simulation time (add t_diff to sim time)
        self.time[0] += self.time[2]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.y_store.push_back(y)
            self.z_store.push_back(z)
            self.m_store.push_back(self.diff_index)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
            
        # Update Event Counters
        self.event_counters[0] += 1  # total events
        self.event_counters[2] += 1  # diffuse event
        
        # Update the Process Counter for selected process
        self.process_counters[self.process_index] += 1
        
        # Calculate time to next dep event
        self.time[1] = self.dep_times[self.dep_index] - self.time[0]
        
        # Choose next thermal process
        self.Update_Next_Thermal_Process()
    
    
    
    cdef void Update_Next_Thermal_Process(self):
        """ Figure out the next thermal process  """
        
        # Calculate partial rates
        self.Solver.Calc_Partial_Rates()
             
        # Calculate cummulative rates        
        self.Solver.Calc_Cummulative_Rates()          
        
        # Calculate time to next diffusion event
        self.time[2] = self.Solver.Calc_tdiff()
        
        # Select a process
        self.process_index = self.Solver.Select_Process()
        
        # Select a particle
        if self.solver_type == 0 or self.solver_type == 1:
            self.diff_index = self.Solver.Select_Particle(self.process_index)
        elif self.solver_type == 2:
            self.diff_index = self.Solver.Select_Particle_Set(self.process_index)

    





#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
###########################################
###   Cython Extension: KMC_Engine_2D   ###
###########################################


cdef class KMC_CyEngine_2D(KMC_CyEngine_3D):
    """
    KMC_Engine_2D Class: Child Class of KMC_Engine_Cy_3D. Inherits from KMC_Engine_Cy_3D and overrides dimension specific methods with 2D versions.
      
   
    Overriden Methods
    -----------------
   
    """
        
    

    def __init__(self, np.int32_t  lx, np.int32_t  length, np.int32_t  num_proc, np.int32_t  solver_type, np.int32_t  save_events, np.int32_t  non_therm_process, np.int32_t [:,::1] lattice2d, np.int32_t [::1] surface2d, np.double_t [::1] time, np.int32_t [:,::1] coordinate_table, np.int8_t [:,::1] process_table, np.int8_t [::1] process_table_update, np.int32_t [:,::1] move_table, np.int64_t [::1] process_counters, np.int64_t [::1] event_counters, KMC_Solver.Solver Solver, KMC_Lattice_2D.Lattice_2D Lattice2d):
        # variables
        self.lx = lx
        self.length = length
        self.num_proc = num_proc
        self.solver_type = solver_type
        self.save_events = save_events
        self.non_therm_process = non_therm_process
        
        # lattice memoryviews
        self.lattice2d = lattice2d
        self.surface2d = surface2d
        
        # times memoryview
        self.time = time      # current simulation times. [0] = simulation time, [1] = time to next dep, [2] = time to next diff event
        
        # data table memoryviews
        self.coordinate_table = coordinate_table
        self.process_table = process_table
        self.process_table_update = process_table_update
        self.move_table = move_table
        
        # counter memoryviews
        self.process_counters = process_counters
        self.event_counters = event_counters
        
        # Cython extension object
        self.Solver = Solver
        self.Lattice2d = Lattice2d
                
        # create new variables and initialize to zero
        self.atom_index = 2          # particle id for current depositing particle (for lattice value and indexing tables)
        self.dep_index = 0            # index for depositing particle in current pulse
        self.diff_index = 0           # index for particle u next diffusion process
        self.process_index = 0        # index of an elementary process
        
        
        
    # Overriden Methods
    cpdef void Return_All_Events(self, np.int32_t[::1] X, np.int32_t[::1] Y, np.int32_t[::1] Z, np.int32_t[::1] M, double[::1] T):
        """ Return complete event data to numpy arrays and clear store vectors. """
        
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
        
    
    
    cpdef void Update_Dep_Coordinates(self, double[::1] dep_times, np.int32_t[:,::1] dep_coordinates):
        """ Update Dep Coordinate Values at start of new pulse """
                
        self.dep_times = dep_times
        self.dep_coordinates = dep_coordinates
        self.dep_index = 0  # reset the deposition index
    
    
    
    cdef void Deposit(self):
        """ Deposition step """
        
        cdef np.int32_t test,x,z
        
        # Get Deposition Coordinates
        x = self.dep_coordinates[0,self.dep_index]   # get the deposition x coordinate
        
        # Find a z coordinate to land (switch to top down calculation, concerned about voids )
        z = 0
        test = 1
        while test != 0:
            z += 1
            test = self.lattice2d[x,z]
            
        # Energetic Processes here
        if self.non_therm_process == 0:     # none enabled
            pass
        elif self.non_therm_process == 1:   # downward funneling
            pass
                   
        # Deposit particle into lattice
        self.lattice2d[x,z] = self.atom_index
        self.coordinate_table[self.atom_index,0] = x   # store x position
        self.coordinate_table[self.atom_index,1] = z   # store z position
        
        # Search the Neighborhood and update Process Tables and Lists
        self.Lattice2d.Check_Neighborhood(x,z,x,z)
        
        # Advance simulation time
        self.time[0] = self.dep_times[self.dep_index]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.z_store.push_back(z)
            self.m_store.push_back(self.atom_index)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
            
        # Update Event Counters
        self.event_counters[0] += 1  # total events
        self.event_counters[1] += 1  # dep event
            
        # Update Deposition Indices
        self.atom_index += 1
        self.dep_index += 1
        
        # Calculate time to next dep event
        self.time[1] = self.dep_times[self.dep_index] - self.time[0]
        
        # Choose next thermal process
        self.Update_Next_Thermal_Process()
        
            
    
    cdef void Diffuse(self):
        """ Execute the diffusion move """
        
        # Note: Here diff_index is current index of particle selected to move
        
        cdef np.int32_t x,z,x_old,z_old
        
        # particles current positions
        x_old = self.coordinate_table[self.diff_index,0]
        z_old = self.coordinate_table[self.diff_index,1]
    
        # new position (Move Table Index: [(x,y,z),process])
        x = modulus(x_old+self.move_table[self.process_index,0],self.lx)
        z = z_old + self.move_table[self.process_index,1]
        
        # delete id from old position and add to new position
        self.lattice2d[x_old,z_old] = 0
        self.lattice2d[x,z] = self.diff_index
        
        self.coordinate_table[self.diff_index,0] = x
        self.coordinate_table[self.diff_index,1] = z
    
        # Search the Neighborhood and update Process Tables and Lists
        self.Lattice2d.Check_Neighborhood(x,z,x_old,z_old)
        
        # Advance simulation time (add t_diff to sim time)
        self.time[0] += self.time[2]
        
        # Record Event if saving all events
        if self.save_events == 1:
            self.x_store.push_back(x)
            self.z_store.push_back(z)
            self.m_store.push_back(self.diff_index)
            self.t_store.push_back(self.time[0])
            self.event_counters[3] += 1
            
        # Update Event Counters
        self.event_counters[0] += 1  # total events
        self.event_counters[2] += 1  # diffuse event
        
        # Update the Process Counter for selected process
        self.process_counters[self.process_index] += 1
        
        # Calculate time to next dep event
        self.time[1] = self.dep_times[self.dep_index] - self.time[0]
        
        # Choose next thermal process
        self.Update_Next_Thermal_Process()








# End of Module