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

Low-level Python Class containing all Python Data Structures. Controls single step in the KMC algoithm. Contains 
code to decide between deposition and diffusion processes. Saves event data and simulation statistics.

Owns the Cython classes: Solver and Processes_3D

Version: 3.3b

"""



# import python modules
import numpy as np
import time

# cimport cython modules
cimport numpy as np
cimport cython

# KMC modules
import KMC_Miscellaneous_v3p3 as KMC_Misc
cimport KMC_Solver_v3p3 as KMC_Solver
cimport KMC_Processes_2D_v3p3 as KMC_Proc_2D
cimport KMC_Processes_3D_v3p3 as KMC_Proc_3D




####################################
###   Python Class: KMC_Engine   ###
####################################
#--------------------------------------------------------------------------------------------------------------------------------------------------


class KMC_Engine:
    """
    KMC_Engine Base Class: Contains data structures for simulation and controls step sequences
    
    
    Attributes
    ----------
    lattice:               Array storing particle postions and id
    coordinate_table:      Stores coordinates for each deposited particle
    move_table:            Array of moves for moving particle throughout Lattice    
    process_table:         Array of current available process for each particle
    process_table_update:  New status for selected particle
    
    rate_catalog:          Dictionary of elementary process rates
    rates:                 Array of elementary process rates (in order of process lists)
    moves:                 Array of moves
    keys:                  List of names of each elementary process
        
    ni:                    Array of process list sizes
    ri:                    Partical sum of rates for each process list
    total_rate:            Total rate of all processes
    cummualitve_rates:     Array of cummulative partial rates
    
    Sim_Stats:             Statistics object
    process_counters:      Keeps track of processes selected
    event_counters:        Kepps track of number of events
    
    
    Cython Classes
    --------------
    Solver:  KMC solver algorithm
    
    
    Methods
    -------
    Initialize:             Create all Data Structures
    Specific_Init:          Initialize dimmension specific attributes
    Get_Cummulative_Rates:  Get cummulative rate data from Solver vector.
    Get_Process_Lists:      Get process lists from Solver vector.
    First_Step:             Execute first deposition event
    Step:                   Exuecute deposition or diffusion event
    Step_Anneal:            Execute diffusion event
    Save_Events:            Add event coordinates to save data list
    
    """


    
    def __init__(self,Parameters):
        self.Parameters = Parameters
        
        # extract useful parameters from dictionary
        self.save = self.Parameters['Save_Output_File']      # save all events
        self.path = self.Parameters['Save_Path_Simulation']
        
        self.lx = self.Parameters['Lx']          # length of substrate
        self.ly = self.Parameters['Ly']          # length of substrate
        self.n = self.Parameters['n']            # number of particles deposited per pulse
        self.depth = self.Parameters['depth']    # number of incomplete layers (z coordinate)
        self.pulses = self.Parameters['Pulses']  # number of pulses
        
        self.downward = self.Parameters['Downward_Funneling']    # Enable Downward Funneling Process
        
        # Solver Type: Vector_Linear = 0, Vector_Binary = 1, Set = 2
        if self.Parameters['Solver_Type'] == 'Linear':
            self.solver_type = 0
        elif self.Parameters['Solver_Type'] == 'Binary':
            self.solver_type = 1
        elif self.Parameters['Solver_Type'] == 'Set':
            self.solver_type = 2
        
        self.Initialize()
               

        
    def Initialize(self):
        """ Initialize the KMC_Engine. """
        
        # set up variables
        self.atom_index = np.array([0],dtype=np.int32)   # index of current particle to be deposited (2 less than partice ID in Lattice)
        self.dep_index = np.array([0],dtype=np.int32)    # index of current particle in pulse (for selecting dep times, coordinates)

        # save all events?
        if self.save == True:
            self.save_events = 1
        elif self.save == False: 
            self.save_events = 0
        
        # simulation times
        self.time = np.array([0.0],dtype=np.double)        # simulation time, (array to pass as memoryview to Cython Classes)
        self.time_diff = np.array([0.0],dtype=np.double)   # time to next diffusion event (array to pass as memoryview to Cython Classes)
        self.time_dep = np.array([0.0],dtype=np.double)    # time to next deposition event (array to pass as memoryview to Cython Classes)
        
        # Create the Rate and Process Catalogs and Arrays
        self.rate_catalog, self.rates, self.move_table, self.keys, self.num_proc = KMC_Misc.Process_Catalog(self.Parameters).Return_Rates()
        self.length = self.rates.shape[0]       # length of the rates array
        
        # Create the Rate Tables    
        self.ni = np.zeros([self.length],dtype=np.int32)    # Total moves possible per process
        self.ri = np.zeros([self.length],dtype=np.double)   # Partial rates for each elementary process        
        self.total_rate = np.array([0.0],dtype=np.double)   # Total Rate of all Processes (array to pass as memoryview to Cython Classes)
        
        # Debug Data strucutres (These aren't used in the KMC algorithm, they are here for the user)
        self.cummulative_rates = np.zeros([self.length],dtype=np.double)    # Cummulative partial rates
        self.max_ni = np.zeros([self.length],dtype=np.int32)                # Keep track of maximum number of particles in each list
        
        # Create Process Tables
        self.process_table = np.zeros([self.n*self.pulses,self.length],dtype=np.int8)  # Array to store available processes for each particle
        self.process_table_update = np.zeros([self.length],dtype=np.int8)              # Array to test for change in process status
               
        # Initialize Simulation Statistics
        self.Sim_Stats = KMC_Misc.Statistics(self.Parameters['Enable_Plots'], self.path, self.keys, self.num_proc)
        self.process_counters = np.zeros([self.length],dtype=np.int64)   # array to store selected processes
        self.event_counters = np.zeros([4],dtype=np.int64)               # array to store event counts: Total, Dep, Diffuse, Event Vector Size
            
        # Create Instance of Solver (this will be passed to Processes)
        self.seed = int(1000*time.perf_counter())     # random seed from clock time
        self.Solver = KMC_Solver.Solver(self.solver_type, self.rates, self.length, self.time, self.time_diff, self.ni, self.ri, self.total_rate, self.max_ni, self.seed)
        
        # Now call dimension specific methods of child class
        self.Specific_Init()
        


    # Methods of child classes
    def Specific_Init(self):
        pass
    
    def Save_Events(self):
        pass
        
    def Add_Substrate(self):
        pass
    
    def Activate_Substrate(self):
        pass
    

    
    def Get_Cummulative_Rates(self):
        """ Extract cummulative rate data from Solver vector """
        self.Solver.Return_Cummulative_Rates(self.cummulative_rates)
        

        
    def Get_Process_Lists(self):
        """ Extract process list data from Solver vector """
        List = [[] for _ in range(self.length)]
        self.process_lists = self.Solver.Return_Process_Lists(List)
        
        
    
    ##########################
    ###   Sequence Steps   ###
    ##########################
        
    def First_Step(self, dep_times, dep_coordinates, end_of_pulse):
        """ Kinetic Monte Carlo Step - Initial Deposition to get things going """          
        self.dep_times = dep_times               # set up dep times for full pulse
        self.dep_coordinates = dep_coordinates   # Coordinates for deposition
        self.end_of_pulse = end_of_pulse                  # set the end of pulse time
        
        self.Process.Update_Dep_Coordinates(self.dep_times,self.dep_coordinates)  # first time we have these
        
        self.dep_index[0] = 0    # reset deposition index counter
                
        # deposit 1st particle
        self.Process.Deposit()
        
        return self.time[0]
    

      
    def Step(self):
        """ Kinetic Monte Carlo Step - Deposition and Diffusion Sequence """          
        # decide to deposit or diffuse
        if self.time_dep[0] < self.time_diff[0]:     
            self.Process.Deposit()
        else:
            self.Process.Diffuse()
                    
        return self.time[0]
    

            
    def Step_Anneal(self):
        """ Kinetic Monte Carlo Step - Diffusion Only Sequence """        
        # check if pulse ends before next event and if so break out of loop
        if self.end_of_pulse - self.time[0] > self.time_diff[0]:
            self.Process.Diffuse()
        else:
            # return end of pulse time
            self.time[0] = self.end_of_pulse
        
        return self.time[0]
    
    
    
    def Step_Post_Anneal_Start(self,final_time):
        """ Post Anneal - set final time """
        self.end_of_pulse = final_time
    
    
    
    



#--------------------------------------------------------------------------------------------------------------------------------------------------    
######################################
###   Python Class: KMC_Engine_2D  ###
######################################



class KMC_Engine_2D(KMC_Engine):
    """
    KMC_Engine child class: for 2D lattices and simulations
    
    
    Specific Attributes
    -------------------
    lattice:          2D Lattice (int32)
    coordinate_table: stores x,y,z position of all atoms
    
        
    Cython Objects
    --------------
    Processes_2D: Cython Extension Type
    
    
    Methods
    -------
    Specific_init:  Create Lattice Data Structure
    Save_events:
    Save_lattice: Process the Lattice for saving
    Add_Substrate: Create substrate features on top of base layer
    Activate_Substrate: Make substrate particles into live particles
    Save_events: Get list of event data from Cython vectors
    Save_lattice: Get lattice data (this is now obsolete)
    
    """


       
    def Specific_Init(self):
        """ Initialize dimension specific attributes """
        
        # Create the Lattice
        self.lattice = np.zeros([self.lx,self.depth],dtype=np.int32)
                
        # Create the Coordinate Table (current position of all particles in Lattice)
        self.coordinate_table = np.zeros([self.n*self.pulses,2],dtype=np.int32)
        
        # Add the substrate
        self.Add_Substrate()                
        
        # Create Downward Funneling Moves Table (not sure where this will end up)
        
        # Create instance of Processes_2D (also pass instance of Solver)                
        self.Process = KMC_Proc_2D.Processes_2D(self.lx,self.length,self.num_proc,self.save_events,self.solver_type,self.time,self.time_dep,self.time_diff,
                                    self.atom_index,self.dep_index,self.lattice,self.coordinate_table,self.move_table,self.process_table,
                                    self.process_table_update,self.process_counters,self.event_counters,self.Solver)
                            
        # Activate the substrate particles if 'Active'
        if self.Parameters['Substrate_Type'] != 'Flat':
            if self.Parameters['Substrate_Particle_State'] == 'Active':
                self.Activate_Substrate()
            

    
    def Add_Substrate(self):
        """ Set-up a starting substrate.
        
        This function does the following:
            
        Set up the default flat substrate. This is 2 layers for a 2D simulation.
        Call Substrate Generator and request list substrate particle positions and ids
        Place all substrate particles
        For active substrates, call Proceeses_3D.Check_Neighborhood to set up process table
        
        """
        
        # Set up the Default Starting Substrate (this is always set)
        self.lattice[:,0:2] = 1  # these are alway passive particles
        
        # variable for setting first deposited particle id        
        if self.Parameters['Substrate_Type'] == 'Flat':
            pass
        elif self.Parameters['Substrate_Type'] == 'Islands' or self.Parameters['Substrate_Type'] == 'Steps':
            # Create instance of Substrate Generator
            self.Substrate = KMC_Misc.Substrate_Generator(self.Parameters)

            # Generate a substrate (substrate_count is 1 higher than last substrate particle)
            self.substrate_list,self.substrate_count = self.Substrate.Generate_Substrate()
            
            # If Passive, just place atoms in the lattice  
            if self.Parameters['Substrate_Particle_State'] == 'Passive':
                for i,k,m in self.substrate_list:
                    self.lattice[i,k] = m
                    
            # If Active substrate, expand the Process Table and Coordinate table to make room for substrate particles
            if self.Parameters['Substrate_Particle_State'] == 'Active':
                self.process_table = np.append(self.process_table,np.zeros([self.substrate_count-2,self.length],dtype=np.int8),axis=0)
                self.coordinate_table = np.append(self.coordinate_table,np.zeros([self.substrate_count-2,2],dtype=np.int32),axis=0)
                
                
                
    def Activate_Substrate(self):
        """ Activate the substrate particles. """
        
        # Convert substrate list to Dep_Coordinates
        dep_times = np.zeros([len(self.substrate_list)])
        dep_coordinates = np.zeros([len(self.substrate_list)],dtype=np.int32)
        for i in range(len(self.substrate_list)):
            dep_coordinates[i] = self.substrate_list[i][0]
                
        # Send Dep Coordinates to Processes_2D
        self.Process.Update_Dep_Coordinates(dep_times,dep_coordinates)
        
        # Pre-deposit substrate particles
        for i in range(len(self.substrate_list)):            
            self.Process.Deposit()
                    


    def Save_Events(self):
        """ extract event data for save file. """
        
        # length of event data vectors
        length = self.event_counters[3]
        
        # make numpy arrays
        x_store = np.zeros([length],dtype=np.int32) 
        z_store = np.zeros([length],dtype=np.int32)
        m_store = np.zeros([length],dtype=np.int32)
        t_store = np.zeros([length],dtype=np.double)
        
        # extract the data
        self.Process.Return_Events(x_store,z_store,m_store,t_store)
        
        return (x_store,z_store,m_store,t_store)

        
           
            




#--------------------------------------------------------------------------------------------------------------------------------------------------
######################################
###   Python Class: KMC_Engine_3D  ###
######################################
        
    

class KMC_Engine_3D(KMC_Engine):
    """
    KMC_Engine child class: for 3D lattices and simulations
    
    
    Specific Attributes
    -------------------
    lattice:          3D Lattice (int32)
    coordinate_table: stores x,y,z position of all atoms
    
        
    Cython Objects
    --------------
    Processes_3D: Cython Extension Type
    
    
    Methods
    -------
    Specific_init:  Create Lattice Data Structure
    Save_events:
    Save_lattice: Process the Lattice for saving
    Add_Substrate: Create substrate features on top of base layer
    Activate_Substrate: Make substrate particles into live particles
    Save_events: Get list of event data from Cython vectors
    Save_lattice: Get lattice data (this is now obsolete)
    
    """


       
    def Specific_Init(self):
        """ Initialize dimension specific attributes """
        
        # Create the Lattice
        self.lattice = np.zeros([self.lx,self.ly,self.depth],dtype=np.int32)
                
        # Create the Coordinate Table (current position of all particles in Lattice)
        self.coordinate_table = np.zeros([self.n*self.pulses,3],dtype=np.int32)
        
        # Add the substrate
        self.Add_Substrate()                
        
        # Create Downward Funneling Moves Table (not sure where this will end up)
        self.df_moves = np.array([[0,1],[1,0],[0,-1],[-1,0]])
        
        # Create instance of Processes_3D (also pass instance of Solver)        
        if self.Parameters['Enable_Processes'] == 'All':
            self.Process = KMC_Proc_3D.Processes_3D(self.lx,self.ly,self.length,self.num_proc,self.save_events,self.solver_type,self.time,self.time_dep,self.time_diff,
                                        self.atom_index,self.dep_index,self.lattice,self.coordinate_table,self.move_table,self.process_table,
                                        self.process_table_update,self.process_counters,self.event_counters,self.Solver)
            
        elif self.Parameters['Enable_Processes'] == 'No_Uphill_4NN':
            self.Process = KMC_Proc_3D.Processes_3D_A(self.lx,self.ly,self.length,self.num_proc,self.save_events,self.solver_type,self.time,self.time_dep,self.time_diff,
                                        self.atom_index,self.dep_index,self.lattice,self.coordinate_table,self.move_table,self.process_table,
                                        self.process_table_update,self.process_counters,self.event_counters,self.Solver)
            
        elif self.Parameters['Enable_Processes'] == 'No_Detach':
            self.Process = KMC_Proc_3D.Processes_3D_B(self.lx,self.ly,self.length,self.num_proc,self.save_events,self.solver_type,self.time,self.time_dep,self.time_diff,
                                        self.atom_index,self.dep_index,self.lattice,self.coordinate_table,self.move_table,self.process_table,
                                        self.process_table_update,self.process_counters,self.event_counters,self.Solver)
            
        elif self.Parameters['Enable_Processes'] == 'No_Edge':
            self.Process = KMC_Proc_3D.Processes_3D_C(self.lx,self.ly,self.length,self.num_proc,self.save_events,self.solver_type,self.time,self.time_dep,self.time_diff,
                                        self.atom_index,self.dep_index,self.lattice,self.coordinate_table,self.move_table,self.process_table,
                                        self.process_table_update,self.process_counters,self.event_counters,self.Solver)
            
        elif self.Parameters['Enable_Processes'] == 'No_Detach_or_Edge':
            self.Process = KMC_Proc_3D.Processes_3D_D(self.lx,self.ly,self.length,self.num_proc,self.save_events,self.solver_type,self.time,self.time_dep,self.time_diff,
                                        self.atom_index,self.dep_index,self.lattice,self.coordinate_table,self.move_table,self.process_table,
                                        self.process_table_update,self.process_counters,self.event_counters,self.Solver)
                    
        # Activate the substrate particles if 'Active'
        if self.Parameters['Substrate_Type'] != 'Flat':
            if self.Parameters['Substrate_Particle_State'] == 'Active':
                self.Activate_Substrate()
            

    
    def Add_Substrate(self):
        """ Set-up a starting substrate.
        
        This function does the following:
            
        Set up the default flat substrate
        Call Substrate Generator and request list substrate particle positions and ids
        Place all substrate particles
        For active substrates, call Proceeses_3D.Check_Neighborhood to set up process table
        
        """
        
        # Set up the Default Starting Substrate (this is always set)
        self.lattice[:,:,0] = 1  # these are alway passive particles
        
        # variable for setting first deposited particle id        
        if self.Parameters['Substrate_Type'] == 'Flat':
            pass
        elif self.Parameters['Substrate_Type'] == 'Islands' or self.Parameters['Substrate_Type'] == 'Steps':
            # Create instance of Substrate Generator
            self.Substrate = KMC_Misc.Substrate_Generator(self.Parameters)

            # Generate a substrate (substrate_count is 1 higher than last substrate particle)
            self.substrate_list,self.substrate_count = self.Substrate.Generate_Substrate()
            
            # If Passive, just place atoms in the lattice  
            if self.Parameters['Substrate_Particle_State'] == 'Passive':
                for i,j,k,m in self.substrate_list:
                    self.lattice[i,j,k] = m
                    
            # If Active substrate, expand the Process Table and Coordinate table to make room for substrate particles
            if self.Parameters['Substrate_Particle_State'] == 'Active':
                self.process_table = np.append(self.process_table,np.zeros([self.substrate_count-2,self.length],dtype=np.int8),axis=0)
                self.coordinate_table = np.append(self.coordinate_table,np.zeros([self.substrate_count-2,3],dtype=np.int32),axis=0)
                
                
                
    def Activate_Substrate(self):
        """ Activate the substrate particles. """
        
        # Convert substrate list to Dep_Coordinates
        dep_times = np.zeros([len(self.substrate_list)])
        dep_coordinates = np.zeros([2,len(self.substrate_list)],dtype=np.int32)
        for i in range(len(self.substrate_list)):
            dep_coordinates[0,i] = self.substrate_list[i][0]
            dep_coordinates[1,i] = self.substrate_list[i][1]
                
        # Send Dep Coordinates to Processes_3D
        self.Process.Update_Dep_Coordinates(dep_times,dep_coordinates)
        
        # Pre-deposit substrate particles
        for i in range(len(self.substrate_list)):            
            self.Process.Deposit()
                    


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
        self.Process.Return_Events(x_store,y_store,z_store,m_store,t_store)
        
        return (x_store,y_store,z_store,m_store,t_store)

        
           
            

            
            


    
# End of Module