####################################################################################################################
####################################################################################################################
#########################                                                                  #########################
#########################            PLD Kinetic Monte Carlo Processes Module              #########################
#########################                                                                  #########################
####################################################################################################################
####################################################################################################################


"""
Kinetic Monte Carlo Processes Module

Contains code to create the rate catalog for thermal processes

Version: 3.4.1a

"""

# import modules
import numpy as np
from scipy.constants import physical_constants, convert_temperature




#################################
###   Process Catalog Class   ###
#################################
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



class Process_Catalog:
    """
    Class to set up elementary processes and rates. Sets the order of Process Table and Process Lists.
    
    This class sets the order in the process table and corresponding moves.
    
    Future Goal is to be able to set processes here and have the main code automatically adapt. Not there yet.
    If changes are made here, the corresponding change must be made in corresponding "Processes" class.
    
    
    Inputs
    -------
    parameters:  activation energies and substrate temperature
    
    
    Methods:
    --------
    Create_Catalog:   Selects the process catalog based on parameter inputs
    Return_Rates:     Returns the rate_catalog, rates, moves, and keys
    Deposition_2D:
    Deposition_3D:
    
        
    Outputs
    -------
    rate_catalog:  Dictionary of Rate Names and Values
    rates:         Ordered Array of all elementary process rates
    moves:         Associated lattice moves that go with rates.
    keys:          Ordered list of keys corresponding to the rates array
    numProc:       Number of Processes for each direction (Primary and Secondary Moves)
    
    
    """

    def __init__(self,Parameters):
        self.Parameters = Parameters

        self.dim = Parameters['Dimension']
        self.type = Parameters['Simulation_Type']
        
        self.w0 = Parameters['w0']
        self.temperature = self.Parameters['Substrate_Temperature']
        
        self.diff = Parameters['Ea_diffusion']
        self.es = Parameters['Ea_ehrlich_schwoebel']
        self.detach = Parameters['Ea_detach']   
        self.edge = Parameters['Ea_edge']
        self.corner = Parameters['Ea_corner']
        
        # constants
        self.tk = convert_temperature(self.temperature,'Celsius', 'Kelvin')
        self.kB = physical_constants[ 'Boltzmann constant in eV/K' ][0]        
        self.kT = self.tk*self.kB  # in eV
        
        self.Create_Catalog()
        

        
    def Create_Catalog(self):
        """ Determine which catalog to create and create it """
        if self.type == 'Deposition' or self.type == 'Diffusion' or self.type == 'Density':        
            if self.dim == '2D':
                self.Simulation_2D()
            elif self.dim == '3D':
                self.Simulation_3D()
        elif self.type == 'Ising':
            self.Ising_2D()
            

                
    def Return_Rates(self):
        """ Return the calculated rates """
        
        return (self.rate_catalog,self.rates,self.moves,self.keys,self.num_barriers_per_move)    
        

        
    def Simulation_2D(self):
        """ Set up Process Table for 2D Deposition (1+1) """
        
        # Set Primary and Secondary Moves (order based on first secondary move being to the left of the primary move)
        Moves = ['Left','Left_Down','Left_Up',\
                 'Up','Up_Left','Up_Right',\
                 'Right','Right_Up','Right_Down',\
                 'Down','Down_Right','Down_Left']        
        Moves_X = [-1,-1,-1,\
                   0,-1,1,\
                   1,1,1,\
                   0,1,-1]        
        Moves_Z = [0,-1,1,\
                   1,1,1,\
                   0,1,-1,\
                   -1,-1,-1]
        
        # Set Barriers for Primary and Secondary Moves (name should include each barrier present)
        Primary = ['Diff','Diff_Detach1','Diff_Detach2','Edge2','Edge2_Detach1']
        Secondary = ['Diff_ES', 'Diff_Detach1_ES', 'Diff_Detach2_ES']
        
        Barriers = [Primary,Secondary,Secondary,Primary,Secondary,Secondary,Primary,Secondary,Secondary,Primary,Secondary,Secondary]
        
        self.num_barriers_per_move = len(Primary)+len(Secondary)+len(Secondary) # number of processes per primary move
    
        # Create Rate Catalog (keys must match the Barriers)
        self.rate_catalog = {}
        self.rate_catalog['Diff'] = self.w0*np.exp(-self.diff/self.kT)   # free edge diffusion
        self.rate_catalog['Diff_Detach1'] = self.w0*np.exp(-(self.detach+self.diff)/self.kT)  # detach from kink
        self.rate_catalog['Diff_Detach2'] = self.w0*np.exp(-(2*self.detach+self.diff)/self.kT)  # detach from edge
        self.rate_catalog['Edge2'] = self.w0*np.exp(-(self.edge+self.diff)/self.kT)  # two edge diffusion
        self.rate_catalog['Edge2_Detach1'] = self.w0*np.exp(-(self.detach+self.edge+self.diff)/self.kT)  # two edge diffusion + detach 
        self.rate_catalog['Diff_ES'] = self.w0*np.exp(-(self.diff+self.es)/self.kT)   # erhlich-schwoebel barrier
        self.rate_catalog['Diff_Detach1_ES'] = self.w0*np.exp(-(self.detach+self.diff+self.es)/self.kT)   # erhlich-schwoebel barrier plus break 1 bond
        self.rate_catalog['Diff_Detach2_ES'] = self.w0*np.exp(-(2*self.detach+self.diff+self.es)/self.kT)   # erhlich-schwoebel barrier plus break 2 bonds
        
        # Create corresponding Process Table, Rate Table, and Move Table to send to solver
        self.rates = []  # Rate Table
        self.moves = []   # Move Table
        self.keys = []  # Table of Keys corresponding to each entry (Move_Barriers)
        
        for i in range(len(Moves)):
            for j in range(len(Barriers[i])):
                self.rates.append(self.rate_catalog[Barriers[i][j]])
                self.moves.append([Moves_X[i],Moves_Z[i]])
                self.keys.append((Moves[i]+'_'+Barriers[i][j]))
                
        self.rates = np.array(self.rates,dtype=np.double)
        self.moves = np.array(self.moves,dtype=np.int32)
                    

    
    def Simulation_3D(self):
        """ Set up Process Table for 3D Deposition (2+1) """
        
        # Set order of moves in the process table (straight,straight-down,straight-up,straight-left,straight-right)   
        Moves = ['North','North_Down','North_Up','North_West','North_East',\
                 'East','East_Down','East_Up','East_North','East_South',\
                 'South','South_Down','South_Up','South_East','South_West',\
                 'West','West_Down','West_Up','West_South','West_North']
        
        # set corresponding  coordinate moves
        Moves_X = [0,0,0,-1,1,\
                   1,1,1,1,1,\
                   0,0,0,1,-1,\
                   -1,-1,-1,-1,-1]    
        Moves_Y = [1,1,1,1,1,\
                   0,0,0,1,-1,\
                   -1,-1,-1,-1,-1,\
                   0,0,0,-1,1]
        Moves_Z = [0,-1,1,0,0,\
                   0,-1,1,0,0,\
                   0,-1,1,0,0,\
                   0,-1,1,0,0]
        
        # Set Barriers for Primary and Secondary Moves (name should include each barrier present)
        Barrier_Primary = ['Diff','Diff_Detach1','Diff_Detach2','Diff_Detach3','Edge','Edge_Detach1','Edge_Detach2']
        Barrier_Down = ['Diff_ES','Diff_Detach1_ES','Diff_Detach2_ES','Diff_Detach3_ES']
        Barrier_Up = ['Diff_Detach1_ES','Diff_Detach2_ES','Diff_Detach3_ES','Diff_Detach4_ES']
        Barrier_Corner = ['Corner','Corner_Detach1','Corner_Detach2']
        
        # Set Barriers for each move (can have multiple barriers for each move)
        Barriers = [Barrier_Primary,Barrier_Down,Barrier_Up,Barrier_Corner,Barrier_Corner,\
                    Barrier_Primary,Barrier_Down,Barrier_Up,Barrier_Corner,Barrier_Corner,\
                    Barrier_Primary,Barrier_Down,Barrier_Up,Barrier_Corner,Barrier_Corner,\
                    Barrier_Primary,Barrier_Down,Barrier_Up,Barrier_Corner,Barrier_Corner,]
        
        # number of processes per primary move
        self.num_barriers_per_move = len(Barrier_Down)+len(Barrier_Up)+len(Barrier_Primary)+len(Barrier_Corner)+len(Barrier_Corner)
        
        # Create Rate Catalog (keys must match the Barriers)
        self.rate_catalog = {}
        self.rate_catalog['Diff'] = self.w0*np.exp(-self.diff/self.kT)   # free surface diffusion
        self.rate_catalog['Diff_Detach1'] = self.w0*np.exp(-(self.detach+self.diff)/self.kT)  # detach from island, break 1 bond
        self.rate_catalog['Diff_Detach2'] = self.w0*np.exp(-(2*self.detach+self.diff)/self.kT)  # detach from island, break 2 bonds
        self.rate_catalog['Diff_Detach3'] = self.w0*np.exp(-(3*self.detach+self.diff)/self.kT)  # detach from island, break 3 bonds
        self.rate_catalog['Diff_ES'] = self.w0*np.exp(-(self.diff+self.es)/self.kT)   # downhill barrier
        self.rate_catalog['Diff_Detach1_ES'] = self.w0*np.exp(-(self.detach+self.diff+self.es)/self.kT)   # detach, break 1 bond, down
        self.rate_catalog['Diff_Detach2_ES'] = self.w0*np.exp(-(2*self.detach+self.diff+self.es)/self.kT)   # detach, break 2 bonds, up/down
        self.rate_catalog['Diff_Detach3_ES'] = self.w0*np.exp(-(3*self.detach+self.diff+self.es)/self.kT)   # detach, break 3 bonds, up/down
        self.rate_catalog['Diff_Detach4_ES'] = self.w0*np.exp(-(4*self.detach+self.diff+self.es)/self.kT)   # detach, break 4 bonds, up only
        self.rate_catalog['Edge'] = self.w0*np.exp(-(self.diff+self.edge)/self.kT)   # free edge diffusion
        self.rate_catalog['Edge_Detach1'] = self.w0*np.exp(-(self.diff+self.edge+self.detach)/self.kT)  # detach from kink plus edge diffusion, break 1 bond
        self.rate_catalog['Edge_Detach2'] = self.w0*np.exp(-(self.diff+self.edge+2*self.detach)/self.kT)  # detach from kink plus edge diffusion, break 2 bonds
        self.rate_catalog['Corner'] = self.w0*np.exp(-(self.diff+self.edge+self.corner)/self.kT)   # free corner barrier
        self.rate_catalog['Corner_Detach1'] = self.w0*np.exp(-(self.diff+self.edge+self.corner+self.detach)/self.kT)   # corner barrier, break 1 bond
        self.rate_catalog['Corner_Detach2'] = self.w0*np.exp(-(self.diff+self.edge+self.corner+2*self.detach)/self.kT)   # corner barrier, break 2 bonds
                           
        # Create corresponding Process Table, Rate Table, and Move Table to send to solver
        self.rates = []  # Rate Table
        self.moves = []   # Move Table
        self.keys = []  # Table of Keys corresponding to each entry (Move_Barriers)
        
        for i in range(len(Moves)):
            for j in range(len(Barriers[i])):
                self.rates.append(self.rate_catalog[Barriers[i][j]])
                self.moves.append([Moves_X[i],Moves_Y[i],Moves_Z[i]])
                self.keys.append((Moves[i]+'_'+Barriers[i][j]))
                
        self.rates = np.array(self.rates,dtype=np.double)
        self.moves = np.array(self.moves,dtype=np.int32)



    def Downward_Funneling_Moves(self):
        """ Define Downward Funneling Moves. Only in-plane coordinate changes. """
        
        # Array is 1D and defines move of x coordinate. Left or Right
        if self.dim == '2D':
            DF_Moves = np.array([-1,1],dtype=np.int8)
            
        # Array is 2D, first index is x-position, 2nd index is y-position. Order is north, east, south, west
        elif self.dim == '3D':
            DF_Moves = np.array([[0,1],[1,0],[0,-1],[-1,0]],dtype=np.int8)  
        
        return DF_Moves










# End of Module