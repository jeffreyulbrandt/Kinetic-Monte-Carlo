########################################################################################################################
########################################################################################################################
#########################                                                                      #########################
#########################            PLD Kinetic Monte Carlo Miscellaneous Module              #########################
#########################                                                                      #########################
########################################################################################################################
########################################################################################################################


"""
Kinetic Monte Carlo Miscellaneous Module

Contains a bunch of useful functions and support classes

Version: 3.3a

"""

# import modules
from pathlib import Path  # this is used
import numpy as np
import csv
import time
import tables

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from scipy.constants import physical_constants, convert_temperature




###################
###   Classes   ###
###################
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Process Catalog Class

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
        if self.type == 'Deposition':        
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




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        
        

# Deposition Class

class Deposition:
    """
    Deposition Class: Generate a Deposition Pulse, including times, coordinates, incident angles, and energies
    
    
    Inputs
    ------
    Parameters:  parameters dictionary from input script
    
    
    Methods
    -------
    Deposition_Rates:          return average and instantaneous deposition rates
    Create_Deposition_Pulse:   return array and deposition times and coordinates
    Uniform_Deposition:        creates array with uniform deposition rate
    Maxwellian:                creates array with Maxwell-Boltzmann distribution of dep times (future)
    Energy_Distribution:       create an energy distribution (future)
    Angle_Distribution:        deposition angle distribution (future)
    
    
    Output
    ------
    Dep_Times: 1D array of deposition times, 1 larger than n
    Dep_Coordinates:   2D Array of dep coordinates
    
    
    """
    
    def __init__(self,Parameters):
        self.Parameters = Parameters
        
        self.dim = self.Parameters['Dimension']   # dimension of simulation
        self.shape = self.Parameters['Pulse_Shape']    # shape of deposition distribution        
        self.dep_width = self.Parameters['Pulse_Width']    # width of the dep pulse in seconds
        self.dwell = Parameters['Dwell_Time']    # dwell time in seconds
        self.n = self.Parameters['n']    # number of particles in pulse
        
        self.Lx = Parameters['Lx']
        self.Ly = Parameters['Ly']
        
        self.dt = self.dep_width/self.n   # average time between dep events for uniform deposition
        
        
        
    def Deposition_Rates(self):
        """ Return deposition rates """
        
        if self.dim == '2D':
            Dep_Rate_Avg = self.n/(self.dwell*self.Lx)
            Dep_Rate_Peak = self.n/(self.dep_width*self.Lx)
            
        if self.dim == '3D':
            Dep_Rate_Avg = self.n/(self.dwell*self.Lx*self.Ly)
            Dep_Rate_Peak = self.n/(self.dep_width*self.Lx*self.Ly)
                    
        return (Dep_Rate_Avg, Dep_Rate_Peak)


        
    def Create_Deposition_Pulse(self,time):
        """ Create Data for a Deposition Pulse """
        
        # times
        self.time = time
        if self.shape == 'Uniform':
            self.Uniform_Deposition()
        elif self.shape == 'Maxwellian':
            self.Maxwell()
        
        # coordinates
        if self.dim == '2D':
            self.Dep_Coordinates = np.random.randint(0,self.Lx,self.n)
            self.Dep_Coordinates = self.Dep_Coordinates.astype(np.int32)
        elif self.dim == '3D':
            self.Dep_Coordinates = np.vstack((np.random.randint(0,self.Lx,self.n),np.random.randint(0,self.Ly,self.n)))
            self.Dep_Coordinates = self.Dep_Coordinates.astype(np.int32)
            
        # Future: Incidence angles along x-direction
        
        # Future: Incidence Energies
        
        return (self.Dep_Times,self.Dep_Coordinates)
        

    
    def Uniform_Deposition(self,sigma=0.3):
        """ Generates a uniform pulse of deposition times with random noise """
        
        # Note: make one more particle than needed. It won't actually be deposited, it just prevents an error in Solver
        Noise = np.random.normal(0,sigma*self.dt,self.n)  # random gaussian noise for dep times
        Noise = np.insert(Noise,0,0)
        self.Dep_Times = np.linspace(self.time,self.time+self.dep_width+self.dt,self.n+1) - Noise  # deposition times
        self.Dep_Times = self.Dep_Times.astype(np.double)



    def Maxwellian(self,time,sigma):
        """ Generate a Maxwell-Boltzmann distrubted deposition pulse """
        pass

    
    
    def Energy_Distribution(self):
        """ Generate energy distribution based on Pulse_Shape """
        pass
    

    
    def Angle_Distribution(self):
        """ Generate angle distribution. Returns Azimuthal and Polar Angles """
        pass
    
    
    
    def Plot(self):
        """ Plot distributions """
        pass
        
        
        
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        


# Substrate Generator Class
        
class Substrate_Generator:
    """
    Generate starting substrates based off input parameters
    """

    def __init__(self,Parameters):
        self.Parameters = Parameters
        
        # extract relevant parameters
        self.dim = Parameters['Dimension']
        self.Lx = Parameters['Lx']
        self.Ly = Parameters['Ly']
        self.depth = Parameters['depth']
        
        self.style = self.Parameters['Substrate_Type']
        self.state = self.Parameters['Substrate_Particle_State']
        
        self.grid_layout = self.Parameters['Feature_Layout']    # 'Uniform', 'Correlated', or 'Random'
        self.numX = self.Parameters['Feature_Spacing'][0]
        self.numY = self.Parameters['Feature_Spacing'][1]
        self.grid_sigma = self.Parameters['Feature_Spacing'][2]
        
        self.feature_dist = Parameters['Size_Distribution']     # 'None', 'Gaussian', or 'Correlated'
        self.feature_length = Parameters['Size_Values'][0]      # avg radius or step length
        self.feature_width = Parameters['Size_Values'][1]       # width if Gassian
        
        
        
    def Generate_Substrate(self):
        """ Return list of particles (tuple) and number of particles to KMC_Engine """
        
        self.List = []
        
        # Islands
        if self.style == 'Islands':
            if self.dim == '3D':
                self.Make_Islands()
            elif self.dim == '2D':
                self.Make_Islands_2D()
            print ('n',self.n)
            print ('max of List',np.max(self.Surface))

        # Steps
        if self.style == 'Steps':
            self.Make_Steps()
            self.Surface = (self.Lattice==0).argmax(axis=2)-1  # for plotting
            print ('n',self.n)
            print ('max of List',np.max(self.Lattice))
        
        return (self.List,self.n)
    
 
    
    def Make_Islands(self):
        """ create array of islands on surface """
        # Make a surface
        self.Surface = np.zeros([self.Lx,self.Ly],np.int32)
        
    
        ####################################
        ###   Calculate Island Spacing   ###
        ####################################
        
        # Uniform spacing between islands in x direction
        spacing_x = int(self.Lx/self.numX)  
        start_x = int(0.5*spacing_x)
        
        # Uniform spacing between islands in y direction
        spacing_y = int(self.Ly/self.numY)
        start_y = int(0.5*spacing_y)
        
        # Uniform spacing (num of features should be factors of substrate lengths)
        X = np.arange(start_x,self.Lx,spacing_x)
        Y = np.arange(start_y,self.Ly,spacing_y)      
        coord = [[p1, p2] for p1 in X for p2 in Y]
        
        # For correlated distribution, add random offset
        if self.grid_layout == 'Correlated':
            for k in range(len(coord)):
                coord[k][0] += np.random.randint(-self.grid_sigma,self.grid_sigma+1)
                coord[k][1] += np.random.randint(-self.grid_sigma,self.grid_sigma+1)
        
        # If random distribution, then replace X,Y
        if self.grid_layout == 'Random':
            X = np.random.randint(0,self.Lx,X.shape[0])
            Y = np.random.randint(0,self.Ly,Y.shape[0])
            coord = [[p1, p2] for p1 in X for p2 in Y]
            
            
        ######################################################
        ###   Calculate Islands Sizes and add to Surface   ###
        ######################################################
        
        # first active particle id
        self.n = 2
        
        # Loop through coord and make islands
        for k in range(len(coord)):
            if self.feature_dist == 'None':
                tmp = self.Circle(coord[k][0],coord[k][1],self.feature_length)
                
            elif self.feature_dist == 'Gaussian':
                # Test a couple of different random numer generators
                radius = int(np.random.normal(self.feature_length,self.feature_width))  # normal dist
#                radius = np.random.randint(self.feature_length-self.feature_width,self.feature_length+self.feature_width+1)  # uniform dist                
                tmp = self.Circle(coord[k][0],coord[k][1],radius) # list of particles in island
                
            elif self.feature_dist == 'Correlated':
                Radii = self.Radial_Distribution(coord,k)
                Radii = Radii[np.nonzero(Radii)]  # remove the zero
                radius = int(self.feature_width*np.min(Radii))  # less than half the distance to nearest island                
                tmp = self.Circle(coord[k][0],coord[k][1],radius)
            
            # add particles to surface
            for i,j in tmp:
                if self.state == 'Passive':
                    self.Surface[i%self.Lx,j%self.Ly] = 1                    
                elif self.state == 'Active':
                    self.Surface[i%self.Lx,j%self.Ly] = self.n
                    self.n += 1
                
        # Convert surface particles to list of particles
        for i in range(self.Lx):
            for j in range(self.Ly):
                m = self.Surface[i,j]
                if m > 0:
                    tmp = (i,j,1,m)
                    self.List.append(tmp)
                    
                    
    
    def Make_Islands_2D(self):
        """ 2D version of island generator 
        
        Note: Y values in the 3D case are treated as Z values here. This allows tall islands to be specified.
        
        Currently only allows uniform islands.
        
        """
        
        # Make a space
        self.Surface = np.zeros([self.Lx,self.depth],np.int32)
        
        # Width and height of islands
        width = self.feature_length
        height = self.feature_width
        
        # Uniform spacing between islands in x direction
        spacing_x = int(self.Lx/self.numX)  
        start_x = int(0.5*spacing_x)
        
        # Uniform spacing (num of features should be factors of substrate lengths)
        X = np.arange(start_x,self.Lx,spacing_x)
        print (X)
                    
        # For correlated distribution, add random offset
        if self.grid_layout == 'Correlated':
            for k in range(X.shape[0]):
                X[k] += np.random.randint(-self.grid_sigma,self.grid_sigma+1)
        
        # If random distribution, then replace X
        if self.grid_layout == 'Random':
            X = np.random.randint(0,self.Lx,X.shape[0])
            
        # first active particle id
        self.n = 2
            
        # Make the Islands
        for j in range(X.shape[0]):
            tmp = []
            for i in range(width):
                shift = X[j] + i - int(0.5*width)
                # for heights, start at layer 2.
                for k in range(2,height+2):
                    tmp.append((shift,k))
                    
            # add particles to surface
            for i,k in tmp:
                if self.state == 'Passive':
                    self.Surface[i%self.Lx,k] = 1                    
                elif self.state == 'Active':
                    self.Surface[i%self.Lx,k] = self.n
                    self.n += 1

        # Convert surface particles to list of particles
        for i in range(self.Lx):
            for k in range(self.depth):
                m = self.Surface[i,k]
                if m > 0:
                    tmp = (i,k,m)
                    self.List.append(tmp)

                    
    def Make_Steps(self):
        """ Make steps on the surface 
        
        User specifies avg step length
        
        Algorithm determines number of steps and where to place them. For active substrates, need to deposit lower layers
        first.
        
        """
        
        # How many steps
        steps = int(np.floor(0.5*self.Lx/self.feature_length))
        print ('Steps',steps)
        
        # legth of each step
        lengths = np.zeros([steps],dtype=np.int32)
        for i in range(steps):
            lengths[i] = self.feature_length + i*2*self.feature_length
        lengths = np.flip(lengths)
        print ('Lengths',lengths)
        
        # start of each step
        starts = (0.5*self.Lx*np.ones(lengths.shape[0]) - 0.5*lengths).astype(np.int32)
        print ('starts',starts)
        
        # Make a sub-lattice
        self.Lattice = np.zeros([self.Lx,self.Ly,steps+1],np.int32)
        
        # first active particle id
        self.n = 2
        
        # fill in steps
        if self.state == 'Passive':
            for i in range(steps):
                self.Lattice[starts[i]:(starts[i]+lengths[i]),:,i] = 1
        elif self.state == 'Active':
            for k in range(steps):
                for i in range(starts[k],(starts[k]+lengths[k])):
                    for j in range(self.Ly):
                        self.Lattice[i,j,k] = self.n
                        self.n += 1
    
        # Convert surface particles to list of particles
        for k in range(steps):
            for i in range(self.Lx):
                for j in range(self.Ly):
                    m = self.Lattice[i,j,k]
                    if m > 0:
                        tmp = (i,j,k+1,m)   # make z one higher
                        self.List.append(tmp)
                        
                        
                        
    def Make_Steps_2D(self):
        """ Make steps on the 2D surface 
        
        User specifies avg step length
        
        Algorithm determines number of steps and where to place them. For active substrates, need to deposit lower layers
        first.
        
        """
        
        # How many steps
        steps = int(np.floor(0.5*self.Lx/self.feature_length))
        print ('Steps',steps)
        
        # legth of each step
        lengths = np.zeros([steps],dtype=np.int32)
        for i in range(steps):
            lengths[i] = self.feature_length + i*2*self.feature_length
        lengths = np.flip(lengths)
        print ('Lengths',lengths)
        
        # start of each step
        starts = (0.5*self.Lx*np.ones(lengths.shape[0]) - 0.5*lengths).astype(np.int32)
        print ('starts',starts)
        
        # Make a sub-lattice
        self.Lattice = np.zeros([self.Lx,steps+1],np.int32)
        
        # first active particle id
        self.n = 2
        
        # fill in steps
        if self.state == 'Passive':
            for i in range(steps):
                self.Lattice[starts[i]:(starts[i]+lengths[i]),i] = 1
        elif self.state == 'Active':
            for k in range(steps):
                for i in range(starts[k],(starts[k]+lengths[k])):
                    self.Lattice[i,k] = self.n
                    self.n += 1
    
        # Convert surface particles to list of particles
        for k in range(steps):
            for i in range(self.Lx):
                m = self.Lattice[i,k]
                if m > 0:
                    tmp = (i,k+1,m)   # make z one higher
                    self.List.append(tmp)
    

    
    def Radial_Distribution(self,coord,i):
        """ Calculate the radial distribution funnction for island CoM.
        
        Inputs:
        -------
        i:  index a particle to test
        
        Outputs:
        --------
        Radii:  Array of radial distace to each other particle
        
        """
        
        # Note: Currently is not taking into account periodic boubary conditions!
        
        # arrays
        size = len(coord)
        Radii = np.zeros(size,dtype=np.double)  # distance to other particles
        
        # calculate distances
        for k in range(size):
            Radii[k] = np.sqrt((coord[k][0]-coord[i][0])**2 + (coord[k][1]-coord[i][1])**2)
            
        return Radii
    
    
    
    def Plot(self):
        """ Make a plot of the substrate for evaluation """
        
#        fig,ax = plt.subplots(1,1,figsize=(0.2*self.Lx,0.2*self.Ly))
        fig,ax = plt.subplots(1,1,figsize=(10,10))
#        ax.imshow(np.transpose(self.Surface),origin='lower')
        ax.pcolormesh(np.transpose(self.Surface),edgecolors='k',lw=0.005)
        ax.set_aspect('equal')
        plt.tight_layout()
    

    
    def Circle(self,x,y,r):
        """ Generate list of coordinates for a circle of radius r at x,y """
        
        # set ranges
        xrange = np.arange(x-r-1,x+r+2)
        yrange = np.arange(y-r-1,y+r+2)
        
        List = []
        for i in xrange:
            for j in yrange:
                R = np.sqrt((x-i)**2+(y-j)**2)
                if R >= r:
                    pass
                else:
                    List.append((i,j))
                    
        return List
    
        


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        



# Save File Class

class Save:
    """ 
    Save File Class. Creates standard Save File format for On-the-Fly Analysis Average save files using PyTables and hdf5 files
    Future: Want to save data after each simulation and replace with each new simulation.
    
    Attributes:
    -----------
    Parameters:  Parameters dictionary
    path:        Path to the save file. Pathlib object
    file_name:   Name prefix of the File
    
    Methods:
    --------
    Sep_Up:  Create file and set up hierarchy
    Add_Initial_Data:  Add results of first simulation
    Add_New_Data:   Add and sum data from simulations
    Save_Data:  Recieve data and save to appropriate location
    
    """        
    
    def __init__(self,Parameters,path):
        self.Parameters = Parameters
        self.path = path  # path object
        self.file_name = self.Parameters['Save_File_Name']
        self.depth = self.Parameters['depth']
        self.num_sims = 1   # index to track number of sims saved
        
        self.Set_Up()
        
        
    
    def Set_Up(self):
        """ Set up the save file """
        # create hdf5 file
        self.Save_Name = str(self.path / (self.file_name + '_Averaged_Analysis.hdf5'))
        Save_File = tables.open_file(self.Save_Name,mode='a')
        
        # Create Data hierarchy
        root = Save_File.root
        
        Save_File.create_group(root,'Number_Sims','Number of Simulations')  # this tracks the number of sims saved
        Save_File.create_group(root,'Analysis_Times','Analysis Times')
        Save_File.create_group(root,'Coverage','Coverage Analysis')
        Island = Save_File.create_group(root,'Island_Size','Island Analysis')
        for i in range(0,self.depth):
            Save_File.create_group(Island,'Layer_'+str(i),'Island Analysis Layer '+str(i))
        Save_File.create_group(root,'Structure_Factor','Structure Factor Analysis')
        
        # Close File
        Save_File.close()
    


    def Add_Initial_Data(self,Analysis_Times,Actual_Times,Cov,Refl,Island_Data,Island_Index,Structure_Factor,RMS,Height):
        """ Initial data from first simulation """
        self.Analysis_Times = Analysis_Times
        self.Actual_Times = Actual_Times
        self.Cov = Cov
        self.Refl = Refl
        self.Island_Data = Island_Data
        self.Island_Index = Island_Index
        self.Structure_Factor = Structure_Factor
        self.RMS = RMS
        self.Height = Height

        
        
    def Add_New_Data(self,Actual_Times,Cov,Refl,Island_Data,Island_Index,Structure_Factor,RMS,Height):
        """ add and average latest data set """
        # sum actual times        
        self.Actual_Times += Actual_Times
        
        # sum coverage and reflectivity results
        self.Cov += Cov
        self.Refl += Refl
        self.RMS += RMS
        self.Height += Height
        
        # combine island distribution data
        self.Island_Data,self.Island_Index = Combine(self.Island_Data,self.Island_Index,Island_Data,Island_Index)
        
        # sum structure factor results
        self.Structure_Factor += Structure_Factor
        
        # update sims counter
        self.num_sims += 1
        
 
    
    def Save_Data(self):
        """ Save averaged data to file """
        
        points = self.Actual_Times.shape[0]       
        Save_File = tables.open_file(self.Save_Name,mode='r+')
        
        # save number of sims
        Save_File.create_array('/Number_Sims','Number_Sims',np.array([self.num_sims]),'Number of Simulations')
        
        # save analysis times        
        Save_File.create_array('/Analysis_Times','Analysis_Times',self.Analysis_Times,'Analysis Times')
        Save_File.create_array('/Analysis_Times','Actual_Times',self.Actual_Times,'Actual Times')
        
        # save coverage analysis        
        Save_File.create_array('/Coverage','Coverage',self.Cov,'Coverage')
        Save_File.create_array('/Coverage','Reflectivity',self.Refl,'Reflectivity')
        Save_File.create_array('/Coverage','Roughness',self.RMS,'Roughness')
        Save_File.create_array('/Coverage','Avg_Height',self.Height,'Average Height')
            
        # save island size distribution
        length = len(str(points)) + 1  # number of digits in save id    
        for layer in range(self.depth):
            for i in range(len(self.Island_Index[layer])):
                # number id for save points
                ii = '0'*(length-len(str(self.Island_Index[layer][i]))) + str(self.Island_Index[layer][i])
                
                # check if an array exists and save it
                if self.Island_Data[layer][i].shape[0] != 0:
                    Save_File.create_array('/Island_Size/Layer_'+str(layer),'P'+str(ii),self.Island_Data[layer][i],'P'+str(ii))
                    
        # save structure factor analysis
        Save_File.create_array('/Structure_Factor','Structure_Factor',self.Structure_Factor,'Structure Factor')
        
        Save_File.close()
        
        
        
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    
    
    
# Timer Class

class Timer:
    """ 
    Data Structure to store timer data and calculate timer stats.
    """        
    
    def __init__(self):
        self.timer_begin = time.perf_counter()
        self.start_time = 0
        self.store_times = []

    
    def Timer_Start(self):
        """ begin a timer """        
        self.start_time = time.perf_counter()

        
    def Record_Time(self):
        """ Save delta time """
        ttt = time.perf_counter()-self.start_time
        self.store_times.append(ttt)
        return ttt

        
    def Average(self):
        avg = np.average(self.store_times)
        std = np.std(self.store_times)
        
        return (avg, std)        




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        


# Simulation Statistics Class
        
class Statistics:
    """ Data structure for simulation statistics. Saves data into .csv file. Try to keep number of entries in file less than 256
    which is the number pf columns in Excel.
    
    Attributes
    ----------
    Computer_Time
    Sim_Time
    Steps
    Dep_Events
    Diff_Events
    Processes_Available
    Processes_Chosen
    
    Methods
    -------
    update_counters: Update chosen events
    record_data:
    analyze:
    Plot:   Simulation Statistics Plot

    
    """
    
    def __init__(self,plot_stats,path,keys,num_proc):
        
        self.plot_stats = plot_stats  # True/False, whether to make stats plot
        self.path = path              # file path to save data
        self.keys = keys              # keys for the elementary process
        self.num_proc = num_proc        # number of unique processes
        
        self.Initialize()
        
    
    
    def Initialize(self):
        """ Set up all data structures """
        
        # data storage
        self.Sim_Times = []         # Simulation Times
        self.Steps = []             # Total Steps per Pulse
        self.Dep_Events = []        # Number of Deposition Events
        self.Diff_Events = []       # Number of Diffusion Events
        self.Ni_Count = []          # Snapshot of number of particles in each process list
        self.Processes_Chosen = []  # Cummulative number of chosen processes
        
        # Header list for save file
        self.Ni_Header = []
        self.Chosen_Header = []
        for i in range(int(len(self.keys)/4)):
            key = self.keys[i][6:]  # cut off the 'North_'
            self.Ni_Header.append('Ni_'+ key)
            self.Chosen_Header.append('Chosen_' + key)
        
        # timer
        self.exe_timer = Timer()
        self.exe_timer.Timer_Start()
    
        
        
    def Record_Data(self,sim_time,Process_Counters,Event_Counters,Ni):
        """ add steps data """       
        self.exe_timer.Record_Time()
        self.Sim_Times.append(sim_time)
        self.Steps.append(Event_Counters[0])
        self.Dep_Events.append(Event_Counters[1])
        self.Diff_Events.append(Event_Counters[2])
        
        # sum across all directions
        sum_Ni = np.array([Ni[i]+Ni[i+self.num_proc]+Ni[i+2*self.num_proc]+Ni[i+3*self.num_proc] for i in range(self.num_proc)],dtype=np.int64)
        self.Ni_Count.append(sum_Ni)
        
        sum_Proc = np.array([Process_Counters[i]+Process_Counters[i+self.num_proc]+Process_Counters[i+2*self.num_proc]+Process_Counters[i+3*self.num_proc] for i in range(self.num_proc)],dtype=np.int64)
        self.Processes_Chosen.append(sum_Proc)
    

    
    def Analyze(self):
        """ analyze and store simulation stats """
        # get average pulse stats
        pass
    

    
    def Save_Stats(self,SaveName):
        """ save statistics to a file """
        
        Header = ['Exe_Time','Sim_Time','Steps','Dep_Events','Diff_Events'] + self.Ni_Header + self.Chosen_Header
        
        with open(SaveName, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(Header)
            for i in range(len(self.Steps)):
                line = [self.exe_timer.store_times[i],self.Sim_Times[i],self.Steps[i],self.Dep_Events[i],self.Diff_Events[i]]+\
                        self.Ni_Count[i].tolist()+self.Processes_Chosen[i].tolist()
                writer.writerow(line)
        

            
    def Plot(self):
        """ Plot statistics """
        
#        XLabel = ['Pulse'+str(i) for i in range(1,len(self.Steps)+1)]
#        y_pos = np.arange(len(XLabel))
        
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,4))
        
        ax1.set_title('Simulation Statistics - Number of Steps')
        ax1.set_ylabel('Steps')
#        ax1.bar(y_pos, self.Steps, align='center', alpha=0.5)
#        ax1.set_xticks(y_pos,XLabel)
        ax1.plot(self.Steps)
        
        ax2.set_title('Simulation Statistics - Pulse Time')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_xlabel('Simulation Time (s)')
        ax2.plot(self.Sim_Times,self.exe_timer.store_times)
        
        
        

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Progress Bar Class


class Progress_Bar:
    """ Progress Bar Class
    
    Inputs:
    -------
    prefix: Str to print at start of progress bar
    suffix: Str to print at end of progress bar
    decimals: number of decimals to print in percent complete
    length:  character length of the bar
    fill:  bar fill character (Str)
    printEnd:  End character
    
    
    """
    
    def __init__(self,prefix='',suffix='',decimals=1,length=100,fill='|',printEnd='\r'):
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.printEnd = printEnd
        
    
    def Print(self,iteration,total):
        """ Printout of Progress Bar """
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(self.length * iteration // total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end = self.printEnd)
        
        # Print New Line on Complete
        if iteration == total:
            print()
    



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        


#####################
###   Functions   ###
#####################
            
            
# develop combine function
def Combine(Island_Size_1, Island_Index_1, Island_Size_2, Island_Index_2, Dtype=np.uint32):
    """ Combine two sets of island size distribution data. """
    # size of sets (must be the same)
    max0 = len(Island_Size_1)
    
    # new data structures for combined data
    Island_Size = [[] for _ in range(max0)]
    Island_Index = [[] for _ in range(max0)]
    
    for i in range(max0):
        # find common indices
        a = np.array(Island_Index_1[i])
        b = np.array(Island_Index_2[i])
        if a.shape[0] == 0 and b.shape[0] == 0:
            pass
        else:
            common = np.union1d(a,b).astype(Dtype)
            common_list = common.tolist()
            for ii in common_list:
                Island_Index[i].append(ii)

            # loop across common indices
            for j in common:
                # check which arrays contain the index
                # if zero, then its not in array
                a1 = np.where(a == j)[0]
                b1 = np.where(b == j)[0]

                # add data to new array
                if a1.shape[0] != 0 and b1.shape[0] == 0:
                    a2 = Island_Size_1[i][a1[0]].astype(Dtype)
                    Island_Size[i].append(a2)
                elif a1.shape[0] == 0 and b1.shape[0] != 0:
                    b2 = Island_Size_2[i][b1[0]].astype(Dtype)
                    Island_Size[i].append(b2)
                elif a1.shape[0] != 0 and b1.shape[0] != 0:
                    hist1 = Island_Size_1[i][a1[0]][0,:]
                    bins1 = Island_Size_1[i][a1[0]][1,:]
                    hist2 = Island_Size_2[i][b1[0]][0,:]
                    bins2 = Island_Size_2[i][b1[0]][1,:]

                    bins = np.union1d(bins1,bins2).astype(Dtype)  # combine bins
                    hist = np.zeros([bins.shape[0]],dtype=Dtype)
                    for k in range(bins.shape[0]):
                        # check each set
                        c = np.where(bins1==bins[k])[0]
                        d = np.where(bins2==bins[k])[0]
                        if c.shape[0] > 0:
                            hist[k] += hist1[c[0]]
                        if d.shape[0] > 0:
                            hist[k] += hist2[d[0]]
                            
                    c2 = np.vstack((hist,bins))
                    Island_Size[i].append(c2)
                            
    return (Island_Size,Island_Index)




def Time_Convert(Time_in_Seconds):
    """ Convert seconds to hours, minutes, seconds """
    
    m,s = divmod(Time_in_Seconds,60)
    h,m = divmod(m,60)

    return (h,m,s)



# scientific functions
def Pre_Factor(T):
    """ Calculate the Prefactor based on crystal vibration frequency: freq = kT/h.  """
    Tk = convert_temperature(T,'Celsius', 'Kelvin')
    kB = physical_constants[ 'Boltzmann constant in eV/K' ][0]
    h = physical_constants[ 'Planck constant in eV s' ][0]
    w0 = (Tk*kB)/h
    print ('Prefactor =','{:.4}'.format(w0),'Hz')
    return (w0)




# General Plotting Functions

def Plot_Lattice_2D(Lattice,size):
    """ Plot a side view of a 1+1 lattice """
    # resize lattice to size
    Lattice = Lattice[:size,:]
    
    Substrate = np.ones([Lattice.shape[0],3])  # make a substrate for plotting purposes
        
    fig,ax = plt.subplots(figsize=(14,6)) # 
    ax.set_aspect('equal')
    
    # plot the substrate as black squares with white outline
    xdata = np.arange(0,Substrate.shape[0])
    zdata = np.arange(0,Substrate.shape[1])
    for a in xdata:
        for b in zdata:
            ax.add_patch(Rectangle((a,b),1,1,fc='k',ec='w',lw=1.0))
            
    # plot deposited particles as red squares with black outline
    for i in range(Lattice.shape[0]):
        for k in range(Lattice.shape[1]):
            if Lattice[i,k] == 1:
                ax.add_patch(Rectangle((i,k+Substrate.shape[1]),1,1,fc='g',ec='k',lw=1.0))
            elif Lattice[i,k] > 1:
                ax.add_patch(Rectangle((i,k+Substrate.shape[1]),1,1,fc='r',ec='k',lw=1.0))
                            
    ax.set_xlim(0,Lattice.shape[0])
    ax.set_ylim(0,Lattice.shape[1]+Substrate.shape[1]+5)
    

    
def Plot_Lattice_3D_Top_Down(Lattice,size_x,size_y):
    """ Plot a top down view of 3D lattice """
    
    Col = ['k','red','lightsalmon','mistyrose','w','b','c']  # colors for heights
    
    # resize lattice to size
    Lattice = Lattice[:size_x,:size_y,:]
    height = Lattice.shape[2]  # depth of lattice
    
    fig,ax = plt.subplots(figsize=(15,15))
    ax.set_aspect('equal')
    
    # plot the substrate as black squares with white outline
    xdata = np.arange(0,size_x)
    ydata = np.arange(0,size_y)
    for a in xdata:
        for b in ydata:
            ax.add_patch(Rectangle((a,b),1,1,fc='k',ec='w',lw=0.5))
            
    # plot rest of lattice.
    for i in range(size_x):
        for j in range(size_y):
            for k in range(height):
                if Lattice[i,j,k] > 0:
                    ax.add_patch(Rectangle((i,j),1,1,fc=Col[k],ec='w',lw=0.5))
            
    ax.set_xlim(0,size_x)
    ax.set_ylim(0,size_y)
    


def Plot_Lattice_3D_Top_Down_2(Lattice,size_x,size_y):
    """ Plot top down view of lattice as image """
    
    # resize lattice to size
    Lattice = Lattice[:size_x,:size_y,:]
        
    # Calculate Surface Image
    Surface = (Lattice==0).argmax(axis=2)-1
#    minZ = np.min(Surface)
#    maxZ = np.max(Surface)
#    print (minZ,maxZ)
    
    # make meshgrid
    X,Y = np.meshgrid(np.arange(Lattice.shape[0]),np.arange(Lattice.shape[1]))
    
    # create colormap
    colors = [(0,0,0), (0,1,0), (1,0,0), (0,0,1), (1,0,1), (0,1,1), (1,1,0)]  # Black -> G -> R -> B -> V -> C -> Y
#    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','e377c2','7f7f7f','bcbd22','17becf']
    cmap_name = 'my_list'
    Custom = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))
    
    # make figure
    fig,ax = plt.subplots(figsize=(18,18))
    ax.set_aspect('equal')
    
#    ax.pcolormesh(X,Y,Surface,vmin=np.min(Surface),vmax=np.max(Surface),edgecolors='k',lw=0.005,cmap=Custom)
    ax.pcolormesh(X,Y,Surface,vmin=0,vmax=4,edgecolors='k',lw=0.005,cmap=Custom)   
#    ax.pcolormesh(X,Y,Surface,vmin=0,vmax=4,cmap=Custom)   # without edge lines
    
#    fig.colorbar(im)
    


    


# End of Module