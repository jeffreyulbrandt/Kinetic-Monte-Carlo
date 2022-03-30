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

Version: 3.4.1a

"""



# import modules
from pathlib import Path  # this is used
import numpy as np
import pickle
import csv
import time
import tables

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from scipy.constants import physical_constants, convert_temperature


#####################
###   Constants   ###
#####################

colors_mpl = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']  # matplotlib default colors
              
colors = [(0,0,0), (0,1,0), (1,0,0), (0,0,1), (1,0,1), (0,1,1), (1,1,0)]  # Black -> G -> R -> B -> V -> C -> Y (appaerntly can add 4th value for alpha)




###################
###   Classes   ###
###################

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



#####################################
###   Substrate Generator Class   ###
#####################################
        
class Substrate_Generator:
    """
    Generate starting substrates based off input parameters.
    
    Creates islands, steps, or load custom coordinates (i.e previous simulation)
    
    Inputs
    ------
    Parameters:   Parameters dictionary form input script
        
    Attributes
    ----------
    dim:      simulation dimension
    Lx,Ly:    substrate dimensions
    depth:    surface normal dimension
    style:    Substrate Type: Islands, Steps, Custom
    state:    
    
    Outputs
    -------
    
    """

    def __init__(self,Parameters):
        self.Parameters = Parameters
        
        # extract relevant parameters
        self.dim = self.Parameters['Dimension']
        self.Lx = self.Parameters['Lx']
        self.Ly = self.Parameters['Ly']
        self.depth = self.Parameters['depth']
        
        self.style = self.Parameters['Substrate_Type']
        self.state = self.Parameters['Substrate_Particle_State']
        self.file_name = self.Parameters['Substrate_File']           # name of the substrate file
        
        self.grid = self.Parameters['Feature_Grid']                  # 'Square', 'Hex'
        self.grid_layout = self.Parameters['Feature_Layout']         # 'Uniform', 'Correlated', 'Random', 'Custom'
        self.numX = self.Parameters['Feature_Spacing'][0]
        self.numY = self.Parameters['Feature_Spacing'][1]
        self.grid_sigma = self.Parameters['Feature_Spacing'][2]      # Guassian sigma for island spacing. Only for Correlated.
        
        self.feature_dist = self.Parameters['Size_Distribution']     # 'None', 'Gaussian', or 'Correlated'
        self.feature_length = self.Parameters['Size_Values'][0]      # avg radius or step length
        self.feature_height = self.Parameters['Size_Values'][1]      # height of the islands or steps
        self.feature_width = self.Parameters['Size_Values'][2]       # width if Gassian
        
        

        
    def Generate_Substrate(self):
        """ Return list of particles (tuple) and number of particles to KMC_Engine """
        
        # coordinate list. List of tuples of particle coordinates: (x,y,z,m) for 3D, (x,z,m) for 2D
        self.List = []
        
        # Islands
        if self.style == 'Islands':
            if self.dim == '3D':
                self.Make_Islands()
            elif self.dim == '2D':
                self.Make_Islands_2D()
#            print ('n',self.n)
#            print ('max of List',np.max(self.Surface))

        # Steps
        if self.style == 'Steps':
            self.Make_Steps()
            self.Surface = (self.Lattice==0).argmax(axis=2)-1  # for plotting
#            print ('n',self.n)
#            print ('max of List',np.max(self.Lattice))
                        
        # Custom
        if self.style == 'Custom':
            self.Load_Custom()
#            print ('n',self.n)
#            print ('max of List',np.max(self.Surface))
        
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
                    tmp = (i,j,2,m)      # set z = 2 to put in 3rd layer
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
                    
                    

                    
    def Load_Custom(self):
        """ Load a custom substrate file. Format must be numpy array in a .npy file """
        
        # Load the lattice data
        path = Path.cwd() / self.file_name
        
        coord = np.load(path)
        
        # Go through coordiante table and add particles to list
        if self.dim == '3D':
            Lattice = np.zeros([self.Lx,self.Ly,self.depth],np.int32)   # recreate physical lattice
            Lattice[:,:,0:2] = 1.0
            self.Surface = np.zeros([self.Lx,self.Ly],np.int32)         # make a surface for plotting
        elif self.dim == '2D':
            Lattice = np.zeros([self.Lx,self.depth],np.int32)          # recreate physical lattice
            Lattice[:,0:2] = 1.0
            self.Surface = np.zeros([self.Lx],np.int32)                # make a surface for plotting
                
        # first active particle id
        self.n = 2
        
        # make list of particles, fill in Lattice           
        if self.state == 'Passive':
            for i in range(coord.shape[0]):
                if self.dim == '3D':
                    x = coord[i,0]; y = coord[i,1]; z = coord[i,2]
                    Lattice[x,y,z] = 1
                    tmp = (x,y,z,1)
                if self.dim == '2D':
                    x = coord[i,0]; z = coord[i,1]
                    Lattice[x,z] = 1
                    tmp = (x,z,1)
                self.List.append(tmp)
        elif self.state == 'Active':
            for i in range(coord.shape[0]):
                if self.dim == '3D':
                    x = coord[i,0]; y = coord[i,1]; z = coord[i,2]
                    Lattice[x,y,z] = self.n
                    tmp = (x,y,z,self.n)
                if self.dim == '2D':
                    x = coord[i,0]; z = coord[i,1]
                    Lattice[x,z] = self.n
                    tmp = (x,z,self.n)
                self.List.append(tmp)
                self.n += 1

        # create surface (for plotting purposes)
        if self.dim == '3D':
            for i in range(Lattice.shape[0]):
                for j in range(Lattice.shape[1]):
                    column = Lattice[i,j,:]
                    layer_index = np.where(column == 0)[0][0]
                    self.Surface[i,j] = layer_index - 1
        if self.dim == '2D':
            for i in range(Lattice.shape[0]):
                    column = Lattice[i,:]
                    layer_index = np.where(column == 0)[0][0]
                    self.Surface[i] = layer_index - 1
    


    
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
        
        xx,yy = np.meshgrid(np.arange(self.Lx),np.arange(self.Ly))
        
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        if self.Lx > 200:
            ax.imshow(np.transpose(self.Surface))
        else:
            ax.pcolormesh(xx.T,yy.T,self.Surface.T,edgecolors='k',lw=0.4,antialiased=True)
        ax.set_aspect('equal')
        plt.tight_layout()
        
        

        
    def Plot_2D(self):
        """ Plot for a 1+1 simulation """
        pass
    


    
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
        


###########################
###   Save File Class   ###
###########################        

class Save:
    """ 
    Save File Class. Creates standard Save File format for On-the-Fly Analysis and Data save files using PyTables and hdf5 files
    Future: Want to save data after each simulation and replace with each new simulation.
    
    Attributes:
    -----------
    Parameters:  Parameters dictionary
    path:        Path to the save file. Pathlib object
    file_type:   Type of file being saved
    file_name:   Name prefix of the File
    
    Methods:
    --------
    Sep_Up:  Create file and set up hierarchy
    Add_Initial_Data:  Add results of first simulation
    Add_New_Data:   Add and sum data from simulations
    Save_Data:  Recieve data and save to appropriate location
    
    """        
    
    def __init__(self,Parameters,path,file_type):
        self.Parameters = Parameters
        self.path = path             # path object
        self.file_type = file_type   # type of file being saved
        
        # extract variables from Parameters Dictionary       
        self.file_name = self.Parameters['Save_File_Name']
        self.depth = self.Parameters['depth']
        
        self.coverage = self.Parameters['Coverage']
        self.island = self.Parameters['Island_Analysis']
        self.structure = self.Parameters['Structure_Factor']
        
        # make new variables
        self.num_sims = 1    # index to track number of sims saved
        
        # create the file
        self.Set_Up()
        
        
        
    ###############################
    ###   File Set Up Methods   ###
    ###############################    
    
    def Set_Up(self):
        """ Set up the save file """
        
        if self.file_type == 'Data':
            self.save_name = str(self.path / (self.file_name + '_Data.hdf5'))
            self.Set_Up_Data()
        elif self.file_type == 'Analysis':
            self.save_name = str(self.path / (self.file_name + '_On_the_Fly_Analysis.hdf5'))
            self.Set_Up_Analysis()
        elif self.file_type == 'Analysis_Avg':
            self.save_name = str(self.path / (self.file_name + '_Averaged_Analysis.hdf5'))
            self.Set_Up_Analysis()
            
            
    
    def Set_Up_Data(self):
        """ Set up a save file for data """
        
        # Make a .hdf5 file with pytables errays        
        save_file = tables.open_file(self.save_name,mode='a')
        
        # Create Dummy Data to start earrays
        if self.Parameters['Dimension'] == '2D':
            x = np.zeros([1,2],dtype=np.uint16)
        elif self.Parameters['Dimension'] == '3D':
            x = np.zeros([1,3],dtype=np.uint16)
        particle = np.zeros([1],dtype=np.uint32)
        time = np.zeros([1],dtype=np.double)
        
        # Create Data hierarchy
        root = save_file.root
        data_group_1 = save_file.create_group(root,'Event_Data','Event Data')
        save_file.create_group(root,'Lattice_Data','Lattice Data')
        save_file.create_group(root,'Surface_Data','Surface Data')
        save_file.create_group(root,'Time_Data','Save Time Data')  # to save time when lattice data was collected
        
        # create e-arrays for event data
        save_file.create_earray(data_group_1,'X',obj=x)
        save_file.create_earray(data_group_1,'ID',obj=particle)
        save_file.create_earray(data_group_1,'Time',obj=time)
        
        # Close File?
        save_file.close()  # note: we can keep file open as long as it doesn't take up too much memory
        
    
    
    def Set_Up_Analysis(self):
        """ Set up the save file for analysis and averaged analysis """

        # create hdf5 file
        save_file = tables.open_file(self.save_name,mode='a')
        
        # Create Data hierarchy
        root = save_file.root        
        save_file.create_group(root,'Number_Sims','Number of Simulations')  # this tracks the number of sims saved
        save_file.create_group(root,'Analysis_Times','Analysis Times')
        save_file.create_group(root,'Coverage','Coverage Analysis')
        save_file.create_group(root,'Island_Count','Island Count Analysis')
        Island = save_file.create_group(root,'Island_Size','Island Analysis')
        for i in range(0,self.depth):
            save_file.create_group(Island,'Layer_'+str(i),'Island Analysis Layer '+str(i))
        save_file.create_group(root,'Structure_Factor','Structure Factor Analysis')
        
        # Close File
        save_file.close()
        
    

    
    #############################    
    ###   Data File Methods   ###
    #############################
    
    def Save_Event_Data(self,save_times,save_id,save_coordinates):
        """ Save Event Data to File. """
        
        # open the save file
        save_file = tables.open_file(self.save_name,mode='r+')
        
        # append data to the save file
        save_file.root.Event_Data.Time.append(save_times)
        save_file.root.Event_Data.ID.append(save_id)        
        save_file.root.Event_Data.X.append(save_coordinates)
        
        # close the file
        save_file.close()
        


        
    def Save_Lattice_Data(self,data,i):
        """ Save Lattice data to file. """
                
        save_file = tables.open_file(self.save_name,mode='r+')
        save_file.create_array('/Lattice_Data','T'+i,data,'Lattice'+i)
               
        save_file.close()
        
        

        
    def Save_Surface_Data(self,data,i):
        """ Save surface only data. Smallest file, but lose layer info. """
        
        save_file = tables.open_file(self.save_name,mode='r+')
        save_file.create_array('/Surface_Data','T'+i,data,'Surface'+i)
        save_file.close()
        
    

    
    def Save_Time_Data(self,data):
        """ Save times when lattice data was collected. """
        
        save_time = np.array(data)
        save_file = tables.open_file(self.save_name,mode='r+')        
        save_file.create_array('/Time_Data','Times',save_time,'Save Times')        
        save_file.close()
    
    

    
    #################################
    ###   Analysis File Methods   ###
    #################################
    
    def Save_Analysis(self,Data):
        """ Save Analysis data at end of simulation 
        
        Data File Structure
        -------------------
        Data = [layers, [Analysis Times], [Coverage Data], [Island Data], [Structure Factor Data]]
        layers:                 number of analyzed layers
        Actual Times:           [Analysis Times, Actual save times]
        Coverage Data:          [coverage, reflectivity, rms roughness, average height]
        Island Data:            [number of clusters, layers, [island data], [island index]]
        Structure Factor Data:  [structure factor array]
        
        """
        
        num_layers = Data[0]        # number of analyzed layers
        points = Data[1][0].shape[0]   # number of analysis points
        save_file = tables.open_file(self.save_name,mode='r+')
        
        # save number of sims
        save_file.create_array('/Number_Sims','Number_Sims',np.array([1]),'Number of Simulations')
                
        # save analysis times
        save_file.create_array('/Analysis_Times','Analysis_Times',Data[1][0],'Analysis Times')        
        save_file.create_array('/Analysis_Times','Actual_Times',Data[1][1],'Actual Times')
        
        # save coverage analysis
        if self.coverage == True:
            save_file.create_array('/Coverage','Coverage',Data[2][0],'Coverage')
            save_file.create_array('/Coverage','Reflectivity',Data[2][1],'Reflectivity')
            save_file.create_array('/Coverage','Roughness',Data[2][2],'Roughness')
            save_file.create_array('/Coverage','Avg_Height',Data[2][3],'Average Height')
            
        # save island counts
        if self.island == True:
            save_file.create_array('/Island_Count','Island_Count',Data[3][0],'Island Count')
            
            island_data = Data[3][1]
            island_index = Data[3][2]
            
            # save island size distribution
            length = len(str(points)) + 1     # number of digits in save id 
            for layer in range(num_layers):
                for i in range(len(island_index[layer])):
                    # number id for save points
                    ii = '0'*(length-len(str(island_index[layer][i]))) + str(island_index[layer][i])  
                    
                    # check if an array exists and save it
                    if island_data[layer][i].shape[0] != 0:
                        save_file.create_array('/Island_Size/Layer_'+str(layer+1),'P'+str(ii),island_data[layer][i],'P'+str(ii))
                    
        # Save Struture Factor Data
        if self.structure == True:
            save_file.create_array('/Structure_Factor','Structure_Factor',Data[4][0],'Structure Factor')
        
        save_file.close()
    
    

    
    #####################################
    ###   Averaged Analysis Methods   ###
    #####################################

    def Add_Initial_Data(self,Data):
        """ Initial data from first simulation """
        
        self.num_layers = Data[0]  # number of analyzed layers
        
        self.analysis_times = Data[1][0]
        self.actual_times = Data[1][1]
        
        if self.coverage == True:
            self.cov = Data[2][0]
            self.refl = Data[2][1]
            self.rms = Data[2][2]
            self.height = Data[2][3]
        
        if self.island == True:
            self.num_clusters = Data[3][0]
            self.island_data = Data[3][1]
            self.island_index = Data[3][2]
        
        if self.structure == True:
            self.structure_factor = Data[4][0]
        

        
        
    def Add_New_Data(self,Data):
        """ add new data to summed data """
        # sum actual times        
        self.actual_times += Data[1][1]
        
        # sum coverage and reflectivity results
        if self.coverage == True:
            self.cov += Data[2][0]
            self.refl += Data[2][1]
            self.rms += Data[2][2]
            self.height += Data[2][3]
        
        # combine island distribution data
        if self.island == True:
            self.num_clusters += Data[3][0]
            self.island_data,self.island_index = Combine(self.island_data,self.island_index,Data[3][1],Data[3][2])
        
        # sum structure factor results
        if self.structure == True:
            self.structure_factor += Data[4][0]
        
        # update sims counter
        self.num_sims += 1
        
 

    
    def Save_Data(self):
        """ Save averaged data to file. """
            
        points = self.actual_times.shape[0]       
        save_file = tables.open_file(self.save_name,mode='r+')
        
        # save number of sims
        save_file.create_array('/Number_Sims','Number_Sims',np.array([self.num_sims]),'Number of Simulations')
        
        # save analysis times
        save_file.create_array('/Analysis_Times','Analysis_Times',self.analysis_times,'Analysis Times')           
        save_file.create_array('/Analysis_Times','Actual_Times',self.actual_times,'Actual Times')
        
        # save coverage analysis
        if self.coverage == True:
            save_file.create_array('/Coverage','Coverage',self.cov,'Coverage')
            save_file.create_array('/Coverage','Reflectivity',self.refl,'Reflectivity')
            save_file.create_array('/Coverage','Roughness',self.rms,'Roughness')
            save_file.create_array('/Coverage','Avg_Height',self.height,'Average Height')
            
        # save island size distribution
        if self.island == True:
            save_file.create_array('/Island_Count','Island_Count',self.num_clusters,'Island Count')
            
            length = len(str(points)) + 1  # number of digits in save id    
            for layer in range(self.num_layers):
                for i in range(len(self.island_index[layer])):
                    # number id for save points
                    ii = '0'*(length-len(str(self.island_index[layer][i]))) + str(self.island_index[layer][i])
                    
                    # check if an array exists and save it
                    if self.island_data[layer][i].shape[0] != 0:
                        save_file.create_array('/Island_Size/Layer_'+str(layer),'P'+str(ii),self.island_data[layer][i],'P'+str(ii))
                    
        # save structure factor analysis
        if self.structure == True:
            save_file.create_array('/Structure_Factor','Structure_Factor',self.structure_factor,'Structure Factor')
        
        save_file.close()
        
        
        
 
       
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    
    
#######################    
###   Timer Class   ###
#######################
        
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
        
        

#######################################
###   Simulation Statistics Class   ###
#######################################
        
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



##############################
###   Progress Bar Class   ###
##############################

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
        
    
    
##################################
###   Data Combine Functions   ###
##################################
            
            
# combine function for combining island size distribution data
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




##########################
###   Save Functions   ###
##########################


def Save_Parameters_Pickle(parameters,name):
    """ Save parameters dictionary to pickle file. """
        
    # pickle file
    file = open(name, 'wb')         
    pickle.dump(parameters, file)       
    file.close()
        
        
        
def Save_Parameters_CSV(parameters,name):
    """ Save parameters dictionary to .csv file """
    
    # csv file
    header = ['Parameter','Value']                
    with open(name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header)
        for key, value in parameters.items():
            line = [key, value]
            writer.writerow(line)
            
            

def Save_Final_Lattice_old(lattice,save_name):
    """ Save the lattice at the end of simulation. Cut off layers with no particles. """
    
    layer_cov = np.count_nonzero(lattice,axis=(0,1))
    print (layer_cov)
    if layer_cov[-1] == 0:
        layer_index = np.where(layer_cov == 0)[0][0]
        lattice = lattice[:,:,0:layer_index]
    
    np.save(save_name,lattice)
    
    
    
def Save_Final_Lattice(lattice,save_name):
    """ Save the lattice at the end of simulation as a coordinate table. """
    
    np.save(save_name,lattice)
    
    
    
            
################
###  Other   ###
################


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




######################################
###   General Plotting Functions   ###
######################################


def Plot_Lattice_2D(Lattice,size):
    """ Plot a side view of a 1+1 lattice. Old version using patches. May be useful for plotting particles as circles. """
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
    """ Plot a top down view of 2+1 lattice. Old version using patches. May be useful for plotting particles as circles. """
    
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
    

    
    
def Plot_Final_Surface_3D(Lattice,size_x,size_y):
    """ Plot top down view of 2+1 surface. """
    
    # resize lattice to size
    Lattice = Lattice[:size_x,:size_y]
    
    # Calculate the lowest surface height
    bottom = np.min(Lattice)
    
    # subtract out complete layers
    Surface = Lattice - bottom
    
    # make meshgrid
    X,Y = np.meshgrid(np.arange(Lattice.shape[0]),np.arange(Lattice.shape[1]))
    
    # create colormap
    cmap_name = 'my_list'
    Custom = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))
    
    # make figure
    fig,ax = plt.subplots(figsize=(18,18))
    ax.set_aspect('equal')
    
#    ax.pcolormesh(X,Y,Surface,vmin=np.min(Surface),vmax=np.max(Surface),edgecolors='k',lw=0.005,cmap=Custom)
    ax.pcolormesh(X,Y,Surface,vmin=0,vmax=7,edgecolors='k',lw=0.4,cmap=Custom,antialiased=True)   
#    ax.pcolormesh(X,Y,Surface,vmin=0,vmax=4,cmap=Custom)   # without edge lines
    
#    fig.colorbar(im)
    
  
    
    
def Plot(x,y,xlabel,ylabel):
    """ Generic plot of x,y data. """
    
    plt.figure()
    plt.plot(x,y,'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    


    


# End of Module