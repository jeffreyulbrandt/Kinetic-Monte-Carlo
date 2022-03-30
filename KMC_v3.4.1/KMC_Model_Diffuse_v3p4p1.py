##################################################################################################################
##################################################################################################################
#########################                                                                #########################
#########################            PLD Kinetic Monte Carlo Diffuse Module              #########################
#########################                                                                #########################
##################################################################################################################
##################################################################################################################



"""
Kinetic Monte Carlo - Diffuse Module



Version: 3.4.1b

"""



# import python modules
from pathlib import Path   # this is actually used in this module
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# import KMC modules
import KMC_Engine_v3p4p1 as KMC_Engine
import KMC_Miscellaneous_v3p4p1 as KMC_Misc






#########################
###   Diffuse Class   ###
#########################
#--------------------------------------------------------------------------------------------------------------------------------------------------

class Diffuse:
    """
    Diffuse Class: Diffusion only simulation.
    
    
    Inputs
    ------
    Parameters:  Parameters Dictionary from Input Script
    
    
    Attributes
    ----------
    time:   simulation time in seconds
    
    
    Methods
    -------
    Initialize:  Set-up simulation. Create KMC_Engine and On_the_Fly_Analysis objects
    
    """
    
    
    def __init__(self,Parameters,path,seed):
        self.Parameters = Parameters
        self.path = path    # this is a path object
        self.seed = seed    # random number generator seed
        
        # extract useful variables
        self.diff_type = self.Parameters['Diffusion_Type']    # this if the type of simulation to run
        self.coords = self.Parameters['Cluster']              # this is the starting coordinates for the particles
        
        self.enable_printouts = self.Parameters['Enable_Print_Outs']
        self.file_name = self.Parameters['Save_File_Name']
        
        self.Initialize()
        
        
        
    def Initialize(self):
        """ Initialize the Pulse. """
        
        # set up variables
        self.time = 0                       # simulation time
        self.deposited = 0                  # counter for number of deposited particles
        self.n = self.coords.shape[1]       # number of particles in the sim
        self.Parameters['n'] = self.n       # reset the parameters dictionary
        
        # create depsotion times for all particles
        self.dep_times = np.arange(self.coords.shape[1])*1e-12   # deposit particles much faster than diffusion time
                
        # Create File Paths
        self.save_name = str(self.path / (self.file_name + '_Data.hdf5'))                         # Path to Data File
        self.save_parameters = str(self.path / (self.file_name + '_Parameters.pkl'))              # Path to Pickle File
        self.save_csv_summary = str(self.path / (self.file_name + '_Summary.csv'))                # Path to Summary csv File
        
        # Add file paths to Parameters Dictionary
        self.Parameters['Save_Path_Data'] = self.save_name
        self.Parameters['Save_Path_Parameters'] = self.save_parameters
        self.Parameters['Save_Path_Summary'] = self.save_csv_summary
        
        # Save summary pickle file and csv file
        KMC_Misc.Save_Parameters_Pickle(self.Parameters,self.save_parameters)
        KMC_Misc.Save_Parameters_CSV(self.Parameters,self.save_csv_summary)
        
#        # Create an instance of the KMC_Engine
#        if self.Parameters['Dimension'] == '2D':
#            self.Engine = KMC_Engine.KMC_Engine_2D(self.Parameters,self.seed)
#        elif self.Parameters['Dimension'] == '3D':
#            self.Engine = KMC_Engine.KMC_Engine_3D(self.Parameters,self.seed)
            
        # Set up timers
        self.Run_Timer = KMC_Misc.Timer()   # measures time to complete each sim
            




    ##########################################
    ###   Run Diffusion Simulation Types   ###
    ##########################################
            
    def Run(self):
        """ Main Run command  """
        
        if self.diff_type == 'Standard':
            self.Run_Standard()
        elif self.diff_type == 'Island':
            self.Run_Island()
            
            
            

    
    #######################
    ###   Calculations  ###
    #######################
            
    def Center_of_Mass_Initial(self):
        """ calculate center of mass of an initial cluster of size: n """
        
        dr_x0 = 0
        dr_y0 = 0
            
        for i in range(self.n):
            dr_x0 += self.coords[0,i]
            dr_y0 += self.coords[1,i]
            
        dr_x0 = dr_x0/self.n
        dr_y0 = dr_y0/self.n
        
        return (dr_x0,dr_y0)
    
    
    
    def Center_of_Mass_Current(self):
        """ calculate center of mass of a cluster of size: num """
        dr_x = (self.Engine.coordinate_table[0,0] + self.Engine.coordinate_table[1,0] + self.Engine.coordinate_table[2,0])/3
        dr_y = (self.Engine.coordinate_table[0,1] + self.Engine.coordinate_table[1,1] + self.Engine.coordinate_table[2,1])/3
                
        dr_x = 0
        dr_y = 0
            
        for i in range(self.n):
            dr_x += self.Engine.coordinate_table[i,0]
            dr_y += self.Engine.coordinate_table[i,1]
            
        dr_x = dr_x/self.n
        dr_y = dr_y/self.n
        
        return (dr_x,dr_y)
    
    
    
    def Distance2(self,x0,y0,x,y):
        """ Calculate distance between two points """
    
        dr = np.sqrt((x-x0)**2 + (y-y0)**2)
        
        return dr

    
    
    
    
    ####################################
    ###   Island Simulation Methods  ###
    ####################################
    
    def Run_Island(self):
        """ Run an Island Type Simulation """
        
        # Start Timer
        self.Run_Timer.Timer_Start()
        
        # printout
        print ('Starting Island Diffusion Simulation')  
        
        # extract relevant parameters from dictionary
        self.num = self.Parameters['Particle_Number']         # number of particles to simulate in island type
        self.trace = self.Parameters['Tracer_Number']              # number of tajectories to save in island type sim
        
        # store tracer runs
        self.tx = []; self.ty = []
        
        # store taus and drs and steps
        self.tau = []
        self.dr = []
        self.steps = []
        
        # record starting position
        self.x0,self.y0 = self.Center_of_Mass_Initial()
        
        # set up progress bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        # Set-up Engine
        self.Engine = KMC_Engine.KMC_Engine(self.Parameters,self.seed)
        
        # particle id for current diffusing particle
        index = 2
        
        # record tracer runs and store trajectories
        for i in range(self.trace):
            x,y = self.Tracer(index)
            self.tx.append(x); self.ty.append(y)
            self.Progress_Bar.Print(i,self.num)
            index += 1
                    
        # do the rest of the runs
        for i in range(self.trace,self.num):
            self.Single_Run(index)
            self.Progress_Bar.Print(i,self.num)
            index += 1
            
        # save the data
        self.tau = np.array(self.tau)
        self.dr = np.array(self.dr)
        
        Save_Data = np.vstack((self.tau,self.dr))
        Save_Name = str(self.path / (self.file_name + '_Data'))
        np.save(Save_Name,Save_Data)

        # record sim time
        self.Run_Timer.Record_Time()
        
        # prinout
        print ('')
        if self.Run_Timer.store_times[-1] > 3600:
            print ('Simulation Complete. Sim Time = ',str(self.Run_Timer.store_times[-1]/3600), ' hours')
        elif self.Run_Timer.store_times[-1] < 3600 and self.Run_Timer.store_times[-1] > 60:
            print ('Simulation Complete. Sim Time = ',str(self.Run_Timer.store_times[-1]/60), ' minutes')
        else:    
            print ('Simulation Complete. Sim Time = ',str(self.Run_Timer.store_times[-1]), ' seconds')
        print ('')
            

    
    def Single_Run(self,i):
        """ Run until particle goes over edge """        
        
        # deposit particle
        self.Engine.Run_First_Step(self.dep_times,self.coords)
    
        # diffuse until particle drops over edge
        height = self.Engine.coordinate_table[i,2]
        while height == 3:
            self.Engine.Run_One_Step(anneal = 1)
            height = self.Engine.coordinate_table[i,2]
            
        # record tau and dr and steps
        self.tau.append(self.Engine.time[0]-self.time)
        
        dr = np.sqrt((self.Engine.coordinate_table[i,0] - self.x0)**2 + (self.Engine.coordinate_table[i,1] - self.y0)**2)
        self.dr.append(dr)
        
        # rest simulation for next deposition
        self.time = self.Engine.time[0]   # current sim time
        self.dep_times = np.arange(self.coords.shape[1])*1e-12 + self.time   # next dep time       
        self.Engine.lattice[self.Engine.coordinate_table[i,0],self.Engine.coordinate_table[i,1],self.Engine.coordinate_table[i,2]] = 0   # remove old particle

    

    
    def Tracer(self,i):
        """ Diffuse particle to edge, record trajectory """
        
        # store trajectory
        x = []; y = []
        
        # deposit particle
        self.Engine.Run_First_Step(self.dep_times,self.coords)
        x.append(self.Engine.coordinate_table[i,0])
        y.append(self.Engine.coordinate_table[i,1])
        
        # diffuse to edge
        height = self.Engine.coordinate_table[i,2]
        while height == 3:
            self.Engine.Run_One_Step(anneal = 1)
            height = self.Engine.coordinate_table[i,2]
            x.append(self.Engine.coordinate_table[i,0])
            y.append(self.Engine.coordinate_table[i,1])
            
        # record tau and dr
        self.tau.append(self.Engine.time[0]-self.time)
        
        dr = np.sqrt((self.Engine.coordinate_table[i,0] - self.x0)**2 + (self.Engine.coordinate_table[i,1] - self.y0)**2)
        self.dr.append(dr)
        
        # rest simulation for next deposition
        self.time = self.Engine.time[0]   # current sim time
        self.dep_times = np.arange(self.coords.shape[1])*1e-12 + self.time   # next dep time      
        self.Engine.lattice[self.Engine.coordinate_table[i,0],self.Engine.coordinate_table[i,1],self.Engine.coordinate_table[i,2]] = 0   # remove old particle
        


        return (x,y)
    
    
    
    def Island_Results(self):
        """ Analyze results of island simulation """
        
        # stats on stuff
        
        
        Avg_Tau = np.average(self.tau)
        Avg_dr = np.average(self.dr)
        
        print ('Average Tau: ',Avg_Tau)
        print ('Average Distance: ',Avg_dr)
#        print ('drs: ',self.dr)
        
        # Tau Histogram
        Bins = np.linspace(np.min(self.tau),np.max(self.tau),100)
        plt.figure(figsize=(12,6))
        plt.hist(self.tau,bins=Bins)
        
        
        # Plot the substrate
        surface = self.Engine.Surface() - 1
        
        colors = [(0,0,0), (0,1,0), (1,0,0), (0,0,1), (1,0,1), (0,1,1), (1,1,0)]
        cmap_name = 'my_list'
        Custom = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))
        
        X,Y = np.meshgrid(np.arange(surface.shape[0]),np.arange(surface.shape[1]))
        
        plt.figure(figsize=(18,18))
#        plt.pcolormesh(X,Y,surface,vmin=0,vmax=7,edgecolors='k',lw=0.4,cmap=Custom,antialiased=True)
        plt.pcolormesh(X,Y,surface,vmin=0,vmax=7,cmap=Custom)
       
        # Plot the tracer trajectories
        for i in range(self.trace):
            plt.plot(self.tx[i],self.ty[i]) 
            
        # save the trajectory plot
        Save_Name = str(self.path / (self.file_name + 'Trajectory.png'))
        plt.savefig(Save_Name,dpi=300)
            
    




    ######################################
    ###   Standard Simulation Methods  ###
    ######################################
    
    def Run_Standard(self):
        """ Run a Standard Tye Simulation """
        
        # extract relevant parameters from dictionary
        self.end_of_pulse = self.Parameters['Sim_Time']       # similation time in seconds in standard type
        
        # analysis results
        self.displacement = []       # store diplacement from initial position
        
        # set up save files
        if self.save or self.save_lattice or self.save_surface == True:
            self.save_data_file = KMC_Misc.Save(self.Parameters, self.path, 'Data')
            
        if self.diff_type == 'Standard':
            self.length = len(str(int(self.end_of_pulse/self.Parameters['Save_Times']))) + 1  # number of digits in save id

        # set up the save times
        self.save_times = np.arange(0,self.end_of_pulse + self.Parameters['Save_Times'], self.Parameters['Save_Times'])
        self.save_index = 0    # index for analyze time
        self.save_times_actual = []    # save actual times when save occurs
        
        
        
    def Run_Mono(self):
        """ Run Simulation for monomer """
        
        # reset the counters
        self.deposited = 0
        
        # Deposit first particle to get things going
        self.time = self.Engine.First_Step(self.dep_times,self.coords,self.end_of_pulse)
        self.deposited += 1
        
        # Create Progress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        if self.enable_printouts == True:
            self.Progress_Bar.Print(0,self.end_of_pulse)
               
        # Deposit the particles
        while self.deposited < self.n: 
            
            # Do Deposition Sequence
            self.time = self.Engine.Step()
            
            # update deposited counter
            self.deposited += 1
            
            
        # Allow particle to diffuse for fixed time
        while self.time < self.end_of_pulse:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
        
            # Pause the sim at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                
#                # Analyze the displacement of the first particle
#                dr = np.sqrt((self.Engine.coordinate_table[0,0] - self.coords[0,0])**2 + (self.Engine.coordinate_table[0,1] - self.coords[1,0])**2)
#                self.displacement.append(dr)
                
                # Analyze the mean square displacement of the first particle
                dr = (self.Engine.coordinate_table[0,0] - self.coords[0,0])**2 + (self.Engine.coordinate_table[0,1] - self.coords[1,0])**2
                self.displacement.append(dr)
                
                # save the data
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time,self.end_of_pulse)
                    
        return (dr_x,dr_y)



    def Run_Cluster(self):
        """ Run the Simulation for cluster of size n """
        
        # reset the counters
        self.deposited = 0
        
        # Deposit first particle to get things going
        self.time = self.Engine.First_Step(self.dep_times,self.coords,self.end_of_pulse)
        self.deposited += 1
        
        # Create Progress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        if self.enable_printouts == True:
            self.Progress_Bar.Print(0,self.end_of_pulse)
               
        # Deposit the particles
        while self.deposited < self.n: 
            
            # Do Deposition Sequence
            self.time = self.Engine.Step()
            
            # update deposited counter
            self.deposited += 1
            
        # starting center of mass of cluster        
        dr_x0,dr_y0 = self.Center_of_Mass_Initial()   
        
            
        # Allow particle to diffuse for fixed time
        while self.time < self.end_of_pulse:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
        
            # Pause the sim at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                
                # Analyze the mean square displacement of the center of mass
                dr_x,dr_y = self.Center_of_Mass_Current()                 
                dr = (dr_x - dr_x0)**2 + (dr_y - dr_y0)**2
                self.displacement.append(dr)
                
                # save the data
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time,self.end_of_pulse)
                    
        return (dr_x,dr_y)
                    
                    
    def Run_Trimer(self):
        """ Run the Simulation for trimer cluster """
        
        # reset the counters
        self.deposited = 0
        
        # Deposit first particle to get things going
        self.time = self.Engine.First_Step(self.dep_times,self.coords,self.end_of_pulse)
        self.deposited += 1
        
        # Create Progress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        if self.enable_printouts == True:
            self.Progress_Bar.Print(0,self.end_of_pulse)
               
        # Deposit the particles
        while self.deposited < self.n: 
            
            # Do Deposition Sequence
            self.time = self.Engine.Step()
            
            # update deposited counter
            self.deposited += 1
            
        # starting center of mass of cluster        
        dr_x0 = (self.coords[0,0] + self.coords[0,1] + self.coords[0,2])/3
        dr_y0 = (self.coords[1,0] + self.coords[1,1] + self.coords[0,2])/3
            
            
        # Allow particle to diffuse for fixed time
        while self.time < self.end_of_pulse:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
        
            # Pause the sim at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                
                # Analyze the mean square displacement of the center of mass
                dr_x = (self.Engine.coordinate_table[0,0] + self.Engine.coordinate_table[1,0] + self.Engine.coordinate_table[2,0])/3
                dr_y = (self.Engine.coordinate_table[0,1] + self.Engine.coordinate_table[1,1] + self.Engine.coordinate_table[2,1])/3
                dr = (dr_x - dr_x0)**2 + (dr_y - dr_y0)**2
                self.displacement.append(dr)
                
                # save the data
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time,self.end_of_pulse)
                    
                    
                    
    def Run_Pentamer(self):
        """ Run the Simulation for pentamer cluster """
        
        # reset the counters
        self.deposited = 0
        
        # Deposit first particle to get things going
        self.time = self.Engine.First_Step(self.dep_times,self.coords,self.end_of_pulse)
        self.deposited += 1
        
        # Create Progress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        if self.enable_printouts == True:
            self.Progress_Bar.Print(0,self.end_of_pulse)
               
        # Deposit the particles
        while self.deposited < self.n: 
            
            # Do Deposition Sequence
            self.time = self.Engine.Step()
            
            # update deposited counter
            self.deposited += 1
            
        # starting center of mass of cluster        
        dr_x0 = (self.coords[0,0] + self.coords[0,1] + self.coords[0,2] + self.coords[0,3] + self.coords[0,4])/5
        dr_y0 = (self.coords[1,0] + self.coords[1,1] + self.coords[0,2] + self.coords[1,3] + self.coords[0,4])/5
            
            
        # Allow particle to diffuse for fixed time
        while self.time < self.end_of_pulse:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
        
            # Pause the sim at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                
                # Analyze the mean square displacement of the center of mass
                dr_x = (self.Engine.coordinate_table[0,0] + self.Engine.coordinate_table[1,0] + self.Engine.coordinate_table[2,0] + self.Engine.coordinate_table[3,0] + self.Engine.coordinate_table[4,0])/5
                dr_y = (self.Engine.coordinate_table[0,1] + self.Engine.coordinate_table[1,1] + self.Engine.coordinate_table[2,1] + self.Engine.coordinate_table[3,1] + self.Engine.coordinate_table[4,1])/5
                dr = (dr_x - dr_x0)**2 + (dr_y - dr_y0)**2
                self.displacement.append(dr)
                
                # save the data
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time,self.end_of_pulse)
                    
                    
                    
        
    
    
    #############################
    ###   Save Data Methods   ###
    #############################
           
    def Save_Event_Data(self):
        """ Save Event Data to File. """
        
        # get the data
        if self.Parameters['Dimension'] == '2D':
            x,z,save_id,save_times = self.Engine.Save_Events()
            save_coordinates = np.transpose(np.vstack((x,z)))            
        elif self.Parameters['Dimension'] == '3D':
            x,y,z,save_id,save_times = self.Engine.Save_Events()
            save_coordinates = np.transpose(np.vstack((x,y,z)))
        
        self.save_data_file.Save_Event_Data(save_times,save_id,save_coordinates)
    
    
    
    def Save_Lattice_Data(self,i):
        """ Save a snapshot of the lattice via the coordinate table """
        
        # get the coordinates of all current deposited particles. Note: Doesn't include passive substrate particles.
        save_data = self.Engine.Lattice_Coordinates()
        
        # create string for naming array in order
        ii = '0'*(self.length-len(str(i))) + str(i)
        
        self.save_data_file.Save_Lattice_Data(save_data,ii)
        
        
        
    def Save_Surface_Data(self,i):
        """ Save snapshot of the surface. Smallest file, but lose layer info. """

        # Get surface data
        save_data = self.Engine.Surface()
        save_data = save_data.astype(np.uint8)  # max layer is 256 for this dtype. (Shoud just do this in Engine)
       
        # create string for naming array in order
        ii = '0'*(self.length-len(str(i))) + str(i)
        
        self.save_data_file.Save_Surface_Data(save_data,ii)
        
        
        
    def Save_Time_Data(self):
        """ Save times when lattice data was collected. """
        
        self.save_data_file.Save_Time_Data(self.save_times_actual)
        
    
    
    
    
# End of Module