##################################################################################################################
##################################################################################################################
#########################                                                                #########################
#########################            PLD Kinetic Monte Carlo Density Module              #########################
#########################                                                                #########################
##################################################################################################################
##################################################################################################################



"""
Kinetic Monte Carlo - Density Module

Test of ability to create custom simuation model. This one is for studying monomer density.

Version: 3.4.0

"""



# import python modules
from pathlib import Path   # this is actually used in this module, ignore the warning
import numpy as np
import time
import tables
import matplotlib.pyplot as plt

# import KMC modules
import KMC_Engine_v3p4p1 as KMC_Engine
import KMC_Deposition_v3p4p1 as KMC_Dep
import KMC_Miscellaneous_v3p4p1 as KMC_Misc
import KMC_Analysis_v3p4p1 as KMC_Analysis




#######################
###   Pulse Class   ###
#######################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------


class Density:
    """
    Monomer density map simulation. Make a 2D density map.
    
    
    """
    
    
    def __init__(self,Parameters,path):
        self.Parameters = Parameters
        
        self.path = path   # this is a path object
        
        # extract useful variables from Parameters dictionary
        self.enable_plots = self.Parameters['Enable_Plots']
        self.enable_printouts = self.Parameters['Enable_Print_Outs']
        
        self.file_name = self.Parameters['Save_File_Name'] 
        self.save_dt = self.Parameters['Save_Time_Step']
        
        self.dim = self.Parameters['Dimension']
        self.lx = self.Parameters['Lx']
        self.ly = self.Parameters['Ly']
        self.depth = self.Parameters['depth']
        
        self.analysis_style = self.Parameters['Analysis_Style']
        self.analysis_dt = Parameters['Analysis_Delta_Time']
        
        self.dwell_time = self.Parameters['Dwell_Time']
        self.layers = self.Parameters['Layers']
        self.pulses = self.Parameters['Pulses']
        self.n = self.Parameters['n']
        
        self.sims = self.Parameters['Total_Sims']
        
        
        self.Initialize()
        
        
        
    def Initialize(self):
        """ Initialize the Pulse Object """
        
        # Redefine some Parameters        
        self.Parameters['Coverage'] = True
        self.Parameters['Island_Analysis'] = False    
        self.Parameters['Structure_Factor'] = False
        self.Parameters['On_the_Fly_Save'] = False
        
        # Create new variables
        self.time = 0                      # current simulation time
        self.step = 0                      # current simulation step       
                
        # Create File Paths
        self.save_name = str(self.path / (self.file_name + '_Data.hdf5'))                         # Path to Data File
        self.save_parameters = str(self.path / (self.file_name + '_Parameters.pkl'))              # Path to Pickle File
        self.save_csv_summary = str(self.path / (self.file_name + '_Summary.csv'))                # Path to Summary csv File
                
        # Add file paths to Parameters Dictionary
        self.Parameters['Save_Path_Data'] = self.save_name
        self.Parameters['Save_Path_Parameters'] = self.save_parameters
        self.Parameters['Save_Path_Summary'] = self.save_csv_summary
        
        # Set up Save Data file
        self.Save_Setup()
                          
        # Save summary pickle file and csv file
        KMC_Misc.Save_Parameters_Pickle(self.Parameters,self.save_parameters)
        KMC_Misc.Save_Parameters_CSV(self.Parameters,self.save_csv_summary)
               
        # Create an instance of Deposition
        self.Deposition = KMC_Dep.Deposition(self.Parameters)
        
        # Create an instance of On-the_Fly Analysis for the analysis times
        lattice = np.zeros([self.lx,self.ly,self.depth+2],dtype=np.int32)    # dummy lattice
        lattice[:,:,0:2] = 1.0         
        self.Analysis = KMC_Analysis.On_the_Fly_Analysis(self.Parameters,self.path,lattice)   
        self.analysis_times = self.Analysis.analysis_times

        # Set up timers
        self.Run_Timer = KMC_Misc.Timer()   # measures time to complete each sim
        


        
    def Run(self):
        """ Simulation loop. """
        
        # print start info
        print ('Starting Monomer Density Map Simulation')
        print ('')
        print ('')
        
        self.surfaces = np.zeros([self.lx,self.ly,self.analysis_times.shape[0]],dtype=np.uint64)
        self.avg_save_times = np.zeros([self.analysis_times.shape[0]],dtype=np.float64)
        self.coverage = np.zeros([self.depth,self.analysis_times.shape[0]],dtype=np.float64)
        self.reflectivity = np.zeros([self.analysis_times.shape[0]],dtype=np.float64)
        
        for i in range(self.sims):
            
            # print out
            print ('Starting Simulation ', str(i+1), ' out of ', str(self.sims))  
            
            # sim timer
            self.Run_Timer.Timer_Start()

            # reset simulation time, steps, and index
            self.time = 0
            self.step = 0
            self.analysis_index = 1   # don't do the time = 0 point
            
            # Create seed for RNG from clock time
            self.seed = int(1000*time.perf_counter())
            
            # Create an instance of KMC_Engine
            self.Engine = KMC_Engine.KMC_Engine(self.Parameters,self.seed)
            
            # Create Analysis instance
            self.Analysis = KMC_Analysis.On_the_Fly_Analysis(self.Parameters,self.path,self.Engine.lattice)
            self.Analysis.Do_Analysis(self.Engine.lattice,0)    # analyze starting surface
            
            # get deposition coordinates
            self.dep_times, self.dep_coordinates = self.Deposition.Create_Deposition_Pulse(self.time)
            
            # reset data arrays
            surface = np.zeros([self.lx,self.ly,self.analysis_times.shape[0]],dtype=np.uint64)
            save_times = np.zeros([self.analysis_times.shape[0]],dtype=np.float64)
            coverage = np.zeros([self.depth,self.analysis_times.shape[0]],dtype=np.float64)
            
            # record starting surface
            surface[:,:,0] = self.Engine.Surface() - 1
            
            # run simulation
            surface, save_times = self.Run_Sim(surface,save_times)
            
            # recover analysis data
            Save_Data = self.Analysis.Return_Data()
            coverage = Save_Data[2][0]
            refl = Save_Data[2][1]
            
            # sum the surface and times
            self.surfaces += surface
            self.avg_save_times += save_times
            self.coverage += coverage
            self.reflectivity += refl
            
            # sim time
            self.Run_Timer.Record_Time()
            
            # prinout
            print ('')
            print ('Simulation ',str(i+1),' Complete. Simulation Time = ',str(self.Run_Timer.store_times[-1]), ' seconds')
            print ('')
            
      
        # convert sums to average
        self.avg_save_times = self.avg_save_times/self.sims
        self.surfaces = self.surfaces/self.sims
        self.coverage = self.coverage/self.sims
        self.reflectivity = self.reflectivity/self.sims
        
        # Save surfaces and times to file
        self.Save_Surfaces(self.surfaces,self.avg_save_times,self.coverage,self.reflectivity)
        
        # Display final surface plot
        self.Plot()
        
        # Print Final info
        avg_sim_time, stdev = self.Run_Timer.Average()
        print ('Simulations Complete!!!!')
        print ('Average Simulation Time = ',str(avg_sim_time/60),' minutes.')
        
        
        
    def Run_Sim(self,save_surface,save_times):
        """ Run One Simulation """
        
        # run first step to get things going
        self.Engine.Run_First_Step(self.dep_times,self.dep_coordinates)
        
        # Progress Bar                
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        for i in range(self.analysis_times.shape[0]-1):
                                  
            # run simulation
            self.time, self.step, error_code = self.Engine.Run_until_Time(end_time = self.analysis_times[self.analysis_index], anneal=0)
            self.Progress_Bar.Print(self.time,self.dwell_time)
            
            # analyze the coverage
            self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)
            
            # update stats file
            self.Engine.Save_Sim_Stats()
            
            # get surface
            surface = self.Engine.Surface() - 1
            
            # add to sum_surface
            save_surface[:,:,i+1] = np.copy(surface)
            
            # save simulation time
            save_times[i+1] = self.time
            
            # update analysis times index
            self.analysis_index += 1
        
        return save_surface, save_times
    
    

    
    def Run_No_Printouts(self):
        """ This version only has one progress bar for total simulations """
        
        # print start info
        print ('Starting Monomer Density Map Simulation')
        print ('')
        print ('')
        
        self.surfaces = np.zeros([self.lx,self.ly,self.analysis_times.shape[0]],dtype=np.uint64)
        self.avg_save_times = np.zeros([self.analysis_times.shape[0]],dtype=np.float64)
        self.coverage = np.zeros([self.depth,self.analysis_times.shape[0]],dtype=np.float64)
        self.reflectivity = np.zeros([self.analysis_times.shape[0]],dtype=np.float64)
        
        # Set up Proress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        for i in range(self.sims):
            
            # sim timer
            self.Run_Timer.Timer_Start()

            # reset simulation time, steps, and index
            self.time = 0
            self.step = 0
            self.analysis_index = 1   # don't do the time = 0 point
            
            # Create seed for RNG from clock time
            self.seed = int(1000*time.perf_counter())
            
            # Create an instance of KMC_Engine
            self.Engine = KMC_Engine.KMC_Engine(self.Parameters,self.seed)
            
            # Create Analysis instance
            self.Analysis = KMC_Analysis.On_the_Fly_Analysis(self.Parameters,self.path,self.Engine.lattice)
            self.Analysis.Do_Analysis(self.Engine.lattice,0)    # analyze starting surface
            
            # get deposition coordinates
            self.dep_times, self.dep_coordinates = self.Deposition.Create_Deposition_Pulse(self.time)
            
            # reset data arrays
            surface = np.zeros([self.lx,self.ly,self.analysis_times.shape[0]],dtype=np.uint64)
            save_times = np.zeros([self.analysis_times.shape[0]],dtype=np.float64)
            coverage = np.zeros([self.depth,self.analysis_times.shape[0]],dtype=np.float64)
            
            # record starting surface
            surface[:,:,0] = self.Engine.Surface() - 1
            
            # run simulation
            surface, save_times = self.Run_Sim_No_Printouts(surface,save_times)
            
            # recover analysis data
            Save_Data = self.Analysis.Return_Data()
            coverage = Save_Data[2][0]
            refl = Save_Data[2][1]
            
            # sum the surface and times
            self.surfaces += surface
            self.avg_save_times += save_times
            self.coverage += coverage
            self.reflectivity += refl
            
            # sim time
            self.Run_Timer.Record_Time()
            
            # update progress bar
            self.Progress_Bar.Print(i+1,self.sims)
            
                  
        # convert sums to average
        self.avg_save_times = self.avg_save_times/self.sims
        self.surfaces = self.surfaces/self.sims
        self.coverage = self.coverage/self.sims
        self.reflectivity = self.reflectivity/self.sims
        
        # Save surfaces and times to file
        self.Save_Surfaces(self.surfaces,self.avg_save_times,self.coverage,self.reflectivity)
        
        # Display final surface plot
        if self.enable_plots == True:
            self.Plot()
        
        # Print Final info
        avg_sim_time, stdev = self.Run_Timer.Average()
        print ('Simulations Complete!!!!')
        print ('Average Simulation Time = ',str(avg_sim_time),' seconds.')
        


        
    def Run_Sim_No_Printouts(self,save_surface,save_times):
        """ Run One Simulation """
        
        # run first step to get things going
        self.Engine.Run_First_Step(self.dep_times,self.dep_coordinates)
        
        for i in range(self.analysis_times.shape[0]-1):
                                  
            # run simulation
            self.time, self.step, error_code = self.Engine.Run_until_Time(end_time = self.analysis_times[self.analysis_index], anneal=0)
                        
            # analyze the coverage
            self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)
            
            # update stats file
            self.Engine.Save_Sim_Stats()
            
            # get surface
            surface = self.Engine.Surface() - 1
            
            # add to sum_surface
            save_surface[:,:,i+1] = np.copy(surface)
            
            # save simulation time
            save_times[i+1] = self.time
            
            # update analysis times index
            self.analysis_index += 1
        
        return save_surface, save_times    
        
    
    

    
    def Plot(self):
        """ Plot Final Averaged Surface """
        
        Vmin = 1.0 - self.layers
        Vmax = 1.0 + 1.05*self.layers
        
        # surface plot
        plt.figure(figsize=(18,18))
        plt.imshow(self.surfaces[:,:,-1],vmin=Vmin,vmax=Vmax)
        
        # coverage
        plt.figure(figsize=(18,7))
        plt.plot(self.avg_save_times,np.ones([self.avg_save_times.shape[0]]),label='Layer 0')
        for i in range(self.coverage.shape[0]):
            plt.plot(self.avg_save_times,self.coverage[i,:],label='Layer '+str(i+1))
        plt.legend(bbox_to_anchor=(1.17, 1.0))
        
        # reflectivity plot
        plt.figure(figsize=(18,7))
        plt.plot(self.avg_save_times,self.reflectivity)
    
    


    def Save_Setup(self):
        """ Set up Save File """
        
        
        
        self.save_name = str(self.path / (self.file_name + '_Data.hdf5'))
        
        save_file = tables.open_file(self.save_name,mode='a')
        
        root = save_file.root
        save_file.create_group(root,'Time_Data','Time Data')
        save_file.create_group(root,'Surface_Data','Surface Data')
        save_file.create_group(root,'Coverage_Data','Coverage Data')
        save_file.create_group(root,'Reflectivity_Data','Reflectivity Data')
        
        save_file.close()
    
    
    
    def Save_Surfaces(self,surface,save_times,coverage,reflectivity):
        """ Save Surfaces and End Times to .hdf5 file """
        
        save_file = tables.open_file(self.save_name,mode='r+')
        
        # save time data
        save_file.create_array('/Time_Data','Times',save_times,'Save Times')
        
        # save coverage and reflectivity
        save_file.create_array('/Coverage_Data','Coverage',coverage,'Coverage')
        save_file.create_array('/Reflectivity_Data','Reflectivity',reflectivity,'Reflectivity')
        
        # save surface images
        length = len(str(self.analysis_times.shape[0]))     # number of digits in save id
        for i in range(surface.shape[2]):
            ii = '0'*(length-len(str(i))) + str(i)
            save_file.create_array('/Surface_Data','T'+ii,surface[:,:,i],'Surface'+ii)
            
    
        save_file.close()


    
        
        




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#####################
###   Functions   ###
#####################



        
        

        
# End of Module