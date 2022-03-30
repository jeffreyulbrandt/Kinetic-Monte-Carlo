################################################################################################################
################################################################################################################
#########################                                                              #########################
#########################            PLD Kinetic Monte Carlo Pulse Module              #########################
#########################                                                              #########################
################################################################################################################
################################################################################################################



"""
Kinetic Monte Carlo - Pulse Module

Code for Deposition style simulation. Designed for pulsed deposition (PLD), but continuous deposition (MBE) is also supported. 

Version: 3.4.1b

"""



# import python modules
from pathlib import Path   # this is actually used in this module, ignore the warning
import numpy as np

# import KMC modules
import KMC_Engine_v3p4p1 as KMC_Engine
import KMC_Deposition_v3p4p1 as KMC_Dep
import KMC_Miscellaneous_v3p4p1 as KMC_Misc
import KMC_Analysis_v3p4p1 as KMC_Analysis




#######################
###   Pulse Class   ###
#######################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------


class Pulse:
    """
    Pulse Class: General Deposition style simulation for both PLD and MBE.
    
    Inputs
    ------
    Parameters:  Parameters Dictionary from Input Script
    path:        Pathlib path to the save folder
    
    Attributes
    ----------
    lattice:    Simulation lattice.
    
    Methods
    -------
    
    
    """
    
    
    def __init__(self,Parameters,path,seed):
        self.Parameters = Parameters
        
        self.path = path   # this is a path object
        self.seed = seed   # seed for RNG
        
        # extract useful variables from Parameters dictionary
        self.enable_plots = self.Parameters['Enable_Plots']
        self.enable_printouts = self.Parameters['Enable_Print_Outs']
        
        self.save = self.Parameters['Save_All_Events']      
        self.save_lattice = self.Parameters['Save_Lattice']   
        self.save_surface = self.Parameters['Save_Surface']
        
        self.save_dt = self.Parameters['Save_Time_Step']
        self.save_w_analysis = self.Parameters['Save_with_Analysis']
        
        self.file_name = self.Parameters['Save_File_Name'] 
        
        self.dim = self.Parameters['Dimension']
        self.lx = self.Parameters['Lx']
        self.ly = self.Parameters['Ly']
        self.depth = self.Parameters['depth']
        
        self.dwell_time = self.Parameters['Dwell_Time']
        self.pulses = self.Parameters['Pulses']
        
        self.post = self.Parameters['Post_Anneal']
        self.post_anneal_time = self.Parameters['Post_Anneal_Time']
        
        self.analysis_style = self.Parameters['Analysis_Style']
        self.analysis_dt = Parameters['Analysis_Delta_Time']
        self.analysis_step = Parameters['Analysis_Delta_Step']
        self.cov = self.Parameters['Coverage']
        self.island = self.Parameters['Island_Analysis']
        self.structure_factor = self.Parameters['Structure_Factor']
        self.on_the_fly_save = self.Parameters['On_the_Fly_Save']
        self.avg_results = self.Parameters['Average_Results']
        
        self.Initialize()
        
        
        
    def Initialize(self):
        """ Initialize the Pulse Object """
        
        # Create new variables
        self.time = 0                        # current simulation time
        self.step = 0                        # current simulation step
        self.end_of_pulse = 0                # current end of deposition pulse time, starts at zero 
        self.end_of_pulse_events = 0         # keep track of events in pulse at end of pulse
        self.end_of_pulse_events_tot = 0     # keep track of total events at end of pulse        
        self.analysis_time_sum = 0           # total time spent doing analysis in pulse
                
        # Set up timers
        self.Analysis_Timer = KMC_Misc.Timer()   # measures time to complete analysis step
                
        # Create File Paths
        self.save_name = str(self.path / (self.file_name + '_Data.hdf5'))                         # Path to Data File
        self.save_parameters = str(self.path / (self.file_name + '_Parameters.pkl'))              # Path to Pickle File
        self.save_csv_summary = str(self.path / (self.file_name + '_Summary.csv'))                # Path to Summary csv File
        self.save_analysis = str(self.path / (self.file_name + '_On_the_Fly_Analysis.hdf5'))      # Path to Analysis File
                
        # Add file paths to Parameters Dictionary
        self.Parameters['Save_Path_Data'] = self.save_name
        self.Parameters['Save_Path_Parameters'] = self.save_parameters
        self.Parameters['Save_Path_Summary'] = self.save_csv_summary
        if self.cov == True or self.island == True or self.structure_factor == True:
            self.Parameters['Save_Path_Analysis'] = self.save_analysis
                    
        # Set up save files
        if self.save or self.save_lattice or self.save_surface == True:
            self.save_data_file = KMC_Misc.Save(self.Parameters, self.path, 'Data')
        self.length = len(str(int(self.pulses*self.dwell_time/self.save_dt))) + 1    # number of digits in save id
      
        # Save summary pickle file and csv file
        KMC_Misc.Save_Parameters_Pickle(self.Parameters,self.save_parameters)
        KMC_Misc.Save_Parameters_CSV(self.Parameters,self.save_csv_summary)
                
        # Set up save data
        self.save_index = 0             # for save data times
        self.save_times_actual = []     # list to record the actual save times
        
        if self.post == False:
            self.save_times = np.arange(self.save_dt,self.dwell_time*self.pulses + 2*self.save_dt, self.save_dt)
        elif self.post == True:
            self.save_times = np.arange(self.save_dt,self.dwell_time*self.pulses + self.post_anneal_time + 2*self.save_dt, self.save_dt)
               
        # Create an instance of Deposition
        self.Deposition = KMC_Dep.Deposition(self.Parameters)
                
        # Create an instance of KMC_Engine
        self.Engine = KMC_Engine.KMC_Engine(self.Parameters,self.seed)
        
        # Check for On-the-Fly analysis
        if self.cov == True or self.island == True or self.structure_factor == True:
            self.on_the_fly = True
        else:
            self.on_the_fly = False
            
        if self.dim == '2D':
            self.on_the_fly = False   # no on-the-fly analysis available for 2D simulations yet
                  
        # Create an instance of On-the_Fly Analysis (only for 3D simulations) Future: Make Cython only
        if self.on_the_fly == True and self.dim == '3D':             
            self.Analysis = KMC_Analysis.On_the_Fly_Analysis(self.Parameters,self.path,self.Engine.lattice)   
            self.analysis_steps = []     # to save the steps when analysis happens
            self.analysis_times = self.Analysis.analysis_times     # times to do analysis
            self.analysis_times = np.append(self.analysis_times,[1e10])   # add an extra value to prevent index error
            self.analysis_index = 0      # index for on-the-fly analysis (sim time begins after t=0)
            
            self.analysis_data_file = KMC_Misc.Save(self.Parameters, self.path, 'Analysis')
                
        # Run analysis on pre-deposition state (time = 0)        
        if self.on_the_fly == True:
            self.Analysis.Record_Time(self.time,self.analysis_index)
            self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)  # run the analysis
            self.analysis_index += 1
            

        
        
    def Run_Pulse(self,i):
        """ Run a Pulse either to a step or a time. """
        
        # Get Deposition Times and Coordinates
        self.dep_times, self.dep_coordinates = self.Deposition.Create_Deposition_Pulse(i*self.dwell_time)  
        
        # Update End of Pulse Time
        self.end_of_pulse = (i+1)*self.dwell_time    
        
        # Deposit first particle of pulse to get things going
        self.Engine.Run_First_Step(self.dep_times,self.dep_coordinates)
        
        # Create Progress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        if self.enable_printouts == True:
            self.Progress_Bar.Print(self.time-i*self.dwell_time,self.end_of_pulse)
        
        # reset cummulative analysis time
        self.analysis_time_sum = 0  
                
        # Run to time or to step
        if self.analysis_style == 'time':
            if self.on_the_fly == True:
                self.Pulse_Time(i)
            elif self.on_the_fly == False:
                self.Pulse_Time_No_Analysis(i)
        elif self.analysis_style == 'step':
            self.Pulse_Step(i)
        else:
            print ('Not a valid analysis style. Please choose either "time" or "step".')
                    
        # save number current of events
        event_counters, process_counters = self.Engine.Return_Counters()
        self.end_of_pulse_events = event_counters[0] - self.end_of_pulse_events_tot
        self.end_of_pulse_events_tot = event_counters[0]
              
        # save event data
        if self.save == True:
            self.Save_Event_Data()
            
       
    
    
    def Pulse_Time(self,i):
        """ Execute a pulse sequence until a specific time. """
        
        pause_time = 0  # this is just to start the while loop
        
        # Run the full pulse loop
        while pause_time < self.end_of_pulse:

            # Select next time to stop
            next_analysis_time = self.analysis_times[self.analysis_index]
            next_save_time = self.save_times[self.save_index]
            
            if next_analysis_time > next_save_time:
                pause_time = next_save_time
                which_analysis = 0
            elif next_analysis_time < next_save_time:
                pause_time = next_analysis_time
                which_analysis = 1
            elif next_analysis_time == next_save_time:
                pause_time = next_analysis_time   # if they're the same it doesn't matter which
                which_analysis = 2
            
            # Run a block of steps and then pause
            self.time, self.step, error_code = self.Engine.Run_until_Time(end_time = pause_time, anneal=0)
            
            # Do the anaysis
            if which_analysis == 0:
                self.Save_Step()
                if self.enable_printouts == True:    # Update Progres Bar (try it here first)
                    self.Progress_Bar.Print(self.time-i*self.dwell_time,self.dwell_time)
            elif which_analysis == 1:
                self.Analysis_Step()
            elif which_analysis == 2:
                self.Save_Step()
                if self.enable_printouts == True:    # Update Progres Bar (try it here first)
                    self.Progress_Bar.Print(self.time-i*self.dwell_time,self.dwell_time)
                self.Analysis_Step()
        
        
        
    def Pulse_Time_No_Analysis(self,i):
        """ Execute a pulse sequence until a specific time. This version only has save stops. """
        
        pause_time = 0  # this is just to start the while loop
        
        # Run the full pulse loop
        while pause_time < self.end_of_pulse:

            # Select next time to stop
            pause_time = self.save_times[self.save_index]
            
            # Run a block of steps and then pause
            self.time, self.step, error_code = self.Engine.Run_until_Time(end_time=pause_time, anneal=0)
            
            # Do the anaysis
            self.Save_Step()
            if self.enable_printouts == True:    # Update Progres Bar (try it here first)
                self.Progress_Bar.Print(self.time-i*self.dwell_time,self.dwell_time)

                           
            
    def Pulse_Step(self,i):
        """ Execute a pulse sequence until a specific step. Also check for an end time. """
        
        # set end of deposition time to force a stop to analyze data
        end_of_dep = self.dep_times[-1]
        
        # reset pause step so simulation is delta step after start of pulse
        event_counters, process_counters = self.Engine.Return_Counters()
        pause_step = event_counters[0]     # current simulation step
        
        # run pulse until end of dep
        while self.time < end_of_dep:
            
            # Decide what step or time to run to
            pause_step += self.analysis_step
            next_save_time = self.save_times[self.save_index]
            
            if next_save_time <= end_of_dep:
                stop_time = next_save_time
            else:
                stop_time = end_of_dep
            
            # Run a block of steps and then pause
            self.time, self.step, error = self.Engine.Run_until_Step(pause_step, stop_time, anneal=0)
            
            # Do the anaysis     
        
        # Run the rest of the pulse
        while self.time < self.end_of_pulse:
            
            # Decide what step to run to
            pause_step = self.analysis_step
            next_save_time = self.save_times[self.save_index]
            if next_save_time <= self.end_of_pulse:
                stop_time = next_save_time
            else:
                stop_time = self.end_of_pulse
            
            # Run a block of steps and then pause
            self.time, self.step, error = self.Engine.Run_until_Step(pause_step, stop_time, anneal=1)
            
            
    
    def Run_Post_Anneal(self):
        """ Run the post anneal.  """
        
        # reset cummulative analysis time
        self.analysis_time_sum = 0  
              
        # Run to time or to step
        if self.analysis_style == 'time':
            if self.on_the_fly == True:
                self.Post_Anneal_Time()
            elif self.on_the_fly == False:
                self.Post_Anneal_Time_No_Analysis()
        elif self.analysis_style == 'step':
            self.Post_Anneal_Step()
        else:
            print ('Not a valid analysis style. Please choose either "time" or "step".')
            
        
        # save number current of events
        event_counters, process_counters = self.Engine.Return_Counters()
        self.end_of_pulse_events = event_counters[0] - self.end_of_pulse_events_tot
        self.end_of_pulse_events_tot = event_counters[0]
              
        # save event data
        if self.save == True:
            self.Save_Event_Data()
    
    
        
    def Post_Anneal_Time(self):
        """ Runs the post anneal step indpendent of the regular pulse sequence. Analysis by time. """
        
        # End of Simulation Time
        final_time = self.pulses*self.dwell_time + self.post_anneal_time
        pause_time = 0
        
        # Run the KMC until reaching final time
        while pause_time < final_time:
            
            # Select next time to stop
            next_analysis_time = self.analysis_times[self.analysis_index]
            next_save_time = self.save_times[self.save_index]
            
            if next_analysis_time > next_save_time:
                pause_time = next_save_time
                which_analysis = 0
            elif next_analysis_time < next_save_time:
                pause_time = next_analysis_time
                which_analysis = 1
            elif next_analysis_time == next_save_time:
                pause_time = next_analysis_time   # if they're the same it doesn't matter which
                which_analysis = 2
            
            # Run a block of steps and then pause
            self.time, self.step, error_code = self.Engine.Run_until_Time(end_time = pause_time, anneal=1)
            
            # Do the anaysis
            if which_analysis == 0:
                self.Save_Step()
                if self.enable_printouts == True:    # Update Progres Bar (try it here first)
                    self.Progress_Bar.Print(self.time-self.pulses*self.dwell_time,self.post_anneal_time)
            elif which_analysis == 1:
                self.Analysis_Step()
            elif which_analysis == 2:
                self.Save_Step()
                if self.enable_printouts == True:    # Update Progres Bar (try it here first)
                    self.Progress_Bar.Print(self.time-self.pulses*self.dwell_time,self.post_anneal_time)
                self.Analysis_Step()
            
    

    def Post_Anneal_Time_No_Analysis(self):
        """ Runs the post anneal step with out on-the-fly analysis. """
        
        # End of Simulation Time
        final_time = self.pulses*self.dwell_time + self.post_anneal_time
        pause_time = 0
        
        # Run the KMC until reaching final time
        while pause_time < final_time:
            
            # Select next time to stop
            pause_time = self.save_times[self.save_index]
            
            # Run a block of steps and then pause
            self.time, self.step, error_code = self.Engine.Run_until_Time(end_time=pause_time, anneal=1)
            
            # Do the anaysis
            self.Save_Step()
            if self.enable_printouts == True:    # Update Progres Bar (try it here first)
                self.Progress_Bar.Print(self.time-self.pulses*self.dwell_time,self.post_anneal_time)
            
            
            
    def Post_Anneal_Step(self):
        """ Runs the post anneal with anlysis by step. Future. """
        
        # End of Simulation Time
        final_time = self.pulses*self.dwell_time + self.post_anneal_time
        
        # Run the KMC until reaching final time
        while self.time < final_time:
            
            # Do Anneal Sequence
            self.time,self.step,error = self.Engine.Run_until_Time(final_time, anneal=1)
    
            # While paused, dave data and/or do analysis
            
            
            
    def Save_Step(self):
        """ Save Data sequence. """
        
        self.save_times_actual.append(self.time)     # save the actual save time
        self.Engine.Save_Sim_Stats()                 # record snapshot of simulation statistics
        
        if self.save_lattice == True:
            self.Save_Lattice_Data(self.save_index)                  
        if self.save_surface == True:
            self.Save_Surface_Data(self.save_index)
                    
        self.save_index += 1
                
    
            
    def Analysis_Step(self):
        """ Do Analysis sequence. """
        
        self.Analysis_Timer.Timer_Start()               
        self.Analysis.Record_Time(self.time,self.analysis_index)             # store the actual analysis time 
        self.analysis_steps.append(self.step)                                # store the current step
        self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)   # run the analysis
                               
        self.Analysis_Timer.Record_Time()  
        self.analysis_time_sum += self.Analysis_Timer.store_times[-1]   

        self.analysis_index += 1 

    

            
    
    
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
        
        
        
    def Save_Analysis_Data(self):
        """ Save analysis data at end of simulation """
        
        # get data from Analysis Module
        Save_Data = self.Analysis.Return_Data()
        
        # save to file
        self.analysis_data_file.Save_Analysis(Save_Data)

    
        
        




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#####################
###   Functions   ###
#####################



        
        

        
# End of Module