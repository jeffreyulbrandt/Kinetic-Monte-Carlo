################################################################################################################
################################################################################################################
#########################                                                              #########################
#########################            PLD Kinetic Monte Carlo Pulse Module              #########################
#########################                                                              #########################
################################################################################################################
################################################################################################################



"""
Kinetic Monte Carlo - Pulse Module

Code for Deposition style simulation. Designed for pulsed deposition, but continuous deposition can be set
by simulating only one pulse with a wide pulse width.

Version: 3.3c

"""



# import python modules
from pathlib import Path   # this is actually used in this module
import numpy as np
import tables
import pickle
import csv

# import KMC modules
import KMC_Engine_v3p3 as KMC_Engine
import KMC_Miscellaneous_v3p3 as KMC_Misc
import KMC_Analysis_v3p3 as KMC_Analysis






#######################
###   Pulse Class   ###
#######################
#--------------------------------------------------------------------------------------------------------------------------------------------------


class Pulse:
    """
    Pulse Class: General Deposition style simulation. Controls deposition sequence. Saves simulation data after
    each pulse.
    
    
    Inputs
    ------
    Parameters:  Parameters Dictionary from Input Script
    path:        Pathlib path to the save folder
    
    
    Attributes
    ----------
    time:   simulation time in seconds
    deposited:  counter tracking number of deposition events
    end_of_pulse:  end of pulse time
    end_of_pulse_events:  track number of events at end of pulse
    end_of_pulse_events_tot: track cummulative events at end of pulse
    
    
    Methods
    -------
    Initialize:  Set-up simulation. Create KMC_Engine and On_the_Fly_Analysis objects
    Run_Pulse:  Run a deposition plus anneal simulation
    Save_Setup: Makes an hdf5 file to store simulation data
    Save_Event_Data: Saves all simulation events to hdf5 file
    Save_Lattice_Data: Saves lattice at specific times to hdf5 file
    Save_Surface_Data: Saves surface height at specific times to hdf5 file
    Save_Time_Data: Saves simulation times for lattice/surface data to hdf5 file
    Save_Summary: Save simulation parameters to pickle file and csv file
    
    """
    
    
    def __init__(self,Parameters,path):
        self.Parameters = Parameters
        
        self.path = path  # this is a path object
        
        # extract useful parameters from dictionary
        self.enable_printouts = self.Parameters['Enable_Print_Outs']
        self.n = self.Parameters['n']
        self.pulses = self.Parameters['Pulses']
        self.dwell_time = self.Parameters['Dwell_Time']  
        self.file_name = self.Parameters['Save_File_Name']        
        self.save = self.Parameters['Save_Output_File']
        self.save_lattice = self.Parameters['Save_Lattice']   
        self.save_surface = self.Parameters['Save_Surface']
        self.post = self.Parameters['Post_Anneal']
        self.post_time = self.Parameters['Post_Anneal_Time']    # post anneal step time
        
        self.Initialize()
        
    
    
    def Initialize(self):
        """ Initialize the Pulse. """
        # set up variables
        self.time = 0                      # simulation time     
        self.deposited = 0                 # counter for number of deposited particles
        self.end_of_pulse = 0              # end of pulse time
        self.end_of_pulse_events = 0       # keep track of events in pulse at end of pulse
        self.end_of_pulse_events_tot = 0   # keep track of total events at end of pulse
        
        if self.Parameters['Coverage'] == True or self.Parameters['Island_Analysis'] == True or self.Parameters['Structure_Factor'] == True:
                self.on_the_fly = True
        else:
            self.on_the_fly = False
            
        if self.Parameters['Dimension'] == '2D':
            self.on_the_fly = False   # no analysis available for 2D
        
        # Create File Paths
        self.save_name = str(self.path / (self.file_name + '_Data.hdf5'))                         # Path to Data File
        self.save_parameters = str(self.path / (self.file_name + '_Parameters.pkl'))              # Path to Pickle File
        self.save_csv_summary = str(self.path / (self.file_name + '_Summary.csv'))                # Path to Summary csv File
        self.save_analysis = str(self.path / (self.file_name + '_On_the_Fly_Analysis.hdf5'))      # Path to Analysis File
        
        # Add file paths to Parameters Dictionary
        self.Parameters['Save_Path_Data'] = self.save_name
        self.Parameters['Save_Path_Parameters'] = self.save_parameters
        self.Parameters['Save_Path_Summary'] = self.save_csv_summary
        if self.Parameters['Coverage'] == True or self.Parameters['Island_Analysis'] == True or self.Parameters['Structure_Factor'] == True:
            self.Parameters['Save_Path_Analysis'] = self.save_analysis
        
        # set up save files
        if self.save or self.save_lattice or self.save_surface == True:
            self.Save_Setup()
        self.length = len(str(int(self.Parameters['Pulses']*self.dwell_time/self.Parameters['Save_Times']))) + 1  # number of digits in save id
  
        # save summary pickle file
        self.Save_Summary()
        
        # save data times
        if self.post == False:
            self.save_times = np.arange(0,self.dwell_time*self.Parameters['Pulses'] + self.Parameters['Save_Times'], self.Parameters['Save_Times'])
        elif self.post == True:
            self.save_times = np.arange(0,self.dwell_time*self.Parameters['Pulses'] + self.post_time + self.Parameters['Save_Times'], self.Parameters['Save_Times'])
        self.save_index = 0    # index for analyze time
        self.save_time = []    # save actual times when save occurs
        
        # Set up Timers
        self.Engine_Timer = KMC_Misc.Timer()     # removed because timer too slow. Need cython timer
        self.Analysis_Timer = KMC_Misc.Timer()   # measures time to complete analysis step
        self.analysis_time_sum = 0    # keep track of total time of analysis steps per pulse
        
        # Create an instance of Deposition
        self.Deposition = KMC_Misc.Deposition(self.Parameters)
        
        # Create an instance of the KMC_Engine
        if self.Parameters['Dimension'] == '2D':
            self.Engine = KMC_Engine.KMC_Engine_2D(self.Parameters)
        elif self.Parameters['Dimension'] == '3D':
            self.Engine = KMC_Engine.KMC_Engine_3D(self.Parameters)
                       
        # Create an instance of On-the_Fly Analysis (only for 3D simulations)
        if self.Parameters['Dimension'] == '3D':        
            if self.on_the_fly == True:               
                self.Analysis = KMC_Analysis.On_the_Fly_Analysis(self.Parameters,self.path,self.Engine.lattice)   
                self.analysis_times = self.Analysis.analysis_times     # times to do analysis        
                self.analysis_index = 0      # index for on-the-fly analysis (sim time begins after t=0)
            
        # Run analysis on pre-deposition state (time = 0)        
        if self.on_the_fly == True:
            self.Analysis.Record_Time(self.time,self.analysis_index)
            self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)  # run the analysis
            self.analysis_index += 1

        
    
    def Run_Pulse(self,i):
        """ Execute a pulse sequence. """
        # reset the counters
        self.deposited = 0

        # Get Deposition Times and Coordinates
        self.dep_times, self.dep_coordinates = self.Deposition.Create_Deposition_Pulse(self.time)

        # Update End of Pulse Time
        self.end_of_pulse = (i+1)*self.dwell_time
                        
        # Deposit first particle to get things going
        self.time = self.Engine.First_Step(self.dep_times,self.dep_coordinates,self.end_of_pulse)
        self.deposited += 1
        
        # Create Progress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        if self.enable_printouts == True:
            self.Progress_Bar.Print(self.time-i*self.dwell_time,self.end_of_pulse)
            
        self.analysis_time_sum = 0  # reset cummulative analysis time
        

        ###################################
        ###   Start Depsotion Section   ###
        ###################################
        while self.deposited < self.n:            
            # Do Deposition Sequence
            self.time = self.Engine.Step()
                                              
            # Save Data at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time-i*self.dwell_time,self.dwell_time)
                                        
            # On-the-Fly Analysis
            if self.time >= self.analysis_times[self.analysis_index]:                
                self.Analysis_Timer.Timer_Start()               
                self.Analysis.Record_Time(self.time,self.analysis_index)    # record actual analysis time 
                self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)  # run the analysis
                
                self.analysis_index += 1                
                self.Analysis_Timer.Record_Time()  
                self.analysis_time_sum += self.Analysis_Timer.store_times[-1]
            
            # update deposited counter
            self.deposited = self.Engine.event_counters[1] - i*self.n  # subtract of deposited to date


        ################################
        ###   Start Anneal Section   ###
        ################################
        while self.time < self.end_of_pulse:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
            
            # Save Data at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time-i*self.dwell_time,self.dwell_time)
                        
            # On-the-Fly Analysis
            if self.time >= self.analysis_times[self.analysis_index]:
                self.Analysis_Timer.Timer_Start()  
                self.Analysis.Record_Time(self.time,self.analysis_index)    # record actual analysis time
                self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)  # run the analysis
                
                self.analysis_index += 1
                self.Analysis_Timer.Record_Time()
                self.analysis_time_sum += self.Analysis_Timer.store_times[-1]

        
        # save number current of events
        self.end_of_pulse_events = self.Engine.event_counters[0] - self.end_of_pulse_events_tot
        self.end_of_pulse_events_tot = self.Engine.event_counters[0]
              
        # save event data
        if self.save == True:
            self.Save_Event_Data()
    
    
    
    def Run_Pulse_No_Analysis(self,i):
        """ Execute a pulse sequence. This version does not include On-the-Fly Analysis. """
        # reset the counters
        self.deposited = 0

        # Get Deposition Times and Coordinates
        self.dep_times, self.dep_coordinates = self.Deposition.Create_Deposition_Pulse(self.time)

        # Update End of Pulse Time
        self.end_of_pulse = (i+1)*self.dwell_time
                        
        # Deposit first particle to get things going
        self.time = self.Engine.First_Step(self.dep_times,self.dep_coordinates,self.end_of_pulse)
        self.deposited += 1
        
        # Create Progress Bar
        self.Progress_Bar = KMC_Misc.Progress_Bar(prefix = 'Progress:', suffix = 'Complete', length = 50)
        if self.enable_printouts == True:
            self.Progress_Bar.Print(self.time-i*self.dwell_time,self.end_of_pulse)

        ###################################
        ###   Start Depsotion Section   ###
        ###################################
        while self.deposited < self.n:            
            # Do Deposition Sequence
            self.time = self.Engine.Step()
                                              
            # Save Data at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time-i*self.dwell_time,self.dwell_time)

            # update deposited counter
            self.deposited = self.Engine.event_counters[1] - i*self.n  # subtract of deposited to date


        ################################
        ###   Start Anneal Section   ###
        ################################
        while self.time < self.end_of_pulse:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
            
            # Save Data at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time-i*self.dwell_time,self.dwell_time)

        
        # save number current of events
        self.end_of_pulse_events = self.Engine.event_counters[0] - self.end_of_pulse_events_tot
        self.end_of_pulse_events_tot = self.Engine.event_counters[0]
              
        # save event data
        if self.save == True:
            self.Save_Event_Data()
            
            
            
    def Run_Post_Anneal(self):
        """ Runs the post anneal step """

        # End of Simulation Time
        final_time = self.pulses*self.dwell_time+self.post_time        
        self.Engine.Step_Post_Anneal_Start(final_time)
        
        while self.time < final_time:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
            
            # Save Data at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time-self.pulses*self.dwell_time,self.post_time)
                        
            # On-the-Fly Analysis
            if self.time >= self.analysis_times[self.analysis_index]:
                self.Analysis_Timer.Timer_Start()  
                self.Analysis.Record_Time(self.time,self.analysis_index)    # record actual analysis time
                self.Analysis.Do_Analysis(self.Engine.lattice,self.analysis_index)  # run the analysis
                
                self.analysis_index += 1
                self.Analysis_Timer.Record_Time()
                self.analysis_time_sum += self.Analysis_Timer.store_times[-1]
        
        # save number current of events
        self.end_of_pulse_events = self.Engine.event_counters[0] - self.end_of_pulse_events_tot
        self.end_of_pulse_events_tot = self.Engine.event_counters[0]
              
        # save event data
        if self.save == True:
            self.Save_Event_Data()
    
    
    
    def Run_Post_Anneal_No_Analysis(self):
        """ Runs the post anneal step with out on-the-fly analysis """
        
        # End of Simulation Time
        final_time = self.pulses*self.dwell_time+self.post_time        
        self.Engine.Step_Post_Anneal_Start(final_time)
        
        while self.time < final_time:
            
            # Do Anneal Sequence
            self.time = self.Engine.Step_Anneal()
            
            # Save Data at specific simulation times
            if self.time >= self.save_times[self.save_index]:
                self.save_time.append(self.time)     # actual save time
                self.Engine.Sim_Stats.Record_Data(self.time,self.Engine.process_counters,self.Engine.event_counters,self.Engine.ni)   # Sim Stats
                if self.save_lattice == True:
                    self.Save_Lattice_Data(self.save_index)
                if self.save_surface == True:
                    self.Save_Surface_Data(self.save_index)
                self.save_index += 1
                
                # Update Progres Bar (try it here first)
                if self.enable_printouts == True:
                    self.Progress_Bar.Print(self.time-self.pulses*self.dwell_time,self.post_time)
        
        # save number current of events
        self.end_of_pulse_events = self.Engine.event_counters[0] - self.end_of_pulse_events_tot
        self.end_of_pulse_events_tot = self.Engine.event_counters[0]
              
        # save event data
        if self.save == True:
            self.Save_Event_Data()
    
    
    
    def Save_Setup(self):
        """ Set up a hdf5 save file. """
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
        

    
    def Save_Event_Data(self):
        """ Save Event Data to File. """
        
        # get the data
        if self.Parameters['Dimension'] == '2D':
            x,z,save_id,save_times = self.Engine.Save_Events()
            save_coordinates = np.transpose(np.vstack((x,z)))            
        elif self.Parameters['Dimension'] == '3D':
            x,y,z,save_id,save_times = self.Engine.Save_Events()
            save_coordinates = np.transpose(np.vstack((x,y,z)))
        
        # open the save file
        save_file = tables.open_file(self.save_name,mode='r+')
        
        # append data to the save file
        save_file.root.Event_Data.Time.append(save_times)
        save_file.root.Event_Data.ID.append(save_id)        
        save_file.root.Event_Data.X.append(save_coordinates)
        
        # close the file
        save_file.close()
        
     
        
    def Save_Lattice_Data(self,i):
        """ Flatten and Compress lattice data and save to file. """
        
        # Flatten and Compress (my own method)
        data = self.Engine.lattice
        flat = data.flatten()  # flatten the array
        save_data = np.trim_zeros(flat, 'b')   # trim off the trailing zeros
        
        # create string for naming array in order
        ii = '0'*(self.length-len(str(i))) + str(i)
        
        save_file = tables.open_file(self.save_name,mode='r+')
        save_file.create_array('/Lattice_Data','T'+ii,save_data,'Lattice'+ii)
               
        save_file.close()
    
    
    
    def Save_Surface_Data(self,i):
        """ Save surface only data. Smallest file, but lose layer info. """

        if self.Parameters['Dimension'] == '2D':
            save_data = np.copy(self.Engine.lattice)
            save_data = np.argmax(save_data,axis=1)
            save_data = save_data.astype(np.uint8)  # max layer is 256 for this dtype
        elif self.Parameters['Dimension'] == '3D':
            save_data = np.copy(self.Engine.lattice)
            save_data = np.argmax(save_data,axis=2)
            save_data = save_data.astype(np.uint8)  # max layer is 256 for this dtype            
        
        # create string for naming array in order
        ii = '0'*(self.length-len(str(i))) + str(i)
        
        Save_File = tables.open_file(self.save_name,mode='r+')
        Save_File.create_array('/Surface_Data','T'+ii,save_data,'Surface'+ii)
        Save_File.close()
        
    
    
    def Save_Time_Data(self):
        """ Save times when lattice data was collected. """
        self.save_time = np.array(self.save_time)
        save_file = tables.open_file(self.save_name,mode='r+')        
        save_file.create_array('/Time_Data','Times',self.save_time,'Save Times')        
        save_file.close()
 
     
        
    def Save_Summary(self):
        """ Save Simulation Parameters using pickle and a csv file. """
        
        # pickle file
        file = open(self.save_parameters, 'wb')         
        pickle.dump(self.Parameters, file)       
        file.close()
        
        # csv file
        header = ['Parameter','Value']                
        with open(self.save_csv_summary, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            for key, value in self.Parameters.items():
                line = [key, value]
                writer.writerow(line)
  
    
    
    
    
    
# End of Module