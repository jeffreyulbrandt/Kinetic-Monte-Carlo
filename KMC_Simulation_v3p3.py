###############################################################################################################
###############################################################################################################
#########################                                                             #########################
#########################            PLD Kinetic Monte Carlo Main Module              #########################
#########################                                                             #########################
###############################################################################################################
###############################################################################################################


"""
Version: 3.3c



Version History:
-----------------

v0.0 - JGU - 8/10/2019 - Initial test. Python based Procedural style. 1+1 only

v1.0 - JGU - 8/22/2019 - Restructurd to object-oriented style. Still only for 1+1.
v1.1 - JGU - 9/1/2019 - Improved object-oriented structure for efficiency. Still fully Python based.

v2.0 - JGU - 9/3/2019 - Added 3D Solver for 2+1 simulations
v2.1 - JGU - 9/15/2019 - Removed duplicate for loop in Solver/Lattice. Simplify Lattice search code. Added On-the-Fly Analysis.
v2.2 - JGU - 12/19/2019 - Add Uphill diffusion. Add Downward Funneling process. Passive Substrates.

v3.0 - JGU - 2/1/2020 - Convert Lattice to Cython. Speed improvement was a factor of 2. Not bad, but not good enough.
v3.1 - JGU - 2/5/2020 - Complete overhall of Class structure to implement full cythonization. ~160 times faster than python version!
v3.2 - JGU - 6/12/2020 - Mutli-Sim. Average Analysis results. Structure Factor Analysis.
v3.3 - JGU - 6/18/2020 - Restructured Cython modules. Imrpoved circular averaging in Structure Factor calculation. Added new analysis features.


General Info:
-------------
    
This simulation is designed to run a multiscale simulation of Pulsed Laser Depsotion. This module contains 
the complete code needed to execute the KMC simulation. Fast and Slow dynamic processes are carried out 
independently.


Notation Guide:
---------------
All functions, methods, and objects should be named with captial letters with words seperated by an underscore. Variables and class 
attributes should be lowercase with words seperated by underscores. (exception if name is one letter because it can look weird)


Modules:
--------
    - KMC_Simulation:     Simulation Class (python only code)
    - KMC_Pulse:          Pulse Class (python only code)    
    - KMC_Engine:         Main KMC algorithm (python code)
    - KMC_Proceeses_3D:   Lattice algorithms (cython code)
    - KMC_Solver:         KMC temporal algorithms (cython code)
    - KMC_Miscellaneous:  Misc Functions, Rate_Catalog Class, Deposition Class, Timer Class, Statistics Class (python only code)
    - KMC_Analysis:       On-the-Fly Analysis, Coverage Analysis, Island Analysis, Structure Factor (python and cython code)


Classes:
--------
    - Simulation:           Main Top-level Object, controls total simulation. Start from Input Script.
    - Pulse:                Mid-Level control over individual pulses. Also handles saving which is done after each pulse.
    - KMC_Engine_2D(3D):    Low-level python class. Contain data arrays and creates cython classes.
    - Solver:               Low-level Cython Extension. Maintains process lists and selects processes and calculates time steps.
    - Processes:            Low-level Cython Extension for lattice calculations. Local Neighborhood Serach algorithms.
    - Process_Catalog:      Calculate elementary rates and moves.
    - Deposition:           Calculate deposition times and coordinates. Incident Energy and Angles.
    - Substrate_Generator:  Create active and passive substrate feature such as islands and steps.
    - Save:                 Creates .hdf5 files for storing raw data and analyzed data.
    - Timer:                Calculates time to execute sections of the simulation.
    - Statistics:           Stores simuation statistics at fixed time intervals.
    - Progress_Bar:         Real time feedback of simulation progress.
    - On_the_Fly_Analysis:  Analysis of Data. Coverage, Island Distributions, Structure Factor.

    
Input Script:   
-------------

To run a simulation, all Parameters must be set in an input script. Once parameters are set, the simulation
can be run by calling the simulation class and passing it the Parameters dictionary

      


"""


# import python modules
import numpy as np
from pathlib import Path
import datetime 
import time
import csv

# import KMC modules
import KMC_Pulse_v3p3 as KMC_Pulse
import KMC_Miscellaneous_v3p3 as KMC_Misc




############################
###   Simulation Class   ###
############################
#--------------------------------------------------------------------------------------------------------------------------------------------------


class Simulation:
    """
    Simulation Class: Top-level object controling full simulation. Mostly I/O related functions.
    
    Inputs
    ------
    Parameters:   Parameters Dictionary from Input script
    
    Attributes
    ----------
    
    
    Methods
    -------
    Multi_Sim:           Loop for running multiple simulations. Controls averaging of results.
    Start_Simulation:    Initialize the simulation. Create instance of Pulse.
    Execute_Pulse:       Main sequence for a deposition simulation. Loop for all pulses.
    Execute_Diffusion:   Main sequence for diffusion only simulation.
    End_Simulation:      Finish simulation. Save Data.
    results:             Return simulation data to input script.
    
    """
    
    def __init__(self,Parameters):
        self.Parameters = Parameters
        
        # Extract useful parameters from dictionary
        self.num_sims = self.Parameters['Number_of_Simulations']
        self.enable_printouts = self.Parameters['Enable_Print_Outs']
        self.pulses = self.Parameters['Pulses']
        self.post_anneal = self.Parameters['Post_Anneal']
        
        if self.Parameters['Coverage'] == True or self.Parameters['Island_Analysis'] == True or self.Parameters['Structure_Factor'] == True:
                self.on_the_fly = True
        else:
            self.on_the_fly = False
        
#        self.Run()
        
    
    
    def Run(self):
        """ Run the Multi-simulation Loop. """
        
        # Set up Save Directories, Paths, and File Names
        self.file_name_base = Set_File_Names(self.Parameters)
        self.project_folder = self.Parameters['Project_Name']
        self.sim_folder = Date_Time_Stamp() + self.Parameters['Simulation_Name']
        self.root_dir,self.results_dir,self.proj_dir,self.sim_dir = Make_Directory(self.project_folder,self.sim_folder)
        
        # Run the simulation loop
        for i in range(self.num_sims):
            # Run a simulation
            self.Start_Simulation(i+1)
            
            # after first simulation, set up averaged results save object
            if i == 0:
                if self.Parameters['Average_Results'] == True and self.on_the_fly == True:
                    self.Average_Results_File = KMC_Misc.Save(self.Parameters,self.sim_dir)  # create save object
                    self.Average_Results_File.Add_Initial_Data(self.Pulse.Analysis.analysis_times,self.Pulse.Analysis.actual_times,self.Pulse.Analysis.cov,
                                                               self.Pulse.Analysis.refl,self.Pulse.Analysis.island_data,self.Pulse.Analysis.island_index,
                                                               self.Pulse.Analysis.structure_factor,self.Pulse.Analysis.rms,self.Pulse.Analysis.avg_height)
            
            # after second simulation, start averaging results
            if i > 0:
                if self.Parameters['Average_Results'] == True and self.on_the_fly == True:
                    self.Average_Results_File.Add_New_Data(self.Pulse.Analysis.actual_times,self.Pulse.Analysis.cov,self.Pulse.Analysis.refl,
                                                           self.Pulse.Analysis.island_data,self.Pulse.Analysis.island_index,self.Pulse.Analysis.structure_factor,
                                                           self.Pulse.Analysis.rms,self.Pulse.Analysis.avg_height)
                    
#            # Print end of simulation note
#            print ('Sim ',str(i),' Complete.')
            
        # save averaged data to file
        if self.Parameters['Average_Results'] == True and self.on_the_fly == True:
            self.Average_Results_File.Save_Data()
                    
        # maybe save a csv parameters file at simulation level
        self.save_csv_summary = str(self.sim_dir / (self.file_name_base + '_Simulation_Parameters.csv'))   # Path to Summary csv File
        Save_Summary(self.Parameters,self.save_csv_summary)
        
        # sleep for two seconds to prevent same datetime stamp in very fast simulations
        time.sleep(2)
        

        
    def Start_Simulation(self,sim_number):       
        """ Initialize a Simulation. """
        
        # Set up simulation timers and progress bar
        self.Sim_Timer = KMC_Misc.Timer()  # Total Simulation Time
        self.Pulse_Timer = KMC_Misc.Timer()   # Simulation Time per Pulse
        self.Post_Timer = KMC_Misc.Timer()   # Post Anneal Time
        
        self.Sim_Timer.Timer_Start()
        
        # Add deposition rate info to Parameters Dictionary        
        self.dep_rate_avg, self.dep_rate_peak = KMC_Misc.Deposition(self.Parameters).Deposition_Rates()
        self.Parameters['Dep_Rate_Avg'] = self.dep_rate_avg
        self.Parameters['Dep_Rate_Peak'] = self.dep_rate_peak
        
        # Set up Sub directory
        self.save_folder = self.Parameters['Simulation_Name'] + '_Sim' + str(sim_number)
        self.save_dir = Make_Sub_Directory(self.sim_dir,self.save_folder)
        
        self.save_path_final_lattice = str(self.save_dir / (self.file_name_base + '_Final_Lattice.npy'))
        self.save_path_statistics = str(self.save_dir / (self.file_name_base + '_Simulation_Statistics.csv'))
        
        # Add directory and path info to Parameters Dictionary
        self.Parameters['Save_File_Name'] = self.file_name_base
        self.Parameters['Save_Path_Project'] = str(self.proj_dir)
        self.Parameters['Save_Path_Simulation'] = str(self.sim_dir)
        self.Parameters['Save_Path_Save'] = str(self.save_dir)
        self.Parameters['Save_Path_Final_Lattice'] = self.save_path_final_lattice
        self.Parameters['Save_Path_Final_Statistics'] = self.save_path_statistics
        
        # Print simulation start info
        if self.enable_printouts == True:
            Print_Start_Info(self.Parameters,sim_number)
            
        # On-the-Fly Analysis?
        if self.Parameters['Coverage'] == True or self.Parameters['Island_Analysis'] == True or self.Parameters['Structure_Factor'] == True:
            self.on_the_fly = True
        else:
            self.on_the_fly = False
            
        if self.Parameters['Dimension'] == '2D':
            self.on_the_fly = False   # no analysis available for 2D
            
        # simulation level statistics variables    
        self.tot_analysis_time = 0   # variable to keep track of total analysis time
        self.tot_kmc_time = 0        # variable to keep track of total simulation time
        self.time_ratio = 0          # percent of time spent in analysis
        
        # Run the Simulation according to the Simulation_Type
        if self.Parameters['Simulation_Type'] == 'Deposition':            
            self.Pulse = KMC_Pulse.Pulse(self.Parameters,self.save_dir)    # call an instance of Pulse
            self.Execute_Pulse(sim_number)    # start the simulation
        
        elif self.Parameters['Simulation_Type'] == 'Diffusion':
            print ('Diffusion Simulation does not exist yet. Try "Deposition" instead.')
            self.Execute_Diffusion()    # start the simulation
        

        
    def Execute_Pulse(self,sim_number):
        """ Main Sequence for Deposition Style Simulation. """
        
        for i in range(self.pulses):
            # Start the Pulse Timer
            self.Pulse_Timer.Timer_Start()
            
            # Print Start-of-Pulse Message
            if self.enable_printouts == True:
                Print_Start_of_Pulse(i,sim_number,self.num_sims)
            
            # Simulate a Pulse
            if self.on_the_fly == True:
                self.Pulse.Run_Pulse(i)
            else:
                self.Pulse.Run_Pulse_No_Analysis(i)
            
            # record Pulse Time
            self.Pulse_Time = self.Pulse_Timer.Record_Time()
            
            # add up siumation times
            a_time = self.Pulse.analysis_time_sum
            kmc_time = self.Pulse_Timer.store_times[-1]
            
            self.tot_analysis_time += a_time
            self.tot_kmc_time += kmc_time
            
            # Print a End-of-Pulse update
            if self.enable_printouts == True:                
                Print_End_of_Pulse(i,self.Pulse_Timer.store_times[-1],self.Pulse.end_of_pulse_events_tot,self.Pulse.end_of_pulse_events,a_time)
                
                
        # Run a post anneal step?
        if self.post_anneal == True:
            # print post anneal message
            Print_Start_of_Anneal()
            
            # run the anneal step
            self.Post_Timer.Timer_Start()
            if self.on_the_fly == True:
                self.Pulse.Run_Post_Anneal()
            else:
                self.Pulse.Run_Post_Anneal_No_Analysis()
            self.Post_Timer.Record_Time()
                
            # print end of anneal update           
            a_time = self.Pulse.analysis_time_sum
            kmc_time = self.Post_Timer.store_times[-1]
            
            self.tot_analysis_time += a_time
            self.tot_kmc_time += kmc_time
            
            if self.enable_printouts == True:                
                Print_End_of_Anneal(self.Post_Timer.store_times[-1],self.Pulse.end_of_pulse_events_tot,self.Pulse.end_of_pulse_events,a_time)
            
        
        # Finish simulation
        self.End_Simulation()
        
             
        
    def Execute_Diffusion(self):
        """ Main Sequence for Diffusion Style Simulation. """
        pass
        

        
    def End_Simulation(self):
        """ End the simulation. """
        
        # Stop simulation timer
        self.sim_time = self.Sim_Timer.Record_Time()
        
        # Save the statistics file
        self.Pulse.Engine.Sim_Stats.Save_Stats(self.save_path_statistics)
        
        # Save On-the-Fly Analysis results
        if self.Parameters['On_the_Fly_Save'] == True:
            if self.on_the_fly == True:
                self.Pulse.Analysis.Save()
        
        # Save save-time data
        if self.Parameters['Save_Lattice'] == True:
            self.Pulse.Save_Time_Data()
        
        # Compute Average Simulation Times
        time1_avg, time1_std = self.Sim_Timer.Average()
        time2_avg, time2_std = self.Pulse_Timer.Average()
        if self.on_the_fly == True:
            time3_avg, time3_std = self.Pulse.Analysis_Timer.Average()
        elif self.on_the_fly == False:
            time3_avg = 0
        
        self.time_data = (KMC_Misc.Time_Convert(time1_avg),KMC_Misc.Time_Convert(time2_avg),KMC_Misc.Time_Convert(time3_avg))
        
        # Get the ration of analysis time to simulation time
        self.time_ratio = 100*self.tot_analysis_time/self.tot_kmc_time
        
        # Print simulation end info
        if self.enable_printouts == True:
            Print_End_Info(self.save_dir,self.time_data,self.Pulse.Engine.event_counters[0],self.time_ratio)
            
        # Save final lattice
        self.final_lattice = np.copy(self.Pulse.Engine.lattice)
        if self.Parameters['Save_Final_Lattice'] == True:           
            Save_Final_Lattice(self.final_lattice,self.save_path_final_lattice)
                               
        # Display any Final Plots
        if self.Parameters['Enable_Plots']  == True:
            # Statistics Plot
            self.Pulse.Engine.Sim_Stats.Plot()
            
            # Lattice Plot
            if self.Parameters['Dimension']  == '2D':
                KMC_Misc.Plot_Lattice_2D(self.final_lattice,100)
            elif self.Parameters['Dimension']  == '3D':
                KMC_Misc.Plot_Lattice_3D_Top_Down_2(self.final_lattice,200,200)
            
            # Coverage Plot
            if self.on_the_fly == True:
                self.Pulse.Analysis.Plot()
                
                
        # Debug Print outs        
#        print ('DEBUG')
#        print ('')

                


    def results(self,*args):
        """ Return specific simulation results to main script for further analysis. 
        
        args
        ------
        Info:   Return useful information about the simulation
        Sim_Time:  Total computer time of the simulation in seconds
        Total_Events:  Count of total stelps done in the simulation
        Final_Lattice:   Particle Positions at end of simulation
        Sim_Path:  Full path to the Parameters File
        Project_Path: Path to the project folder
        Analysis_Times: Arrays of analysis time stes and actual time steps
        Coverage:  Layer coverage and reflectivity results
        Islands:  Island size distribution results
        Max_Ni:  Array of maximum size of process lists
        
        """
        
        Results = {}  # dictionary to store results data
        
        for argv in args:
            if argv == 'Info':
                Results['Info'] = [self.rate_catalog,self.rates,self.keys]
            elif argv == 'Sim_Time':
                Results['Sim_Time'] = self.sim_time
            elif argv == 'Total_Events':
                Results['Total_Events'] = self.Pulse.Engine.event_counters[0]
            elif argv == 'Final_Lattice':
                Results['Final_Lattice'] = self.final_lattice
            elif argv == 'Sim_Path':
                Results['Sim_Path'] = self.Parameters['Save_Path_Simulation']
            elif argv == 'Project_Path':
                Results['Project_Path'] = self.Parameters['Save_Path_Project']                
            elif argv == 'Analysis_Times':
                Results['Analysis_Times'] = self.Pulse.Analysis.analysis_times
                Results['Actual_Times'] = self.Pulse.Analysis.actual_times
            elif argv == 'Coverage':
                Results['Coverage'] = self.Pulse.Analysis.cov
                Results['Reflectivity'] = self.Pulse.Analysis.refl
            elif argv == 'Islands':
                Results['Island_Size'] = self.Pulse.Analysis.island_data
                Results['Island_Index'] = self.Pulse.Analysis.island_index            
            elif argv == 'Max_Ni':
                Results['Max_Ni'] = self.Pulse.Engine.max_ni
            else:
                return 'Unkown Key: Please see documentation for list of keys.'
            
        return Results


    
#--------------------------------------------------------------------------------------------------------------------------------------------------        
#####################
###   Functions   ###
#####################




def Make_Directory(proj_name,sim_name):
    """  Make top level save directories. Check if directory exists and make one if needed. """
    
    # Set the root directory
    root_dir = Path.cwd()                 # root directory, where input script resides
    results_dir = root_dir / 'Results'    # path to results folder
    proj_dir = results_dir / proj_name     # path to project folder
    sim_dir = proj_dir / sim_name          # path to simulation folder
        
    # Check for Results directory, make one if needed
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Create Project Directory
    Path(proj_dir).mkdir(parents=True, exist_ok=True) 
    
    # Create Simulation Directory
    Path(sim_dir).mkdir(parents=True, exist_ok=True)    
    
    return (root_dir,results_dir,proj_dir,sim_dir)




def Make_Sub_Directory(sim_dir,save_name):
    """ Make save folder for individual simulation. """
    save_dir = sim_dir / save_name         # path to sub folder for individulal simulations of same parameters
    
    # Create Simulation Save Directory
    Path(save_dir).mkdir(parents=True, exist_ok=True) 
    
    return save_dir




def Date_Time_Stamp():
    """ Make a Date-Time stamp for the simulation. """
    
    DT = datetime.datetime.now()
    L = []
    for attr in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        L.append(getattr(DT, attr))
    Year = str(L[0]); Month = str(L[1]); Day = str(L[2]); Hour = str(L[3]); Minute = str(L[4]); Second = str(L[5]); 
    if len(Month) == 1: Month = '0' + Month
    if len(Day) == 1: Day = '0' + Day
    if len(Hour) == 1: Hour = '0' + Hour
    if len(Minute) == 1: Minute = '0' + Minute
    if len(Second) == 1: Second = '0' + Second 

    Stamp = Year + '_' + Month + '_' + Day + '_' + Hour + '_' + Minute + '_' + Second + '_'
    
    return (Stamp)




def Set_File_Names(Parameters):
    """ Create File Name Structures.  
    
    Outputs
    -------
    File_Name_Base: String of base name for save file
    num:  number of particles per pulse in scientific notation for save file name and display printing
    
    """
    
    if Parameters['Dimension'] == '2D':
        file_name_base = '_2D_PLD'
    else:
        file_name_base = '_3D_PLD'
        
    file_name_base = Parameters['Simulation_Name'] + file_name_base
    
    return file_name_base




def Print_Start_Info(Parameters,sim_num):
    """ Print Starting Message. """
    
    if Parameters['Dimension'] == '2D':  
        string1 = 'Pulses = '+str(Parameters['Pulses'])+'   Particles per Pulse = '+str(Parameters['n'])
        string2 = 'Pulse Width = '+str(Parameters['Pulse_Width'])+' s'+'    Dwell Time = '+str(Parameters['Dwell_Time'])+' s'
        string3 = 'Substrate Size = '+str(Parameters['Lx'])
        string4 = 'Average Deposition Rate = ' + str(Parameters['Dep_Rate_Avg'])[0:7] + ' monolayers/s' + \
                  '    Instantaneous Deposition Rate = ' + str(Parameters['Dep_Rate_Peak']) + ' monolayers/s' 
        string5 = 'Starting Simulation ' + str(sim_num) + ' out of ' + str(Parameters['Number_of_Simulations'])
        print ('')
        print ('{:-^113}'.format(''))    # full line of ----
        print ('{:-^113}'.format('  2D Kinetic Monte Carlo Simulation  '))
        print ('{:-^113}'.format(''))
        print ('')
        print ('{:^113}'.format(string1))
        print ('{:^113}'.format(string2))
        print ('{:^113}'.format(string3))
        print ('{:^113}'.format(string4))
        print ('')
#        print ('{:-^113}'.format('  Simlulation Started  '))
        print ('{:-^113}'.format(string5))
        print ('')
        
    if Parameters['Dimension'] == '3D':
        string1 = 'Pulses = '+str(Parameters['Pulses'])+'   Particles per Pulse = '+str(Parameters['n'])
        string2 = 'Pulse Width = '+ str(Parameters['Pulse_Width'])+' s'+'   Dwell Time = '+str(Parameters['Dwell_Time'])+' s'
        string3 = 'Substrate Size = '+ str(Parameters['Lx']) + 'x' + str(Parameters['Ly'])
        string4 = 'Average Deposition Rate = ' + str(Parameters['Dep_Rate_Avg'])[0:7] + ' monolayers/s' + \
                  '    Instantaneous Deposition Rate = ' + str(Parameters['Dep_Rate_Peak']) + ' monolayers/s'
        string5 = 'Starting Simulation ' + str(sim_num) + ' out of ' + str(Parameters['Number_of_Simulations'])          
        print ('')
        print ('{:-^113}'.format(''))
        print ('{:-^113}'.format('  3D Kinetic Monte Carlo Simulation  '))
        print ('{:-^113}'.format(''))
        print ('')
        print ('{:^113}'.format(string1))
        print ('{:^113}'.format(string2))
        print ('{:^113}'.format(string3))
        print ('{:^113}'.format(string4))
        print ('')
#        print ('{:-^113}'.format('  Simlulation Started  '))
        print ('{:-^113}'.format(string5))
        print ('')




def Print_End_Info(save_dir,time_data,tot_events,time_ratio):
    """ Print Final Simulation Message. """
        
    print ('{:-^113}'.format(''))
    print ('{:-^113}'.format('  Simulation Complete  '))
    print ('{:-^113}'.format(''))  
    print ('')
    print('Total Steps = %s' % tot_events)
    print ('')
    print('Execution Time = %s Hours, %s Minutes, %s Seconds' % time_data[0])
    print ('')
    print ('Average Pulse Time = %s Hours, %s Minutes, %s Seconds' % time_data[1])
    print ('')
    print ('Average Analysis Time = %s Hours, %s Minutes, %s Seconds' % time_data[2])
    print ('')
    print ('Percent of Time Spent in Analysis =',str(time_ratio)[0:5],'%')
    print ('')
    print ('Results saved to: ', save_dir)
    print ('{:-^120}'.format(''))
    print ('')
    print ('')
    print ('')




def Print_Start_of_Pulse(i,sim_number,tot):
    """ Print Start of Pulse Message. """
    
    print ('Pulse',str(i+1),'  ( Simulation',str(sim_number),'out of',str(tot),')')

    
    
    
def Print_End_of_Pulse(i,time_data,tot_steps,steps,a_time):
    """ Print End of Pulse Message. """
    
    print ('Pulse',str(i+1), 'Complete!')
    print ('       Pulse Execution Time =',str(time_data),'s')
    print ('       Pulse Analysis Time =',str(a_time),'s.   Percent of Total = ',str(100*a_time/time_data)[0:5],'%')
    print ('       Total Steps:',str(tot_steps),'   Steps in pulse:',str(steps))
    print ('')
    
    
    
    
def Print_Start_of_Anneal():
    """ Print Start of Anneal Step Message """
    
    print ('Post Anneal')
    
    
    
    
def Print_End_of_Anneal(time_data,tot_steps,steps,a_time):
    """ Print End of Anneal Message. """
    
    print ('Post Anneal Complete')
    print ('       Anneal Execution Time =',str(time_data),'s')
    print ('       Anneal Analysis Time =',str(a_time),'s.   Percent of Total = ',str(100*a_time/time_data)[0:5],'%')
    print ('       Total Steps:',str(tot_steps),'   Steps in Anneal:',str(steps))
    print ('')




def Save_Final_Lattice(lattice,save_name):
    """ Save the final state of the lattice. """
    
    # Cut off layers with no particles (this occasionally fails, so remove for now)
#    Layer_Cov = np.count_nonzero(lattice,axis=(0,1))
    
#    print (Layer_Cov)
#    Layer_Index = np.where(Layer_Cov == 0)[0][0]
#    Lattice = lattice[:,:,0:Layer_Index]
    
    # Save the Lattice
    np.save(save_name,lattice)
    
    
    
def Save_Summary(Parameters,Save_CSV_Summary):
    """ Save Simulation Parameters as a csv file. """
    
    # csv file
    header = ['Parameter','Value']                
    with open(Save_CSV_Summary, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header)
        for key, value in Parameters.items():
            line = [key, value]
            writer.writerow(line)
    




# End of Module