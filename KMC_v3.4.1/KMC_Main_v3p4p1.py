###############################################################################################################
###############################################################################################################
#########################                                                             #########################
#########################            PLD Kinetic Monte Carlo Main Module              #########################
#########################                                                             #########################
###############################################################################################################
###############################################################################################################


"""
Version: 3.4.1



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
v3.4.0 - JGU - 9/30/2021 - Create Cython Engine class. (1.2x faster)
v3.4.1 - JGU - 11/18/2021 - Add Run until Step. Change Save data to align with analysis. Add Density Map model.


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
    - KMC_Main:               Simulation Class (python only code)
    - KMC_Model_Pulse:        Pulse Class (python code)    
    - KMC_Engine:             Main KMC algorithm (python and cython code)
    - KMC_Lattice_2D:         Lattice algorithms for 2D lattice (cython code)
    - KMC_Lattice_3D:         Lattice algorithms for 2D lattice (cython code)
    - KMC_Solver:             KMC temporal algorithms (cython code)
    - KMC_Deposition:         Code for deposition pulse. (python only code)
    - KMC_Processes:          Code for defining the elementray processes and rate catalog. (python only code)
    - KMC_Miscellaneous:      Misc. Functions, Timer Class, Statistics Class (python only code)
    - KMC_Miscellaneous_Cy:   Misc. Cython Functions, Random number generator (cython code)
    - KMC_Analysis:           On-the-Fly Analysis, Coverage Analysis, Island Analysis, Structure Factor (python and cython code)
    - KMC_Post_Analysis:      Post Analysis Code. Loading sim data and on-the-fly results. (python code)


Classes:
--------
    - Simulation:             Main Top-level Object, controls total simulation. Start from Input Script.
    - Pulse:                  Mid-Level control over individual pulses. Also handles saving which is done after each pulse.
    - KMC_Engine_2D(3D):      Low-level python class. Contain data arrays and creates cython classes.
    - Solver:                 Low-level Cython Extension. Maintains process lists and selects processes and calculates time steps.
    - Processes:              Low-level Cython Extension for lattice calculations. Local Neighborhood Serach algorithms.
    - Process_Catalog:        Calculate elementary rates and moves.
    - Deposition:             Calculate deposition times and coordinates. Incident Energy and Angles.
    - Substrate_Generator:    Create active and passive substrate feature such as islands and steps.
    - Save:                   Creates .hdf5 files for storing raw data and analyzed data.
    - Timer:                  Calculates time to execute sections of the simulation.
    - Statistics:             Stores simuation statistics at fixed time intervals.
    - Progress_Bar:           Real time feedback of simulation progress.
    - On_the_Fly_Analysis:    Analysis of Data. Coverage, Island Distributions, Structure Factor.
    - Post_Analysis:          Further analysis of simulation and on-the-fly data.

    
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

# import KMC modules
import KMC_Model_Pulse_v3p4p1 as KMC_Pulse
import KMC_Model_Diffuse_v3p4p1 as KMC_Diffuse
import KMC_Model_Density_Map_v3p4p1 as KMC_Density
import KMC_Miscellaneous_v3p4p1 as KMC_Misc




############################
###   Simulation Class   ###
############################
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Simulation:
    """
    Simulation Class: Top-level object controling full simulation. Mostly I/O related functions.
    
    Inputs
    ------
    Parameters:   Parameters Dictionary from Input script
    
    Attributes
    ----------
    
    Objects
    -------
    Sim_Model:       The specific simulation model. Can be Pulse, Diffuse, Ising, or any future custom model.
        
    Methods
    -------
    Initialize:          Global initialize the simulations. Set-up new variables.
    Run:                 Main loop for running multiple simulations. Controls averaging of results.
    Start_Simulation:    Initialize the simulation. Create instance of Pulse.
    Execute_Pulse:       Main sequence for a deposition simulation. Loop for all pulses.
    Execute_Diffusion:   Main sequence for diffusion only simulation.
    End_Simulation:      Finish simulation. Save Data.
    Results:             Return simulation data to input script.
    
    """
    
    def __init__(self,Parameters):
        self.Parameters = Parameters
        
        # Extract useful parameters from dictionary
        self.project_folder = self.Parameters['Project_Name']
        self.sim_name = self.Parameters['Simulation_Name']
        
        self.num_sims = self.Parameters['Number_of_Simulations']
        self.enable_printouts = self.Parameters['Enable_Print_Outs']
        self.enable_plots = self.Parameters['Enable_Plots']
        
        self.sim_type = self.Parameters['Simulation_Type']
        self.dim = self.Parameters['Dimension']
        
        self.save_lattice = self.Parameters['Save_Lattice']
        self.save_surface = self.Parameters['Save_Surface']
        
        self.Lx = Parameters['Lx']
        self.Ly = Parameters['Ly']
        
        self.cov = self.Parameters['Coverage']
        self.island = self.Parameters['Island_Analysis']
        self.structure_factor = self.Parameters['Structure_Factor']
        self.on_the_fly_save = self.Parameters['On_the_Fly_Save']
        self.avg_results = self.Parameters['Average_Results']
        
               
        self.Initialize()
        
        
            
    def Initialize(self):
        """ Initialize the simulation object. Global initialize. """
        
        # Error checking on input parameters (coming soon)
        
        # Check for On-the-Fly analysis (only for deposition, but leave here for now)
        if self.cov == True or self.island == True or self.structure_factor == True:
            self.on_the_fly = True
        else:
            self.on_the_fly = False
            
        if self.dim == '2D':
            self.on_the_fly = False   # no on-the-fly analysis available for 2D simulations yet
        
        # Set up Save Directories, Paths, and File Names
        self.file_name_base = Set_File_Names(self.Parameters)
        self.sim_folder = Date_Time_Stamp() + self.sim_name
        self.root_dir,self.results_dir,self.proj_dir,self.sim_dir = Make_Directory(self.project_folder,self.sim_folder)
        
        # add to Parameters Dictionary
        self.Parameters['Save_File_Name'] = self.file_name_base
        self.Parameters['Save_Path_Project'] = str(self.proj_dir)
        self.Parameters['Save_Path_Simulation'] = str(self.sim_dir)
        
        # auto run the simulations. For now, comment out
#        self.Run()
        
    
    
    def Run(self):
        """ Main simulation loop. """
        
        # Multi-Simulation loop
        for sim_num in range(1,self.num_sims+1):
            
            ############################
            ###   Run a simulation   ###
            ############################

            # Initialize the individual simulations
            self.Initialize_Simulation(sim_num)
            
            # Execute the simulation
            if self.sim_type == 'Deposition':               
                self.Start_Deposition_Simulation(sim_num)
            
            elif self.sim_type == 'Diffusion':
                
                self.Execute_Diffusion(sim_num)
                
            elif self.sim_type == 'Density':                
                self.Execute_Density(sim_num)
                
            elif self.sim_type == 'Ising':
                print ('Ising model simulation does not exist yet. Try "Deposition" or "Diffusion" instead')
                
            # Finalize the simulation
            self.End_Simulation(sim_num)
                       
            ####################################
            ###   Average and Save results   ###
            ####################################
            
            # Note: Avergaing results across several simulations only enabled for deposition simulations
            
            if self.sim_type == 'Deposition':
                # after first simulation, set up averaged results save object (this needs to be improved, probably make a data extract function)
                if sim_num == 1:
                    if self.avg_results == True and self.on_the_fly == True:
                        self.Average_Results_File = KMC_Misc.Save(self.Parameters,self.sim_dir,'Analysis_Avg')  # create save object
                        Save_Data = self.Sim_Model.Analysis.Return_Data()
                        self.Average_Results_File.Add_Initial_Data(Save_Data)
                
                # after second simulation, start averaging results
                if sim_num > 1:
                    if self.avg_results == True and self.on_the_fly == True:
                        Save_Data = self.Sim_Model.Analysis.Return_Data()
                        self.Average_Results_File.Add_New_Data(Save_Data)
                        
        # Save averaged data to file
        if self.sim_type == 'Deposition':
            if self.avg_results == True and self.on_the_fly == True:
                self.Average_Results_File.Save_Data()
                    
        # Save a csv parameters file at simulation level
        self.save_csv_summary = str(self.sim_dir / (self.file_name_base + '_Simulation_Parameters.csv'))   # Path to Summary csv File
        KMC_Misc.Save_Parameters_CSV(self.Parameters,self.save_csv_summary)
        
        # Seep for one second to prevent same datetime stamp in very fast simulations (only matters in deposition simulations)
        if self.sim_type == 'Deposition':
            time.sleep(1.0)




    def Initialize_Simulation(self,sim_number):
        """ Initialize the individual simulations. """
        
        # Create seed for RNG from clock time
        self.seed = int(1000*time.perf_counter())
              
        # set up simulation timer
        self.Sim_Timer = KMC_Misc.Timer()  # Total Simulation Time
        self.Sim_Timer.Timer_Start()   
        
        # Set up the Sub directory
        self.save_folder = self.sim_name + '_Sim' + str(sim_number)
        self.save_dir = Make_Sub_Directory(self.sim_dir,self.save_folder)
        
        self.save_path_final_lattice = str(self.save_dir / (self.file_name_base + '_Final_Lattice.npy'))
        self.save_path_statistics = str(self.save_dir / (self.file_name_base + '_Simulation_Statistics.csv'))
        
        # Add directory and path info to Parameters Dictionary        
        self.Parameters['Save_Path_Save'] = str(self.save_dir)
        self.Parameters['Save_Path_Final_Lattice'] = self.save_path_final_lattice
        self.Parameters['Save_Path_Final_Statistics'] = self.save_path_statistics

        
        
        
    def End_Simulation(self,sim_num):
        """ End the individual simulations. """
        
        # Record the simulation time
        self.sim_time = self.Sim_Timer.Record_Time()
        
        # Save the statistics file (need to generalize)
        self.Sim_Model.Engine.Sim_Stats.Save_Stats(self.save_path_statistics)
        
        # Save actual save-time data
        if self.save_lattice == True or self.save_surface == True:
            self.Sim_Model.Save_Time_Data()
        
        # convert time data to hours-minutes-second format for print-outs
        if self.sim_type == 'Deposition':
            self.time_data = (KMC_Misc.Time_Convert(self.sim_time),KMC_Misc.Time_Convert(self.pulse_times[0]),KMC_Misc.Time_Convert(self.pulse_times[1]))
        else:
            self.time_data = (KMC_Misc.Time_Convert(self.sim_time))
        
        # Print simulation end info
        if self.enable_printouts == True:
            if self.sim_type == 'Deposition':
                if self.dep_type == 'PLD':
                    Print_End_Info_PLD(self.save_dir,self.time_data,self.Sim_Model.Engine.event_counters[0],self.time_ratio)
                elif self.dep_type == 'MBE':
                    Print_End_Info_MBE(self.save_dir,self.time_data,self.Sim_Model.Engine.event_counters[0],self.time_ratio)
                
        if self.enable_printouts == False:   # This is just to keep track of how many sims have been completed
            print ('Simulation ',str(sim_num),' out of ', str(self.Parameters['Number_of_Simulations']), ' Complete!!!')
            
        # Display any Final Plots
        if self.enable_plots  == True:
            
            # Statistics Plot
            self.Sim_Model.Engine.Sim_Stats.Plot()
            
            # Lattice Plot (add final lattice back in)
            final_surface = self.Sim_Model.Engine.Surface()
            if self.dim  == '2D':
                KMC_Misc.Plot_Lattice_2D(final_surface,200)
            elif self.dim  == '3D':
                KMC_Misc.Plot_Final_Surface_3D(final_surface,200,200)
            
            # Coverage Plot
            if self.on_the_fly == True:
                self.Sim_Model.Analysis.Plot()
                
        
        
        
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #####################################################
    ###   Code Block for Deposition Type Simulations  ###
    #####################################################        

        
    def Start_Deposition_Simulation(self,sim_number):       
        """ Initialize a Deposition Type Simulation. """
        
        # Extract variables from Parameters Dictionary
        self.dep_type = self.Parameters['Deposition_Type']
        self.dwell = self.Parameters['Dwell_Time']
        self.pulse_width = self.Parameters['Pulse_Width']
        self.ppl = self.Parameters['Pulses_per_Layer']
        self.layers = self.Parameters['Layers']        
        self.post_anneal = self.Parameters['Post_Anneal']
        
        # Create new variables
        if self.dep_type == 'PLD':
            self.pulses = int(self.ppl*self.layers)     # number of pulses in PLD sim
            self.n = int(self.Lx*self.Ly/self.ppl)      # number of particle per pulse
            self.inst_dep_rate = self.n/(self.Lx*self.Ly*self.pulse_width)   # instantaneous dep rate in monolayers/second
            self.avg_dep_rate = self.n/(self.Lx*self.Ly*self.dwell)          # average dep rate in monolayers/second
            
            self.Parameters['Instantaneous_Deposition_Rate'] = self.inst_dep_rate
            
        elif self.dep_type == 'MBE':
            self.pulses = 1
            self.n = int(self.Lx*self.Ly*self.layers)    # number of particles in MBE simulation
            self.avg_dep_rate = self.n/(self.Lx*self.Ly*self.dwell)  # average dep rate in monolayers/second
            
        else:
            print ('Unknown deposition type. Please choose "PLD" or "MBE".')
                             
        # Add new variables to Parameters dictionary
        self.Parameters['Pulses'] = self.pulses
        self.Parameters['n'] = self.n
        self.Parameters['Average_Deposition_Rate'] = self.avg_dep_rate
        
        # Create the Pulse Instance
        self.Sim_Model = KMC_Pulse.Pulse(self.Parameters,self.save_dir,self.seed)
        
        # Set up simulation timers        
        self.Pulse_Timer = KMC_Misc.Timer()   # Simulation Time per Pulse (for PLD type simulations)
        self.Post_Timer = KMC_Misc.Timer()    # Post Anneal Time      
        
        # Print simulation start info
        if self.enable_printouts == True:
            if self.dep_type == 'PLD':
                Print_Start_Info_PLD(self.Parameters,sim_number)
            elif self.dep_type == 'MBE':
                Print_Start_Info_MBE(self.Parameters,sim_number)
            
        # simulation level statistics variables    
        self.tot_analysis_time = 0   # variable to keep track of total analysis time
        self.tot_kmc_time = 0        # variable to keep track of total simulation time
        self.time_ratio = 0          # percent of time spent in analysis
        
        # start the simulation
        self.Execute_Pulses(sim_number)
        


        
    def Execute_Pulses(self,sim_number):
        """ Main Sequence for Deposition Style Simulation. """
        
        # Run all pulses
        for i in range(self.pulses):
            # Start the Pulse Timer
            self.Pulse_Timer.Timer_Start()
            
            # Print Start-of-Pulse Message
            if self.enable_printouts == True:
                if self.dep_type == 'PLD':
                    Print_Start_of_Pulse(i,sim_number,self.num_sims)
                elif self.dep_type == 'MBE':
                    Print_Start_of_Dep(sim_number,self.num_sims)
            
            # Simulate a Pulse
            self.Sim_Model.Run_Pulse(i)
            
            # record Pulse Time
            self.Pulse_Time = self.Pulse_Timer.Record_Time()
            
            # add up current siumation times
            a_time = self.Sim_Model.analysis_time_sum
            kmc_time = self.Pulse_Timer.store_times[-1]
            
            self.tot_analysis_time += a_time
            self.tot_kmc_time += kmc_time
            
            # Print End-of-Pulse update
            if self.enable_printouts == True:
                if self.dep_type == 'PLD':
                    Print_End_of_Pulse(i,self.Pulse_Timer.store_times[-1],self.Sim_Model.end_of_pulse_events_tot,self.Sim_Model.end_of_pulse_events,a_time)
                elif self.dep_type == 'MBE':
                    Print_End_of_Dep(self.Pulse_Timer.store_times[-1],self.Sim_Model.end_of_pulse_events_tot,self.Sim_Model.end_of_pulse_events,a_time)
                
        # Run a post anneal step?
        if self.post_anneal == True:
            self.Execute_Post_Anneal(sim_number)
            
        # Finalize the Simulation
        self.End_Deposition_Simulation()
            
            

            
    def Execute_Post_Anneal(self,sim_number):
        """ Run a post anneal step """  
        
        # Print post anneal message
        Print_Start_of_Anneal(sim_number,self.num_sims)
        
        # Run the anneal step
        self.Post_Timer.Timer_Start()
        self.Sim_Model.Run_Post_Anneal()
        self.Post_Timer.Record_Time()
            
        # Print end of anneal update           
        a_time = self.Sim_Model.analysis_time_sum
        kmc_time = self.Post_Timer.store_times[-1]
        
        self.tot_analysis_time += a_time
        self.tot_kmc_time += kmc_time
        
        if self.enable_printouts == True:                
            Print_End_of_Anneal(self.Post_Timer.store_times[-1],self.Sim_Model.end_of_pulse_events_tot,self.Sim_Model.end_of_pulse_events,a_time)
       
        


    def End_Deposition_Simulation(self):
        """ End a deposition type simulation """
        
        # Compute Average Pulse and Analysis Times
        pulse_time_avg, pulse_time_std = self.Pulse_Timer.Average()
        if self.on_the_fly == True:
            analysis_time_avg, analysis_time_std = self.Sim_Model.Analysis_Timer.Average()
        elif self.on_the_fly == False:
            analysis_time_avg = 0
            
        self.pulse_times = (pulse_time_avg,analysis_time_avg)  # for final print-outs
    
        # Get the ratio of analysis time to simulation time        
        self.time_ratio = 100*self.tot_analysis_time/self.tot_kmc_time
        
        # Save On-the-Fly Analysis results
        if self.on_the_fly_save == True and self.on_the_fly == True:
            self.Sim_Model.Save_Analysis_Data()

    
    


           
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ####################################################
    ###   Code Block for Diffusion Type Simulations  ###
    ####################################################
                
                
    def Execute_Diffusion(self,sim_number):
        """ Main Sequence for Diffusion Style Simulation. """
        
        # reset some of the parameters
        self.Parameters['Enable_Plots'] = False
        self.enable_plots = False
        self.Parameters['Ea_detach'] = 10.0   # make sure detachment is not activated
        self.Parameters['Ea_edge'] = 10.0     # make sure edge diff is not activated
        
        # Add new variables to Parameters dictionary
        self.Parameters['Pulses'] = self.Parameters['Particle_Number']
             
        # Run the Simulation
        self.Sim_Model = KMC_Diffuse.Diffuse(self.Parameters,self.save_dir,self.seed)
        self.Sim_Model.Run()
        
        # Analyze Results
        if self.Parameters['Diffusion_Type'] == 'Island':
            self.Sim_Model.Island_Results()
            
        elif self.Parameters['Diffusion_Type'] == 'Standard':
            t = self.Sim_Model.save_time
            dr = self.Sim_Model.displacement
            print ('Times:',t)
            print ('Displacement:',dr)
            
            # plot results
            KMC_Misc.Plot(t,dr,'time (s)','displacement (l.u.)')
        
        




#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------  


    ########################################################
    ###   Code Block for Density Model Type Simulations  ###
    ########################################################
                
    
    def Execute_Density(self,sim_number):
        """ Main Sequence for Density Model Simulation. """
        
        # extract relevant parameters
        self.ppl = self.Parameters['Pulses_per_Layer']
        self.layers = self.Parameters['Layers'] 
        
        # set up new parameters
        self.pulses = int(self.ppl*self.layers)     # number of pulses in PLD sim
        self.n = int(self.Lx*self.Ly/self.ppl)      # number of particle per pulse
        
        # Add new variables to Parameters dictionary
        self.Parameters['Pulses'] = self.pulses
        self.Parameters['n'] = self.n
        
        # Make sure analysis is not performed
        self.on_the_fly = False
        
        # Run the simulation
        self.Sim_Model = KMC_Density.Density(self.Parameters,self.save_dir)
        
        if self.enable_printouts == True:
            self.Sim_Model.Run()
        elif self.enable_printouts == False:
            self.Sim_Model.Run_No_Printouts()
        
        # Record the simulation time
        self.sim_time = self.Sim_Timer.Record_Time()
        print ('Total Simulation Time = ',str(self.sim_time/3600),' hours.')
        
    
    

 
        

        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ######################################################
    ###   Code Block for Ising Model Type Simulations  ###
    ###################################################### 
                
    
    def Execute_Ising(self):
        """ Main Sequence for Ising Model Simulation. (Future) """
        pass
        
    

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
                
    # Debug Print outs        
#    print ('DEBUG')
#    print ('')

                


    def Results(self,*args):
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
                Results['Total_Events'] = self.Sim_Model.Engine.event_counters[0]
            elif argv == 'Final_Lattice':
                Results['Final_Lattice'] = self.final_lattice
            elif argv == 'Sim_Path':
                Results['Sim_Path'] = self.Parameters['Save_Path_Simulation']
            elif argv == 'Project_Path':
                Results['Project_Path'] = self.Parameters['Save_Path_Project']                
            elif argv == 'Analysis_Times':
                Results['Analysis_Times'] = self.Sim_Model.Analysis.analysis_times
                Results['Actual_Times'] = self.Sim_Model.Analysis.actual_times
            elif argv == 'Coverage':
                Results['Coverage'] = self.Sim_Model.Analysis.cov
                Results['Reflectivity'] = self.Sim_Model.Analysis.refl
            elif argv == 'Islands':
                Results['Island_Size'] = self.Sim_Model.Analysis.island_data
                Results['Island_Index'] = self.Sim_Model.Analysis.island_index            
            elif argv == 'Max_Ni':
                Results['Max_Ni'] = self.Sim_Model.Engine.max_ni
            else:
                return 'Unkown Key: Please see documentation for list of keys.'
            
        return Results


    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
#####################
###   Functions   ###
#####################




def Make_Directory(proj_name,sim_name):
    """  Make top level save directories. Check if directory exists and make one if needed. """
    
    # Set the root directory
    root_dir = Path.cwd()                  # root directory, where input script resides
    results_dir = root_dir / 'Results'     # path to results folder
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
    File_Name_Base: String of base name for save files
    
    """
    
    if Parameters['Simulation_Type'] == 'Deposition':
        if Parameters['Deposition_Type'] == 'PLD':
            if Parameters['Dimension'] == '2D':
                file_name_base = '_2D_PLD'
            else:
                file_name_base = '_3D_PLD'
        elif Parameters['Deposition_Type'] == 'MBE':
            if Parameters['Dimension'] == '2D':
                file_name_base = '_2D_MBE'
            else:
                file_name_base = '_3D_MBE'
                
    elif Parameters['Simulation_Type'] == 'Diffusion':
        file_name_base = '_Diffusion'
        
    elif Parameters['Simulation_Type'] == 'Density':
        file_name_base = '_Density_Map'
        
    elif Parameters['Simulation_Type'] == 'Ising':
        file_name_base = '_Ising'
        
    file_name_base = Parameters['Simulation_Name'] + file_name_base
    
    return file_name_base




###############################
###   Print Out Functions   ###
###############################
    

def Print_Start_Info_PLD(Parameters,sim_num):
    """ Print Starting Message for Pulsed Deposition simulation """
    
    string1 = 'Pulses = ' + str(Parameters['Pulses']) + '   Pulses per Layer = ' + str(Parameters['Pulses_per_Layer'])
    string2 = 'Pulse Width = ' + str(Parameters['Pulse_Width']) + ' s    Dwell Time = ' + str(Parameters['Dwell_Time']) + ' s'
    string3 = 'Average Deposition Rate = ' + str(Parameters['Average_Deposition_Rate']) + ' monolayers/s'
    string4 =  'Instantaneous Deposition Rate = ' + str(Parameters['Instantaneous_Deposition_Rate']) + ' monolayers/s'
    string6 = 'Particles per Pulse = ' + str(Parameters['n'])
    
    if Parameters['Dimension'] == '2D':
        string5 = 'Substrate Size = '+ str(Parameters['Lx'])
    elif Parameters['Dimension'] == '3D':
        string5 = 'Substrate Size = '+ str(Parameters['Lx']) + 'x' + str(Parameters['Ly'])
    
    
    string7 = '  Starting Simulation ' + str(sim_num) + ' out of ' + str(Parameters['Number_of_Simulations']) + '  '
    
    # print 1st part of message
    print ('')
    print ('{:-^113}'.format(''))    # full line of ----
    print ('{:-^113}'.format('  Kinetic Monte Carlo Simulation  '))
    print ('{:-^113}'.format(''))    # full line of ----
    print ('')
    
    # print middle part of message
    print ('{:^113}'.format('Simulation type: Pulsed Laser Depostion'))
    print ('')

    print ('{:^113}'.format(string1))
    print ('{:^113}'.format(string2))
    if Parameters['Post_Anneal'] == True:
        print ('{:^113}'.format('Post Anneal Time = ' + str(Parameters['Post_Anneal_Time']) + ' s'))
    print ('{:^113}'.format(string3))
    print ('{:^113}'.format(string4))
    print ('{:^113}'.format(string6))
    print ('')
    print ('{:^113}'.format(string5))
    
    # print ending part of message
    print ('')
    print ('{:-^113}'.format(string7))
    print ('')


 
    
def Print_Start_Info_MBE(Parameters,sim_num):
    """ Print Starting Message for Continuous Deposition simulation """
    
    string1 = 'Layers = ' + str(Parameters['Layers'])
    string2 = 'Deposition Time = ' + str(Parameters['Pulse_Width'])
    string3 = 'Average Deposition Rate = ' + str(Parameters['Average_Deposition_Rate']) + ' monolayers/s'
    
    if Parameters['Dimension'] == '2D':
        string5 = 'Substrate Size = '+ str(Parameters['Lx'])
    elif Parameters['Dimension'] == '3D':
        string5 = 'Substrate Size = '+ str(Parameters['Lx']) + 'x' + str(Parameters['Ly'])
    
    
    string7 = '  Starting Simulation ' + str(sim_num) + ' out of ' + str(Parameters['Number_of_Simulations']) + '  '
    
    # print 1st part of message
    print ('')
    print ('{:-^113}'.format(''))    # full line of ----
    print ('{:-^113}'.format('  Kinetic Monte Carlo Simulation  '))
    print ('{:-^113}'.format(''))    # full line of ----
    print ('')
    
    # print middle part of message
    print ('{:^113}'.format('Simulation type: Continuous Depostion'))
    print ('')

    print ('{:^113}'.format(string1))
    print ('{:^113}'.format(string2))
    if Parameters['Post_Anneal'] == True:
        print ('{:^113}'.format('Post Anneal Time = ' + str(Parameters['Post_Anneal_Time']) + ' s'))
    print ('{:^113}'.format(string3))
    print ('')
    print ('{:^113}'.format(string5))
    
    # print ending part of message
    print ('')
    print ('{:-^113}'.format(string7))
    print ('')




def Print_End_Info_PLD(save_dir,time_data,tot_events,time_ratio):
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
    
    
    
    
def Print_End_Info_MBE(save_dir,time_data,tot_events,time_ratio):
    """ Print Final Simulation Message. """
        
    print ('{:-^113}'.format(''))
    print ('{:-^113}'.format('  Simulation Complete  '))
    print ('{:-^113}'.format(''))  
    print ('')
    print('Total Steps = %s' % tot_events)
    print ('')
    print('Execution Time = %s Hours, %s Minutes, %s Seconds' % time_data[0])
    print ('')
    print ('Average Deposition Time = %s Hours, %s Minutes, %s Seconds' % time_data[1])
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
    
    
    
    
def Print_Start_of_Dep(sim_number,tot):
    """ Print Start of Deposition Message """
    
    print ('Deposition','  ( Simulation',str(sim_number),'out of',str(tot),')')




def Print_End_of_Dep(time_data,tot_steps,steps,a_time):
    """ Print End of Pulse Message. """
    
    print ('Deposition Complete!')
    print ('       Deposition Execution Time =',str(time_data),'s')
    print ('       Deposition Analysis Time =',str(a_time),'s.   Percent of Total = ',str(100*a_time/time_data)[0:5],'%')
    print ('       Total Steps:',str(tot_steps),'   Steps in pulse:',str(steps))
    print ('')
    
    
    
    
def Print_Start_of_Anneal(sim_number,tot):
    """ Print Start of Anneal Step Message """
    
    print ('Post Anneal','  ( Simulation',str(sim_number),'out of',str(tot),')')
    
    
    
    
def Print_End_of_Anneal(time_data,tot_steps,steps,a_time):
    """ Print End of Anneal Message. """
    
    print ('Post Anneal Complete')
    print ('       Anneal Execution Time =',str(time_data),'s')
    print ('       Anneal Analysis Time =',str(a_time),'s.   Percent of Total = ',str(100*a_time/time_data)[0:5],'%')
    print ('       Total Steps:',str(tot_steps),'   Steps in Anneal:',str(steps))
    print ('')
    
    


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------      




# End of Module