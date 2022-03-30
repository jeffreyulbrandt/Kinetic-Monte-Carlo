#############################################################################################################################
#############################################################################################################################
#########################                                                                           #########################
#########################            PLD Kinetic Monte Carlo Simulation - Input Script              #########################
#########################                                                                           #########################
#############################################################################################################################
#############################################################################################################################

"""

Input Script to run the PLD Kinetic Monte Carlo code

Version: 3.4.1a

Instructions:
-------------
Set all listed parameters. Then choose a pre-defined simulation or write your own. Run Input Script from command line from 
the directory where the script is saved.

Notes on Return method for Simulation Class
-------------------------------------------
After running a simulation, the results can be returned to the input script using .return([list]), where list is a list of
the return data types.

"""


# Import Modules
import numpy as np
import time

import KMC_Main_v3p4p1 as KMC_Main
import KMC_Miscellaneous_v3p4p1 as KMC_Misc




###################################
###   User Defined Parameters   ###
###################################
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Parameters = {}     # Create Parameters Dictionary

Parameters['Current_Version'] = 'v3.4.1'   # version of KMC running


# Simulation Parameters
Parameters['Project_Name'] = 'Development'      # Name for the project save directory.
Parameters['Simulation_Name'] = 'Test'          # Name of particular simulation within a project. Can be an empty string, '', because it's DateTime stamped

Parameters['Enable_Print_Outs'] = True          # Enable/Disable simulation prinouts.
Parameters['Enable_Plots'] = True               # Display final plots at end of simulation. Set to False if running multiple simulations

Parameters['Solver_Type'] = 'Binary'            # Type of Solver: 'Linear' O(N^2), 'Binary' O(N log N), 'Set' O(N log N) (currently not wokring)

Parameters['Simulation_Type'] = 'Diffusion'    # Simulation Type. Choices: 'Deposition', 'Diffusion', 'Density', 'Ising' (future)
Parameters['Dimension'] = '3D'                  # Either 2D (1+1) or 3D (2+1). For 2D only Lx is used.

Parameters['Number_of_Simulations'] = 1         # Number of back-to-back simulations to perform (fixed parameters). Analysis results will be averaged.



# Save Data Parameters - (Note: Data is now saved at each analysis time step)
Parameters['Save_All_Events'] = False          # Saves all events in the simulation to output file.
Parameters['Save_Lattice'] = False             # Save snapshot of lattice at specified times.
Parameters['Save_Surface'] = False             # Save a top down view of surface at specified times.
Parameters['Save_Time_Step'] = 0.6             # Time interval to save lattice and statistics data (set to resolution needed for post analysis). Also for Progress Bar update.
Parameters['Save_with_Analysis'] = False       # Save data at same time as analysis. This overrides the above time step (though it is still used for progress bar). 



# On-the-Fly Analysis Parameters - Note: currently only enabled for 3D Deposition simulations
Parameters['Coverage'] = True                  # do a coverage analysis
Parameters['Island_Analysis'] = True           # do island size distribution analysis
Parameters['Structure_Factor'] = True          # do structure factor analysis
Parameters['Gaussian_Filter'] = 1.0            # apply gaussian filter to SF analysis. 0 = no filter, > 0 = sigma value

Parameters['Analysis_Style'] = 'time'              # choose whether to analyze simulation by time step ('time') or simulation steps ('step')
Parameters['Analysis_Time_Spacing'] = 'linear'     # delta t spacing of analysis points. Options: 'linear', 'quad', 'log'. For 'log' this is time of first data point
Parameters['Log_Points'] = 100                     # For log spacing option only. This is the number of points per pulse. Set 'Analysis_Delta_Time' to be the first analysis point

Parameters['Analysis_Delta_Time'] = 0.1      # time interval for On-the-Fly analysis in seconds. Set 'Analysis_Style' to 'time'
Parameters['Analysis_Delta_Step'] = 1e4      # step interval for On-the-Fly analysis. This will result in non-linear time step for analysis points. Set 'Analysis_Style' to 'step'

Parameters['On_the_Fly_Save'] = False         # whether to save the on-the-fly results for each simulation. Usually true, unless doing multiple simulations.
Parameters['Average_Results'] = True          # average on-the-fly results and save across mulit-simulations. Only used when running multiple simulations.



# Substrate Parameters
Parameters['Lx'] = 200                   # Substrate size in x direction
Parameters['Ly'] = 200                   # Substrate size in y direction (not used in 2D mode)
Parameters['depth'] = 10                 # Substrate size in z direction. Give yourself enough layers. (be careful setting this, or errors could occur)

Parameters['Substrate_Type'] = 'Islands'           # Substrate Type. Options: 'Flat', 'Islands', 'Steps', 'Custom'

Parameters['Feature_Grid'] = 'Square'             # Feature Grid: 'Square', 'Hex': Hexagonal. This is for 'Islands' type only.
Parameters['Feature_Layout'] = 'Uniform'          # Feature Layout: 'Uniform': uniform grid, 'Correlated': random offset from uniform , 'Random': random layout
Parameters['Feature_Spacing'] = (1,1,0)         # (number in x direction, number in y, avg spread of features (sigma)). For 'Islands' only.

Parameters['Size_Distribution'] = 'Gaussian'        # Size ditribution of features: 'None', 'Gaussian', or 'Correlated'
Parameters['Size_Values'] = (88,1,0)                # (Avg Radius or Step Length, Height of features, Width of distribution)

Parameters['Substrate_Particle_State'] = 'Passive'      # Whether substrate particles are active or not. Options: 'Active', 'Passive'.

Parameters['Substrate_File'] = 'Test.npy'          # For custom substrate, name of coordinates file. File must be in same directory as simulation script.



# Deposition Parameters (Deposition type simulations only)
Parameters['Deposition_Type'] = 'PLD'              # Type of deposition simulation. Options: 'PLD' or 'MBE'
Parameters['Pulses_per_Layer'] = 20                # number of pulses per monolayer. For PLD only
Parameters['Layers'] = 0.05                        # number of monolayers to simulate

Parameters['Dwell_Time'] = 10.0                    # Total pulse period (for PLD) or total deposition time (for MBE) in seconds.
Parameters['Pulse_Shape'] = 'Uniform'              # Shape of the deposition pulse. Options: 'Uniform', 'Non_Uniform'
Parameters['Pulse_Width'] = 1e-5                   # Width of deposition pulse (for PLD) or total deposition time (for MBE, set the same as dwell time) in seconds.

Parameters['Post_Anneal'] = False                  # Run a Post Anneal Step (True/False)
Parameters['Post_Anneal_Time'] = 100.0             # Length of Post Anneal step in seconds.



# Density Map Model Parameters (Density map type simulations only)
Parameters['Total_Sims'] = 10                        # total number of sims



# Diffusion Only Parameters (Diffusion type simulations only)
Parameters['Diffusion_Type'] = 'Island'          # Type of diffusion sim to run. Options: 'Standard' - particles on surface, 'Island' - diffusing off island

Parameters['Cluster'] = np.array([[0,1,2],[0,0,0]])                                                 # set up cluster coordinates for surface diffusion
Parameters['Cluster'] = np.array([[100],[100]])         # one particle in the middle

Parameters['Sim_Time'] = 1.0            # simulation time in seconds (Standard Type only)

Parameters['Particle_Number'] = 30              # how many particles to simulate
Parameters['Tracer_Number'] = 5                # how many tracer diffusion paths to save in simulation



# Thermal Processes and Activation Energies (Slow Dynamics)
Parameters['Enable_Processes'] = 'No_Uphill_4NN'      # Choose which processes to enable. Options: 'All', 'No_Uphill_4NN', 'No_Uphill', 'No_Detach', 'No_Edge'

Parameters['w0'] = 10**13                       # Prefactor for Arrhenius law (Crystal Vibration Frequency: w0 ~ kT/h )
Parameters['Substrate_Temperature'] = 600       # Substrate temperature in Celcius

Parameters['Ea_diffusion'] = 1.8                # Energy Barrier for surface diffusion in eV
Parameters['Ea_ehrlich_schwoebel'] = 0.0        # Energy Barrier for downhill/uphill diffusion in eV
Parameters['Ea_detach'] = 10.3                   # Energy Barrier for detachment from one nearest neighbor in eV
Parameters['Ea_edge'] = 10.7                     # Energy Barrier for Edge diffusion in eV (3D only). Value of 0.0 means same rate as surface diffusion
Parameters['Ea_corner'] = 0.0                   # Energy Barrier for Corner diffusion in eV (3D only)



# Non-Thermal Processes (Fast Dynamics)
Parameters['Downward_Funneling'] = False      # enables downward funneling when particles land on step edges
Parameters['Transient_Mobility'] = False      # enables transient mobility
Parameters['Island_Chipping'] = False         # enable island chipping






#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##############################
###   Simulation Scripts   ###
##############################


# Choose a simulation script to run, or make a new one

"""
Pre-defined Simulation Scripts
------------------------------

Option_1: Generic single simulation
Option_2: Run several simulations varying a parameter
Option_3: Benchmark simulation (for testing speed and memory usage)
Option_4: Standard Time Benchmark test (test computer speed)

"""

Sim_Script = 'Option_1'


# Option 1 - Run a single simulation
if Sim_Script == 'Option_1':
    
    # run the simulation
    Simulation = KMC_Main.Simulation(Parameters)
    Simulation.Run()


    


        
# Option 2 - Run several simulations and vary a parameter
elif Sim_Script == 'Option_2':
    
    E_edge = np.arange(0.0,1.0,0.01)  # Energy Barriers for edge diffusion
    N = E_edge.shape[0]  # number of sims to run
    
    
    for i in range(N): 
        # reset parameters
        Parameters['Ea_edge'] = E_edge[i]
        Parameters['Simulation_Name'] = 'MBE_R1e6_Ea_'+str(E_edge[i])[:4]+'eV'
        
        Simulation = KMC_Main.Simulation(Parameters)
        Simulation.Run()
        
        time.sleep(2)  # to prevent same Date-Time Stamp

        print ('Sim',str(i),'out of',str(N),'Complete!')
    
    print ('Done!!!!!')





# Option 3 - Benchmark Test. Vary L and n for each Solver and record time (add total steps as well)
elif Sim_Script == 'Option_3':
    
    Results = ['Sim_Time','Total_Events']  # return simulation time
    
    L = np.array([50,100,200,300,400,500])  # L for each sim
#    L = np.array([50,100])
    n = L*L/20   # number of particles in each sim
    n = n.astype(np.int32)
    
    # Test scaling of L
    
    # Linear Solver
    sim_times_lin = []
    events_lin = []
    Parameters['Solver_Type'] = 'Linear'
    for i in range(L.shape[0]):
        Parameters['Lx'] = L[i]
        Parameters['Ly'] = L[i]
        Parameters['n'] = n[i]
        
        Simulation = KMC_Main.Simulation(Parameters)
        Simulation.Run()
        Time = Simulation.results(*Results)
        sim_times_lin.append(Time['Sim_Time'])
        events_lin.append(Time['Total_Events'])
        
        time.sleep(2)  # to prevent same Date-Time Stamp
    
    # Binary Solver
    sim_times_bin = []
    events_bin = []
    Parameters['Solver_Type'] = 'Binary'
    for i in range(L.shape[0]):
        Parameters['Lx'] = L[i]
        Parameters['Ly'] = L[i]
        Parameters['n'] = n[i]
        
        Simulation = KMC_Main.Simulation(Parameters)
        Simulation.Run()
        Time = Simulation.results(*Results)
        sim_times_bin.append(Time['Sim_Time'])
        events_bin.append(Time['Total_Events'])
        
        time.sleep(2)  # to prevent same Date-Time Stamp
    
    # Set Solver
    sim_times_set = []
    events_set = []
    Parameters['Solver_Type'] = 'Set'
    for i in range(L.shape[0]):
        Parameters['Lx'] = L[i]
        Parameters['Ly'] = L[i]
        Parameters['n'] = n[i]
        
        Simulation = KMC_Main.Simulation(Parameters)
        Simulation.Run()
        Time = Simulation.results(*Results)
        sim_times_set.append(Time['Sim_Time'])
        events_set.append(Time['Total_Events'])
        
        time.sleep(2)  # to prevent same Date-Time Stamp

    # Print Results
    print ('L',L)
    print ('n',n)
    print ('')
    print ('Linear Times',sim_times_lin)
    print ('Linear Events',events_lin)
    print ('')
    print ('Binary',sim_times_bin)
    print ('Binary Events',events_bin)
    print ('')
    print ('Set',sim_times_set)
    print ('Set Events',events_set)
    


    
    
    
# end of script