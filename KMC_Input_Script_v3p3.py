#############################################################################################################################
#############################################################################################################################
#########################                                                                           #########################
#########################            PLD Kinetic Monte Carlo Simulation - Input Script              #########################
#########################                                                                           #########################
#############################################################################################################################
#############################################################################################################################

"""

Input Script to run the PLD Kinetic Monte Carlo code

Version: 3.3a

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
import matplotlib.pyplot as plt

import KMC_Simulation_v3p3 as KMC_Simulation
import KMC_Miscellaneous_v3p3 as KMC_Misc




###################################
###   User Defined Parameters   ###
###################################
#----------------------------------------------------------------------------------------------------------------------------------------------


Parameters = {}  # Create Parameters Dictionary


# Simulation Parameters
Parameters['Project_Name'] = 'Development'      # Name for the project save folder.
Parameters['Simulation_Name'] = 'Post_Anneal'         # Name of particular simulation within a project. Can be an empty string: '' because it's DateTime stamped

Parameters['Number_of_Simulations'] = 1             # number of simulations to perform (fixed parameters)

Parameters['Enable_Print_Outs'] = True             # Enable/Disable simulation prinouts.
Parameters['Enable_Plots'] = True                  # Display final plots at end of simulation. Turn off if running multiple simulations

Parameters['Simulation_Type'] = 'Deposition'       # Simulation Type. Choices: 'Deposition', 'Diffusion (future)'
Parameters['Dimension'] = '3D'                     # Either 2D (1+1) or 3D (2+1). For 2D only Lx is used.

Parameters['Post_Anneal'] = True                # Run a Post Anneal Step
Parameters['Post_Anneal_Time'] = 60.0          # Length of Post Anneal Step in seconds.

Parameters['Solver_Type'] = 'Binary'            # Type of Solver: 'Linear' O(N^2), 'Binary' O(N log N), 'Set' O(log N)


# Substrate Parameters
Parameters['Lx'] = 500           # substrate size in x direction
Parameters['Ly'] = 500           # substrate size in y direction (not used in 2D mode)
Parameters['depth'] = 20          # how many incomplete layers in the lattice (be careful setting this, or errors could occur)

Parameters['Substrate_Type'] = 'Flat'        # Substrate Style . Options: 'Flat', 'Islands', 'Steps'

Parameters['Feature_Layout'] = 'Uniform'        # Feature Layout: 'Uniform': uniform grid, 'Correlated': offset from uniform , 'Random': completely random layout 
Parameters['Feature_Spacing'] = (10,10,0)       # (number in x direction, number in y, avg spread in lattice units)
Parameters['Size_Distribution'] = 'Gaussian'        # Size ditribution of features: 'None', 'Gaussian', or 'Correlated'
Parameters['Size_Values'] = (10,0)               # (Radius or Step Length, Width of distribution)

Parameters['Substrate_Particle_State'] = 'Passive'      # whether substrate particles are active or not. Options: 'Active', 'Passive'.


# Deposition Parameters (Deposition Type simulation only)
Parameters['Pulses'] = 1                       # number of pulses to simulate
Parameters['Dwell_Time'] = 6.0                 # Pulse Period
Parameters['Pulse_Shape'] = 'Uniform'           # shape of the deposition pulse. Options: 'Uniform', 'Maxwell'
Parameters['Pulse_Width'] = 1e-5                # Width of deposition pulse in seconds (must be less than dwell time)
Parameters['n'] = int(Parameters['Lx']*Parameters['Lx']/20)          # number of particles deposited each pulse


# Future: Diffusion Only Parameters


# Thermal Processes and Activation Energies (Slow Dynamics)
Parameters['Enable_Processes'] = 'No_Uphill_4NN'        # Choose which processes to enable. Options: 'All', 'No_Uphill_4NN', 'No_Detach', 'No_Edge', 'No_Detach_or_Edge'

Parameters['w0'] = 10**6  #10**13             # Prefactor for Arhenius law (Crystal Vibration Frequency: w0 ~ kT/h )
Parameters['Substrate_Temperature'] = 600     # Substrate temperature in Celcius

Parameters['Ea_diffusion'] = 0.7 #1.6114184494753379               # Energy Barrier for surface diffusion in eV
Parameters['Ea_ehrlich_schwoebel'] = 0.0        # Energy Barrier for downhill/uphill diffusion in eV
Parameters['Ea_detach'] = 0.3                  # Energy Barrier for detachment from one nearest neighbor in eV
Parameters['Ea_edge'] = 0.3                    # Energy Barrier for Edge diffusion in eV (3D only). Value of 0 means same as surface diffusion
Parameters['Ea_corner'] = 0.0                   # Energy Barrier for Corner diffusion in eV (3D only)


# Non-Thermal Processes (Fast Dynamics) - currently not enabled
Parameters['Downward_Funneling'] = False      # enables downward funneling when particles land on step edges
Parameters['Transient_Mobility'] = False      # enables transient mobility
Parameters['Island_Chipping'] = False         # enable island chipping


# On-the-Fly Analysis Parameters
Parameters['Coverage'] = True               # do a coverage analysis
Parameters['Island_Analysis'] = True        # do island size distribution analysis
Parameters['Structure_Factor'] = True       # do structure factor analysis
Parameters['Analysis_Time'] = 0.1           # time interval for On-the-Fly analysis
Parameters['On_the_Fly_Save'] = True        # whether to save the on-the-fly results for each simulation. Usually true, unless doing multiple simulations
Parameters['Average_Results'] = False        # average on-the-fly results and save across mulit-simulations. Only used when running multiple simulations


# Save Data Parameters
Parameters['Save_Output_File'] = False         # saves all events in the simulation to output file
Parameters['Save_Lattice'] = False            # save the lattice at specified times. Lattice sites are saved as occupied or unoccupied
Parameters['Save_Surface'] = False            # save a top down view of surface at specified times.
Parameters['Save_Final_Lattice'] = False      # Save a copy of the lattice at end of simulation
Parameters['Save_Times'] = 0.1               # time interval to save lattice and statistics data (set to resolution needed for post analysis). Progress Bar





#----------------------------------------------------------------------------------------------------------------------------------------------
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
    Simulation = KMC_Simulation.Simulation(Parameters)
    Simulation.Run()


    


        
# Option 2 - Run several simulations and vary a parameter
elif Sim_Script == 'Option_2':
    
    E_edge = np.arange(0.0,1.0,0.01)  # Energy Barriers for edge diffusion
    N = E_edge.shape[0]  # number of sims to run
    
    
    for i in range(N): 
        # reset parameters
        Parameters['Ea_edge'] = E_edge[i]
        Parameters['Simulation_Name'] = 'MBE_R1e6_Ea_'+str(E_edge[i])[:4]+'eV'
        
        Simulation = KMC_Simulation.Simulation(Parameters)
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
        
        Simulation = KMC_Simulation.Simulation(Parameters)
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
        
        Simulation = KMC_Simulation.Simulation(Parameters)
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
        
        Simulation = KMC_Simulation.Simulation(Parameters)
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