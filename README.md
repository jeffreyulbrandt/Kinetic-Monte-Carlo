# Kinetic-Monte-Carlo
Kinetic Monte-Carlo code to simulate Pulsed Laser Deposition

## Current version: 3.4.1

## System Rquirements
Tested on Windows and Linux (also should work on Mac).
Must install any standard python3 package containing Cython. Anaconda is recommended.


## How to install (Linux)
- Download all files from github to a local directory
- Open terminal and go to the directory
- Type: python setup.py build_ext --inplace
- Once python extension files are generated, you can move the .pyx, .pxd, and the setup.py files to a seperate folder (example: src) to reduce clutter
- All .py and python extenstion files (.so on Linux/Mac) must be in the same directory as the Input Script file to run the simulation.

- ## How to install (Windows with Anaconda)
- Download all files from github to a local directory
  -  name the directory KMC_v3.4.1
- Open Anaconda Prompt and cd to the directory
  - should say (base) next to directory
  - example: cd\Documents\KMC_v3.4.1
- Type: python setup.py build_ext --inplace
- Once python extension files are generated, you can move the .pyx, .pxd, .cpp, Chrome Documents, and the setup.py files to a seperate folder (example: src) to reduce clutter
- All .py, .ipynb, and python extenstion files (.pyd) must be in the same directory as the Input Script file to run the simulation.


## How to run a simulation
- from Jupyter Notebook (recommended)
  - Launch Jupyter
  - Load the file: KMC_Simulation_Notebook_v3.4.1.ipynb
  - Set the Parameters in the second cell
  - Run each cell starting from the top of the notebook
  - Once the Simulation is complete, the simulation data remains in memory and can be analyzed in other cells
- from Python Terminal
  - Set up Input Script File
    - Open the file: KMC_Input_Script_v3p4p1.py in python editor
    - Set the parameters and save the file. It is recommended to give the a file a new name.
  - Open a termial
  - Launch Python
  - Type: python filename.py where filename is the name of the input script

## How to Analyze Data
- Raw simulation data is saved to an .hdf5 file and can be loaded with the .h5 module
- Analyzed Data is also saved to an .hdf5 file and can be load the same way
