########################################################################################################################
########################################################################################################################
#########################                                                                      #########################
#########################            PLD Kinetic Monte Carlo Post-Analysis Module              #########################
#########################                                                                      #########################
########################################################################################################################
########################################################################################################################


"""
Kinetic Monte Carlo Post-Analysis Module

Contains code for analysis of simulation data

Version: 3.4.0

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



###################
###   Classes   ###
###################

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


###############################
###   Post Analysis Class   ###
###############################

class Post_Analysis:
    """
    Class for analyzing data
    
    """
    
    def __init__(self):
        self.Initialize()
    
    
    
    def Initialize(self):
        pass





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


###############################################
###   Simulation Data Post Analysis Class   ###
###############################################


class Simulation_Data:
    """
    Class for loading simulation data: All Events, Lattice, or Surface
    
    """
    
    def __init__(self):
        self.Initialize()
    
    
    
    def Initialize(self):
        pass




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


##########################################
###   On-the-Fly Post Analysis Class   ###
##########################################


class Post_On_the_Fly:
    """
    Class for loading on-the-fly analysis data
    
    """
    
    def __init__(self):
        self.Initialize()
    
    
    
    def Initialize(self):
        pass
    
    
    
    
    


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    
    
#####################
###   Functions   ###
#####################


# Constant Function
        
def Constant(constant):
    return constant


# Polynomial Functions

def Linear(x,m,b):
    return m*x + b

def Quad(x,a,b,c):
    return a*x*x + b*x + c

def Poly(x,n,*a):
    """ Generic Polynomial of degree n. a is a list of coefficients """
    for i in range(n):
        poly = 1
    
    return poly


# Power Law Function
    
def Power(x,A,k):
    return A*x**k


# Exponential Functions
    
def Exp(x,A,x0,tau):
    return A*np.exp(-(x-x0)/tau)


# Distributions
                    
def Gaussian(x,amp,mu,sigma):
    return amp*(np.exp(-((x-mu)**2)/(sigma**2)))

def Lorentz(x,amp,mu,sigma):
    return amp*()




# X-ray Fitting Functions


































# End of Module