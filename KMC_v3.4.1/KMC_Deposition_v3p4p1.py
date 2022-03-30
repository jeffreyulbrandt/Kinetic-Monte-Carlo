#####################################################################################################################
#####################################################################################################################
#########################                                                                   #########################
#########################            PLD Kinetic Monte Carlo Deposition Module              #########################
#########################                                                                   #########################
#####################################################################################################################
#####################################################################################################################


"""
Kinetic Monte Carlo Miscellaneous Module

Contains a bunch of useful functions and support classes

Version: 3.4.1a

"""

# import modules
import numpy as np
import matplotlib.pyplot as plt




############################
###   Deposition Class   ###
############################
#--------------------------------------------------------------------------------------------------------------------------------------


class Deposition:
    """
    Deposition Class: Generate a Deposition Pulse, including times, coordinates, incident angles, and energies
    
    
    Inputs
    ------
    Parameters:  parameters dictionary from input script
    
    
    Methods
    -------
    Deposition_Rates:          return average and instantaneous deposition rates
    Create_Deposition_Pulse:   return array and deposition times and coordinates
    Uniform_Deposition:        creates array with uniform deposition rate
    Maxwellian:                creates array with Maxwell-Boltzmann distribution of dep times (future)
    Energy_Distribution:       create an energy distribution (future)
    Angle_Distribution:        deposition angle distribution (future)
    
    
    Output
    ------
    Dep_Times: 1D array of deposition times, 1 larger than n
    Dep_Coordinates:   2D Array of dep coordinates
    
    
    """
    
    
    def __init__(self,Parameters):
        self.Parameters = Parameters
        
        self.dim = self.Parameters['Dimension']   # dimension of simulation
        self.shape = self.Parameters['Pulse_Shape']    # shape of deposition distribution        
        self.dep_width = self.Parameters['Pulse_Width']    # width of the dep pulse in seconds
        self.dwell = Parameters['Dwell_Time']    # dwell time in seconds
        self.n = self.Parameters['n']    # number of particles in pulse
        
        self.Lx = Parameters['Lx']
        self.Ly = Parameters['Ly']
        
        self.dt = self.dep_width/self.n   # average time between dep events for uniform deposition
        
        
        
    def Deposition_Rates(self):
        """ Return deposition rates """
        
        if self.dim == '2D':
            Dep_Rate_Avg = self.n/(self.dwell*self.Lx)
            Dep_Rate_Peak = self.n/(self.dep_width*self.Lx)
            
        if self.dim == '3D':
            Dep_Rate_Avg = self.n/(self.dwell*self.Lx*self.Ly)
            Dep_Rate_Peak = self.n/(self.dep_width*self.Lx*self.Ly)
                    
        return (Dep_Rate_Avg, Dep_Rate_Peak)


        
    def Create_Deposition_Pulse(self,time):
        """ Create Data for a Deposition Pulse """
        
        # times
        self.time = time
        if self.shape == 'Uniform':
            self.Uniform_Deposition()
        elif self.shape == 'Maxwellian':
            self.Maxwell()
        
        # coordinates
        if self.dim == '2D':
            self.Dep_Coordinates = np.random.randint(0,self.Lx,self.n)
            self.Dep_Coordinates = self.Dep_Coordinates.astype(np.int32)
        elif self.dim == '3D':
            self.Dep_Coordinates = np.vstack((np.random.randint(0,self.Lx,self.n),np.random.randint(0,self.Ly,self.n)))
            self.Dep_Coordinates = self.Dep_Coordinates.astype(np.int32)
            
        # Future: Incidence angles along x-direction
        
        # Future: Incidence Energies
        
        return (self.Dep_Times,self.Dep_Coordinates)
        

    
    def Uniform_Deposition(self,sigma=0.3):
        """ Generates a uniform pulse of deposition times with random noise """
        
        # Note: make one more particle than needed. It won't actually be deposited, it just prevents an error in Solver
        Noise = np.random.normal(0,sigma*self.dt,self.n)  # random gaussian noise for dep times
        Noise = np.insert(Noise,0,0)
        self.Dep_Times = np.linspace(self.time,self.time+self.dep_width+self.dt,self.n+1) - Noise  # deposition times
        self.Dep_Times = self.Dep_Times.astype(np.double)
        
        # Make last time longer than full simulation (is there a better way to set this last time?)
        self.Dep_Times[-1] = 1e10



    def Maxwellian(self,time,sigma):
        """ Generate a Maxwell-Boltzmann distrubted deposition pulse """
        pass

    
    
    def Energy_Distribution(self):
        """ Generate energy distribution based on Pulse_Shape """
        pass
    

    
    def Angle_Distribution(self):
        """ Generate angle distribution. Returns Azimuthal and Polar Angles """
        pass
    
    
    
    def Plot(self):
        """ Plot distributions """
        
        plt.figure()
        










# End of Module