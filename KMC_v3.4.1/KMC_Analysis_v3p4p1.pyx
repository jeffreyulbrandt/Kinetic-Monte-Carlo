#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

###################################################################################################################
###################################################################################################################
#########################                                                                 #########################
#########################            PLD Kinetic Monte Carlo Analysis Module              #########################
#########################                                                                 #########################
###################################################################################################################
###################################################################################################################

"""
Kinetic Monte Carlo - Analysis Module

Contains code to execute post analysis of simulation data.

Python Class: On_the_Fly_Analysis contains python code

Cython Extension: On_the_Fly_Cy contains code for fast Cluster Analysis and Coverage Analysis

Version: 3.4.1a

"""

# import python modules
from pathlib import Path  # this is used
import numpy as np
import tables
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# cimport modules
cimport numpy as np
from libc.math cimport sqrt
from libcpp.vector cimport vector
cimport cython

# KMC modules
from KMC_Miscellaneous_Cy_v3p4p1 cimport modulus



#----------------------------------------------------------------------------------------------------------------------------------------------------------



############################################
###  Python Class: On the Fly Analysis   ###
############################################
        

class On_the_Fly_Analysis:
    """
    On-the-Fly Analysis Class. Allows simulation analysis while the simulation is running. For now only works with
    3D simulations.
    
    
    Inputs
    ------
    Parameters:   parameters dictionary
    path:         Pathlib object to main save directory
    lattice:      The numpy array of the simulation lattice at the current sim time
    
    
    Attributes
    ----------
    area, inv_area:  The area and inverse area of a layer
    island_cutoff:   this is the largest island size to leave out of total islands count (typically = 1, i.e. no monomers)
    analysis_times:  array of times where simulation is paused to run an analysis step
    points:          length of the analysis times array
    actual_times:    array of actual simulation times when analysis occurs (may differ from analysis_times due to variable time step)
    
    surface:   surface height array
    phase:     Anti-Bragg phase of surface array
    
    cov:         array of layer coverages vs time
    refl:        arrray of specular intensity at anti-bragg
    rms:         root mean square roughness of surface
    avg_height:  average hieght of surface
    
    L:               layer slice of lattice for island size analysis
    labels:          array to label equivalent clusters
    num_clusters:    array of number of clusters vs time
    island_bins:     histogram bins for island size distribution
    isd:             island size distribution
    island_data:     full island size distribution
    island_index:    analysis time index for island data arrays
    
    dr:                circular average step size
    size_x,size_y:     quadrant size of layer to rearranging fourier transform
    rmax:              maximum r for circular average (no more than 1/2 system size L)
    structure_factor:   cirular averaged Strcture Factor
    
    
    Cython Classes
    --------------
    On_the_Fly_Analysis_Cy:  optimized cython analysis algorithms
    
    
    Methods
    -------
    Initialize:         Set-up the analysis object
    Times:              Set times to pause simulation and run analysis
    Do_Analysis:        Start the analysis sequence
    record_time:        Record the actual time of the simulation at an analysis step
    Size_Distribution:  Make a histogram (obsolete)
    Island_Analysis:    Run an insland analysis of a layer
    Structure_Factor:   Structure factor analysis at Anti-Bragg
    Save_Setup:         Set up hdf5 save file
    Save:               Save On-the-Fly analysis data
    Plot:               Make a plot of analyzed results
    
    """



    def __init__(self,Parameters,path,lattice):
        self.Parameters = Parameters
        self.path = path      # path object
        self.lattice = lattice     # starting lattice
        
        # extract relevant parameters
        self.file_name = self.Parameters['Save_File_Name']
        
        self.analysis_style = self.Parameters['Analysis_Style']
        self.analysis_time = self.Parameters['Analysis_Delta_Time']
        self.sampling = self.Parameters['Analysis_Time_Spacing']
        self.log_points = self.Parameters['Log_Points']
        
        self.lx = self.Parameters['Lx']
        self.ly = self.Parameters['Ly']
        self.dim = self.Parameters['Dimension']
        
        self.dwell_time = self.Parameters['Dwell_Time']  
        self.pulses = self.Parameters['Pulses']
        self.width = Parameters['Pulse_Width']
                
        self.post = self.Parameters['Post_Anneal']
        self.post_time = self.Parameters['Post_Anneal_Time']
        
        self.coverage = self.Parameters['Coverage']
        self.island = self.Parameters['Island_Analysis']
        self.structure = self.Parameters['Structure_Factor']
        self.gaussian = self.Parameters['Gaussian_Filter']
        self.save = self.Parameters['On_the_Fly_Save']          
                
        self.Initialize()
        
    

    
    def Initialize(self):
        """ Initialize the Analysis object """
        
        # constants and variables
        self.area = self.lx*self.ly             # Area of Substrate
        self.inv_area = 1/self.area             # Inverse Area of Substrate
        self.norm = 1/(self.area*np.pi*np.pi)   # normalization factor for structure factor
        self.island_cutoff = 1                  # the largest island size not included in island count
        
        k = 0; cov = 1
        while cov == 1:
            cov = np.sum(self.lattice[:,:,k])/self.area
            k += 1            
        self.substrate_layers = k - 1                      # number of complete substrate layers at beginning of simulation (usually 2)
        self.layers = self.lattice.shape[2] - self.substrate_layers     # total number of analyzed layers
        
        
        # set up analysis times
        self.analysis_times = self.Times()             # array of desired analysis times
        self.points = self.analysis_times.shape[0]
        self.actual_times = np.zeros([self.points])    # array to save actual analysis times
        
        
        # create data structures
        self.surface = np.zeros([self.lattice.shape[0],self.lattice.shape[1]],dtype=np.int32)    # surface for rms roughness
        self.phase = np.zeros([self.lattice.shape[0],self.lattice.shape[1]],dtype=np.int32)      # phase surface for fourier transform
        
               
        # Coverage Analysis
        self.cov = np.zeros([self.layers,self.analysis_times.shape[0]],dtype=np.double)    # layer coverage array
        self.refl = np.zeros([self.analysis_times.shape[0]],dtype=np.double)              # Anti-Bragg specular intensity
        self.rms = np.zeros([self.analysis_times.shape[0]],dtype=np.double)               # rms roughness of surface
        self.avg_height = np.zeros([self.analysis_times.shape[0]],dtype=np.double)        # average height of surface
        
        
        # Island Analysis
        self.L = np.zeros([self.lx,self.ly],dtype=np.int32)              # layer slice of lattice for cluster id algorithm
        self.labels = np.arange(0,self.lx*self.ly/2,dtype=np.uint32)     # equivalence classes and labels
        self.labels[0] = 1                                               # set the first cluster label (after analysis, this is the number of islands)
                
        self.num_clusters = np.zeros([self.layers,self.analysis_times.shape[0]],dtype=np.int32)     # total number of cluster excluding monomers
        self.percolation_threshold = np.zeros([self.layers],dtype=np.double)                        # percolation threshold for each layer (not used yet)
        
        self.island_bins = np.arange(1,self.area,dtype=np.int32)                   # array of Bins for island size distribution calculations
        self.isd = np.zeros([self.island_bins.shape[0]+1],dtype=np.int32)          # histogram array for island size distribution
        
        self.island_data = [[] for _ in range(self.layers)]      # data stucture to save live islands size data
        self.island_index = [[] for _ in range(self.layers)]     # data structure to store the corresponding index
        
        
        # Structure Factor Analysis
        
        # Set up k-space
        FT = np.fft.rfft2(self.L)   # dummy fourier transform to set up k space
        dx = 1.0  # lattice spacing
        dy = 1.0  # lattice spacing
        KX = np.fft.rfftfreq(self.L.shape[1])*2*np.pi/dx
        KY = np.fft.fftfreq(self.L.shape[0])*2*np.pi/dy
        self.kx, self.ky = np.meshgrid(KX,KY)                   # k coordinates
        self.kmax = np.max(np.sqrt(self.kx**2+self.ky**2))      # max k off the k array
        
        # set up circular bins
        self.dq = (self.ky[1,0]-self.ky[0,0])*1.0                 # radius of the circular bins
        self.q_bins = np.arange(0,self.kmax+5*self.dq,self.dq)
        self.sub_pix = 10       # number of sub-pixels in each direction (10 seems to be good enough)
        
        self.qmax = np.pi*dx       # max q we want to average to.
        self.max_index = np.where(self.q_bins >= self.qmax)[0][0]
        self.sub_q_bins = self.q_bins[0:self.max_index]        # cut off unwanted pixels
        
        self.structure_factor = np.zeros([self.sub_q_bins.shape[0],self.analysis_times.shape[0]],dtype=np.double)    # array to store structure factor circular average profile
        
        
        # Create Instance of On_the_Fly_Cy
        self.Cy_Analysis = On_the_Fly_Cy(self.substrate_layers,self.lx,self.ly,self.inv_area,self.island_cutoff,self.surface,self.phase,self.L,self.labels,self.cov,self.refl,self.rms,self.avg_height,self.num_clusters,self.percolation_threshold,self.isd,self.sub_pix,self.kx,self.ky,self.q_bins,self.dq,self.structure_factor)
        
        
        # Set up circular bins for structure factor calculation
        self.Cy_Analysis.Create_Bins()
        
        

    
    def Times(self):
        """ Set up Analysis Times """
        
        if self.sampling == 'linear':
            analysis_times = self.Times_Lin()
        elif self.sampling == 'quad':
            analysis_times = self.Times_Quad()
        elif self.sampling == 'log':
            analysis_times = self.Times_Log()
        else:
            print ('Unkown Analysis Time Spacing. Please choose either "linear", "quad", or "log".')
            
        return analysis_times
    


    
    def Times_Lin(self):
        """ Set Analysis Times and return to main simulation. """
        
        # simple linear spaced list
        analysis_times = np.arange(0,self.dwell_time*self.pulses+self.analysis_time,self.analysis_time).astype(np.double)
        
        # decide if extra time point post deposition is needed (only if pulse width is less than sample time)
        if self.width > self.analysis_time:
            pass
        else:
            t = np.arange(self.width,self.dwell_time*self.pulses,self.dwell_time)   # all of the end of deposition times
            for i in range(t.shape[0]):
                ind = np.searchsorted(analysis_times,t[i])
                analysis_times = np.insert(analysis_times,ind,t[i])
                
        # if post anneal, then add extra analysis points on the end
        if self.post == True:
            expand = np.arange(analysis_times[analysis_times.shape[0]-1]+self.analysis_time,analysis_times[analysis_times.shape[0]-1]+self.post_time,self.analysis_time).astype(np.double)
            analysis_times = np.hstack((analysis_times,expand))
                
        return analysis_times
    
    

    
    def Times_Quad(self):
        """ Set Analysis Times using a quadratic scale. """
        
        # Set up one pulse
        analysis_times = []
        t = self.analysis_time
        i = 0
        while t < self.dwell_time:   
            t += self.analysis_time*i
            analysis_times.append(t)
            i += 1
        analysis_times = np.array(analysis_times)
        length = analysis_times.shape[0]
        analysis_times = analysis_times[0:length-1]
        
        # add zero and pulse width
        if self.width < self.analysis_time:
            analysis_times = np.insert(analysis_times,0,self.width)            
        analysis_times = np.insert(analysis_times,0,0)
        
        length = analysis_times.shape[0]     # for post anneal delete
        
        # Copy and Shift for remaining pulses
        tmp = np.copy(analysis_times)
        for i in range(1,self.pulses):
            analysis_times_shift = tmp + i*self.dwell_time
            analysis_times = np.append(analysis_times,analysis_times_shift)
        
        # Add Post anneal
        if self.post == True:
            # delete last pulse
            analysis_times = analysis_times[0:analysis_times.shape[0]-length]
            
            # extend last pulse plus post anneal time
            at2 = []
            t = self.analysis_time + (self.pulses-1)*self.dwell_time
            i = 0
            while t < self.pulses*self.dwell_time + self.post_time:   
                t += self.analysis_time*i
                at2.append(t)
                i += 1
            at2 = np.array(at2)
            at2 = at2[0:-1]
            
            if self.width < self.analysis_time:
                at2 = np.insert(at2,0,self.width+(self.pulses-1)*self.dwell_time)
            at2 = np.insert(at2,0,(self.pulses-1)*self.dwell_time)
            
            analysis_times = np.append(analysis_times,at2)
            
        # add a final data point for end of simulation
        if self.post == True:
            analysis_times = np.append(analysis_times,[self.pulses*self.dwell_time+self.post_time])
        elif self.post == False:
            analysis_times = np.append(analysis_times,[self.pulses*self.dwell_time])
        
        return analysis_times
    
    

    
    def Times_Log(self):
        """ Set Analysis Times using a log scale. Future. """
        
        # Set up one pulse
        start = np.log10(self.analysis_time)
        stop = np.log10(self.dwell_time)
        analysis_times = np.logspace(start,stop,self.log_points)
        
        # add zero and pulse width
        if self.width < self.analysis_time:
            analysis_times = np.insert(analysis_times,0,self.width)            
        analysis_times = np.insert(analysis_times,0,0)
        
        # Copy and Shift for all remaining pulses
        tmp = np.copy(analysis_times[1:])
        for i in range(1,self.pulses):
            analysis_times_shift = tmp + i*self.dwell_time
            analysis_times = np.append(analysis_times,analysis_times_shift)
        
        # Add Post anneal
        if self.post == True:        
            # recalculate number of points
            N = int(self.log_points*self.post_time/self.dwell_time)
    
            # extend pulse plus post anneal time
            start = np.log10(self.analysis_time)
            stop = np.log10(self.post_time)
            at2 = np.logspace(start,stop,N) + self.pulses*self.dwell_time
            
            analysis_times = np.append(analysis_times,at2)
        
        return analysis_times
    
    

    
    def Record_Time(self,time,index):
        """ record actual time of analysis """
        
        self.actual_times[index] = time



        
    def Do_Analysis(self,lattice,index):
        """ Command from main simulation. Starts an analysis step. """
        
        # Update the Lattice and the lattice memoryview
        self.lattice = lattice
        self.Cy_Analysis.Update_Lattice_Data(lattice)
        
        # Coverage Analysis
        self.Cy_Analysis.Coverage(index)      # calculate coverage, surface, and phase
        if self.coverage == True:
            self.Cy_Analysis.RMS(index)           # calculate rms roughness
            self.Cy_Analysis.Reflectivity(index)  # calculate reflectivity
        
        # Island Analysis
        if self.island == True:
            self.Island_Analysis(index)
        
        # Structure Factor Analysis
        if self.structure == True:
            self.Structure_Factor(index)
        
        

    
    def Island_Analysis(self,index):
        """ do island analysis on a layer. (Coverage must be done before this function) """ 
        
        # select a layer and analyze
        for layer,count in enumerate(self.cov[:,index]):
            if count == 0 or count == 1.0:
                pass
            else:
                # reset the labels array
                self.Cy_Analysis.Reset_Labels()
                
                # run Hoshen-Kopleman cluster id algorithm
                self.Cy_Analysis.Hoshen_Kopelman(layer+self.substrate_layers)
               
                # run cluster analysis
                self.clusters = np.zeros([self.labels[0]],dtype=np.int32)
                self.Cy_Analysis.Cluster_Analysis(layer,index,self.clusters)
                
                # remove zeros from histogram for saving and store for analysis
                full_hist = np.vstack((self.isd[1:],self.island_bins))
                reduced_hist = np.delete(full_hist,np.where(full_hist[0] == 0),axis=1)
                                
                # save data in live memory
                self.island_data[layer].append(reduced_hist)   # save to live data
                self.island_index[layer].append(index)         # save the index
                
                

                
    def Structure_Factor(self,index):
        """ Calculate sturcture factor profile """
                
        # Calculate fast Fourier Transform of phase surface
        FT = np.fft.rfft2(self.phase)

        # Calculate Power Spectral density
        FT2 = self.norm*((np.abs(FT))**2)                  # normailzed version
        FT2[0,0] = 0.2*(2*FT2[1,0]+2*FT2[0,1]+FT2[1,1])    # replace the Qr = 0 pixel with it's neighbors to avoid blow up
        
        if self.gaussian == 0:
            pass
        elif self.gaussian > 0:
            FT2 = gaussian_filter(FT2,sigma=self.gaussian,order=0)
        
        # Do circular average (data is stored directly into Cir_Avg)
        self.Cy_Analysis.Circ_Avg(FT2,index)
        
        

    
    def Return_Data(self):
        """ Return set of analyzed data """
        
        Data = []
        Time_Data = []  # list of time data points
        Cov_Data = []   # list of coverage analysis data
        Isl_Data = []   # list of island analysis data
        SF_Data = []    # list of SF data
                
        Data.append(self.layers)
        
        # time data
        Time_Data.append(self.analysis_times)
        Time_Data.append(self.actual_times)
        
        # coverage analysis
        if self.coverage == True:
            Cov_Data.append(self.cov)
            Cov_Data.append(self.refl)
            Cov_Data.append(self.rms)
            Cov_Data.append(self.avg_height)
            
        # island analysis
        if self.island == True:
            Isl_Data.append(self.num_clusters)
            Isl_Data.append(self.island_data)
            Isl_Data.append(self.island_index)
            
        # structure factor analysis
        if self.structure == True:
            SF_Data.append(self.structure_factor)
            
        # combine into single list
        Data.append(Time_Data)
        Data.append(Cov_Data)
        Data.append(Isl_Data)
        Data.append(SF_Data)

        return Data



        
    def Plot(self):
        """ Plotting method for analyzed results """
        
        # coverage and reflectivity plots
        if self.coverage == True:
            
            # Coverage plot (only plot growing layers)
            fig,ax1 = plt.subplots(figsize=(18,5))
            fig.subplots_adjust(left=0.1,right=0.75)
            ax1.plot(self.actual_times,np.ones([self.actual_times.shape[0]]),label='Layer 0')
            for i in range(0,self.cov.shape[0]):
                if np.sum(self.cov[i,:]) > 0:
                    ax1.plot(self.actual_times,self.cov[i,:],label='Layer '+str(i+1))
#            ax1.plot(self.actual_times,self.cov[i+1,:],label='Layer '+str(i+1))
            ax1.legend(bbox_to_anchor=(1.17, 1.0))
            ax1.set_ylim(-0.1,1.1)
            ax1.set_title('Layer Coverage',fontsize=16)
            ax1.set_ylabel('Coverage',fontsize=14)
            ax1.set_xlabel('Simulation Time (s)',fontsize=14)
            
            # Reflectivity plot
            fig2,ax2 = plt.subplots(figsize=(15,5))
            
            ax2.plot(self.actual_times,self.refl)
            ax2.set_title('Specular Reflectivity at Anti-Bragg',fontsize=16)
            ax2.set_ylabel('Reflectivity',fontsize=14)
            ax2.set_xlabel('Simulation Time (s)',fontsize=14)
            
            # RMS plot
            fig3,ax3 = plt.subplots(figsize=(15,5))
            
            ax3.plot(self.actual_times,self.rms)
            ax3.set_title('Rms Roughness of Surface',fontsize=16)
            ax3.set_ylabel('rms Roughness',fontsize=14)
            ax3.set_xlabel('Simulation Time (s)',fontsize=14)
        
        # island counts plot
        if self.island == True:
            fig4,ax4 = plt.subplots(figsize=(18,5))
            fig4.subplots_adjust(left=0.1,right=0.75)
            ax4.plot(self.actual_times,np.ones([self.actual_times.shape[0]]),label='Layer 0')
            for i in range(0,self.num_clusters.shape[0]):
                if np.sum(self.cov[i,:]) > 0:
                    ax4.plot(self.actual_times,self.num_clusters[i,:],label='Layer '+str(i+1))
            ax4.legend(bbox_to_anchor=(1.17, 1.0))
            ax4.set_title('Total Clusters by Layer (not including monomers)',fontsize=16)
            ax4.set_ylabel('Number of Clusters',fontsize=14)
            ax4.set_xlabel('Simulation Time (s)',fontsize=14)
        
        # structure factor plot (change to pcolormesh for the non linear analysis times)
        if self.structure == True:
            tt = np.tile(self.actual_times,(self.structure_factor.shape[0],1))   # time data points
            qr = np.tile(self.sub_q_bins,(self.structure_factor.shape[1],1))  # qr data points  
            qr = np.transpose(qr) 
            fig5,ax5 = plt.subplots(figsize=(15,5))
            ax5.pcolormesh(tt.T,qr.T,self.structure_factor.T)
            ax5.set_title('Structure Factor',fontsize=16)
            ax5.set_ylabel(r'Q$_r$ (a$^{-1}$)',fontsize=14)
            ax5.set_xlabel('Simulation Time (s)',fontsize=14)
            plt.gca().invert_yaxis()






#----------------------------------------------------------------------------------------------------------------------------------------------------------


##########################################
###  Cython Extension: On_the_Fly_Cy   ###
##########################################
    
cdef class On_the_Fly_Cy:
    """  Contain Analysis function for cluster labeling, coverage analysis, and strucutre factor analysis
    
    Unique Attributes
    -----------------
    nx,ny:   layer dimensions
    FT:      fourier transform array of layer
    ix,iy:   pixel lists for circular bins
    
    
    Methods
    -------
    Update_Lattice_Data:  Update lattice at new time step
    
    Coverage:      Calculate layer coverage, avg height, and phase of surface.
    RMS:           Calculate rms roughness of surface.
    Reflectivity:  Get anti-bragg specular intensity.
    
    Reset_Labels:     Resets the equivalence class array for a new analysis.
    Find:             Find lowest cluster label of connected clusters.
    Union:            Set two connected clusters to same cluster label.
    Hoshen_Copelman:  Cluster labeling algorithm.
    
    Cluster_Analysis:   Count clusters and cluster size.
    Size_Distribution:  Make histogram of cluster size distribution.
    Total_Islands:      Calculate total number of islands not including monomers.
    
    Create_Bins:  Create circular bins for structure factor analysis.
    Circ_Avg:     Get circular average.
    Smooth:       Temporary smoothing alogirthm for circular profile (not working very well)
    
    """



    cdef:
        np.int32_t substrate_layers, nx, ny, island_cutoff
        double inv_area
        double[:,::1] cov
        double[::1] refl     
        double[::1] rms
        double[::1] avg_height
        np.int32_t[:,::1] num_clusters
        double[::1] percolation_threshold
        np.int32_t[::1] isd
        np.int32_t sub_pix
        double[:,::1] kx
        double[:,::1] ky
        double[::1] q_bins
        double dq     
        double[:,::1] structure_factor
              
        np.int32_t[:,:,::1] lattice
        np.int32_t[:,::1] surface
        np.int32_t[:,::1] phase
        np.int32_t[:,::1] L
        np.uint32_t[::1] labels
        np.int32_t[::1] clusters
        
        np.int32_t bin_length
        double frac,sub_size,pix_size
        vector[vector[np.int32_t]] ix,iy    
        vector[np.int32_t] store    
        vector[vector[double]] area
        vector[double] store2    
        vector[np.int32_t] size
    
    
               
    def __init__(self, np.int32_t substrate_layers, np.int32_t nx, np.int32_t ny, double inv_area, np.int32_t island_cutoff, np.int32_t[:,::1] surface, np.int32_t[:,::1] phase, np.int32_t[:,::1] L, np.uint32_t[::1] labels, double[:,::1] cov, double[::1] refl, double[::1] rms, double[::1] avg_height, np.int32_t[:,::1] num_clusters, double[::1] percolation_threshold, np.int32_t[::1] isd, np.int32_t sub_pix, double[:,::1] kx, double[:,::1] ky, double[::1] q_bins, double dq, double[:,::1] structure_factor):
        
        # variables
        self.substrate_layers = substrate_layers
        self.nx = nx
        self.ny = ny
        self.inv_area = inv_area
        self.island_cutoff = island_cutoff
        
        # memoryviews
        self.surface = surface
        self.phase = phase
        self.L = L
        self.labels = labels
        self.cov = cov
        self.refl = refl
        self.rms = rms
        self.avg_height = avg_height
        self.num_clusters = num_clusters
        self.percolation_threshold = percolation_threshold
        self.isd = isd
        self.sub_pix = sub_pix
        self.kx = kx
        self.ky = ky
        self.q_bins = q_bins
        self.dq = dq
        self.structure_factor = structure_factor
        
        # make the circular average bin vectors
        self.bin_length = self.q_bins.shape[0]
        self.frac = 1/(self.sub_pix*self.sub_pix)    # fractional area of each subpixel
        self.sub_size = self.dq/self.sub_pix   # sub pixel size
        self.pix_size = 0.5*(self.ky[1,0]-self.ky[0,0])   # size of half a pixel
        
        # populate the bins vectors
        for i in range(self.bin_length):
            self.ix.push_back(self.store)
            self.iy.push_back(self.store)
            self.area.push_back(self.store2)
            self.size.push_back(0)
            
    

        
    # Update Lattice Data    
    cpdef void Update_Lattice_Data(self, np.int32_t[:,:,::1] lattice):
        """ Update the Lattice Data """
        self.lattice = lattice
        
    
    
    
    #########################
    ###   Phase Methods   ###
    #########################
    
    
#    cdef void Phase_General(self, Py_ssize_t i, Py_ssize_t j, np.double q, np.double a):
#        """ Calculate phase surface from height at Qz. Future. Not trivial, need complex exp. """
#        
#        pass
#    
#    
#    
#    cdef void Phase_Anti_Bragg(self, Py_ssize_t i, Py_ssize_t j):
#        """ Special Case of Anti-Bragg Qz. """
#        
#        if modulus(self.surface[i,j],2) != 0:
#            self.phase[i,j] = 1
#        elif modulus(self.surface[i,j],2) == 0:
#            self.phase[i,j] = -1
#            
#            
#            
#    cpdef void Phase(self):
#        """ Calculate Phase Array for full surface """
#        cdef Py_ssize_t i,j,k
#        
#        for i in range(self.nx):
#            for j in range(self.ny):
                
    
    
        
    ############################
    ###   Coverage Methods   ###
    ############################
        
    cpdef void Coverage(self, np.int32_t index):
        """ Calculate Layer Coverage, Surface Height, and Phase """
        cdef Py_ssize_t i,j,k
        cdef np.int32_t height
    
        height = 0
        for i in range(self.nx):
            for j in range(self.ny):
                # reset surface height
                self.surface[i,j] = 0
                
                # calculate coverage and height
                for k in range(0,self.cov.shape[0]):
                    if self.lattice[i,j,k+self.substrate_layers] > 0:
                        self.cov[k,index] += 1
                        self.surface[i,j] += 1
                        
                height += self.surface[i,j]  # for avg height
                        
                # calculate phase        
                if modulus(self.surface[i,j],2) != 0:
                    self.phase[i,j] = 1
                elif modulus(self.surface[i,j],2) == 0:
                    self.phase[i,j] = -1
        
        # calculate avg height            
        self.avg_height[index] = height*self.inv_area
        
        # scale to area of substrate            
        for i in range(0,self.cov.shape[0]):
            self.cov[i,index] = self.cov[i,index]*self.inv_area        
        
        
        
    cpdef void RMS(self, np.int32_t index):
        """ Calculate rms roughness of surface """
        cdef Py_ssize_t i,j,k
        cdef double mean_square
        
        mean_square = 0
        for i in range(self.nx):
            for j in range(self.ny):
                mean_square += (self.avg_height[index]-self.surface[i,j])**2
                
        self.rms[index] = sqrt(mean_square*self.inv_area)
        

    
    cpdef void Reflectivity(self, np.int32_t index):
        """ Anti-Bragg Reflectivity """
        cdef np.int32_t i
        cdef double complex R
        cdef double Refl
        
        R = 1+0j
        for i in range(self.cov.shape[0]):
            R += 2*((-1)**(i+1))*self.cov[i,index]
        
        self.refl[index] = R.real**2 + R.imag**2

    
    
    
    ############################################
    ###   Island Size Distribution Methods   ###
    ############################################
    
    cpdef void Reset_Labels(self):
        """ Reset the labels array """
        cdef np.uint32_t i
        
        for i in range(self.labels.shape[0]):
                self.labels[i] = i
        self.labels[0] = 1
        
        
    
    cdef np.int32_t Find(self, np.int32_t x):
        """ Find lowest label of equivalence class """
        cdef np.int32_t y,z
        
        y = x
        while self.labels[y] != y:
            y = self.labels[y] 
        while self.labels[x] != x:
            z = self.labels[x]
            self.labels[x] = y
            x = z      
            
        return y
    
    
    
    cdef void Union(self, np.int32_t x, np.int32_t y):
        """ Sets two cluster to same equivalence class """
        self.labels[self.Find(x)] = self.Find(y)
    
    
           
    cpdef void Hoshen_Kopelman(self, np.int32_t layer):
        """ Identify and Label all Clusters in a layer using Hoshen-Kopelman algorithm 
        
        Inputs
        ------
        L:  One layer of the simulation lattice
        labels:  list of equivalence classes
        
        Outputs
        -------
        L: Array updated with cluster labels
        
        """
        
        cdef np.int32_t x,y,i,j,left,up,right,down,site
        
        # Convert layer to -1 = occupied, 0 = unoccupied    
        for i in range(self.nx):
            for j in range(self.ny):
                self.L[i,j] = 0   # reset
                if self.lattice[i,j,layer] > 0:
                    self.L[i,j] = -1
       
        # First Loop - Pass through lattice to label all clusters
        # Scan up to last rown
        for x in range(0,self.nx):
            for y in range(self.ny):
                if self.L[x,y] == -1:
                    left = self.L[x,modulus(y-1,self.ny)]
                    up = self.L[modulus(x-1,self.nx),y]
                    if left < 1 and up < 1:
                        self.L[x,y] = self.labels[0]
                        self.labels[0] += 1
                    elif left > 0 and up < 1:
                        self.L[x,y] = self.Find(left)
                    elif left < 1 and up > 0:
                        self.L[x,y] = self.Find(up)
                    elif left > 0 and up > 0:
                        self.Union(left,up)
                        self.L[x,y] = self.Find(left)
                            
            # check right periodic boundary condition
            if self.L[x,self.ny-1] > 0:
                site = self.L[x,self.ny-1]
                right = self.L[x,0]
                if right > 0:
                    self.Union(right,site)
                    self.L[x,self.ny-1] = self.Find(right)
                                
        # check bottom periodic boundary condition
        for y in range(self.ny):
            if self.L[self.nx-1,y] > 0:
                site = self.L[self.nx-1,y]
                down = self.L[0,y]
                if down > 0:
                    self.Union(site,down)
                    self.L[self.nx-1,y] = self.Find(site)
        
        # Second Loop - Pass through lattice again to relabel in ascending order     
        for x in range(0,self.nx):
            for y in range(self.ny):
                if self.L[x,y] > 0:
                    self.L[x,y] = self.Find(self.L[x,y])

        
    
    cpdef void Cluster_Analysis(self, np.int32_t layer, np.int32_t index, np.int32_t[::1] clusters):
        """ Run analysis on cluster data in a layer """
        cdef Py_ssize_t i,j
        cdef np.int32_t m
        
        self.clusters = clusters
        
        # Initial Loop through the lattice
        for i in range(self.nx):
            for j in range(self.ny):
                
                # check if occupied
                if self.L[i,j] > 0:
                    m = self.L[i,j]
                    
                    # add to cluster count
                    self.clusters[m] += 1
        
        # loop through clusters and create histogram
        self.Size_Distribution()
        
         # loop through histogram and count total islands (this updates Num_Clusters)
        self.Total_Islands(layer,index)
                    
                    
                                        
    cdef void Size_Distribution(self):
        """ Calculation Island Size Distribution Histogram """
        cdef Py_ssize_t i
        cdef np.uint32_t m
        
        for i in range(self.isd.shape[0]):
            self.isd[i] = 0  
        
        for i in range(self.clusters.shape[0]):
            m = self.clusters[i]  # size of cluster
            self.isd[m] += 1
    
    
    
    cdef void Total_Islands(self, np.int32_t layer, np.int32_t index):
        """ Count total number of islands not including monomers """
        cdef Py_ssize_t i
        
        for i in range(self.island_cutoff+1,self.isd.shape[0]):
            if self.isd[i] > 0:
                self.num_clusters[layer,index] += self.isd[i]
    
    
                
                
    
    ####################################
    ###   Structure Factor Methods   ###
    ####################################
            
    cdef double Radius(self, double x, double y):
        """ calculate radius """

        cdef double r

        r = sqrt(x**2 + y**2)

        return r
    
    
    
    cpdef list Return_Area(self, list areas):
        """ return area vector as list """
        cdef Py_ssize_t i,j
        
        for i in range(self.bin_length):
            for j in range(self.area[i].size()):
                areas[i].append(self.area[i][j])
            
        return areas
    
    
    
    cpdef list Return_X(self, list IX):
        """ return x pixel vector as list """
        cdef Py_ssize_t i,j
        
        for i in range(self.bin_length):
            for j in range(self.ix[i].size()):
                IX[i].append(self.ix[i][j])
            
        return IX
    
    
    
    cpdef list Return_Y(self, list IY):
        """ return y pixel vector as list """
        cdef Py_ssize_t i,j
        
        for i in range(self.bin_length):
            for j in range(self.iy[i].size()):
                IY[i].append(self.iy[i][j])
            
        return IY
    
    
    
    cpdef list Return_Size(self, list SIZE):
        """ return size vector """
        cdef Py_ssize_t i,j
        
        for i in range(self.bin_length):
            SIZE.append(self.size[i])
            
        return SIZE
        
            
       
    @cython.cdivision(True) 
    cpdef void Create_Bins(self):
        """ set up circular bins and fractional areas """

        cdef np.int32_t i,j,k,m,n,q
        cdef double r1,r2,r3,r4,r5,r_min,r_max
        cdef double x,y,r
        cdef np.int32_t r_ind,min_ind,max_ind

        # loop the array and calculate sub pixel area factors
        q = 0
        for i in range(self.kx.shape[0]):
            for j in range(self.ky.shape[1]):
                q += 1
                # get center of mass coordinate
                x = self.kx[i,j]
                y = self.ky[i,j]

                # calculate r_min and r_max for the pixel (include center of pixel for the 0 case)
                r1 = self.Radius(x,y)
                r2 = self.Radius(x,y+2*self.pix_size)
                r3 = self.Radius(x+2*self.pix_size,y)
                r4 = self.Radius(x+2*self.pix_size,y+2*self.pix_size)
                r_min = min(r1,r2,r3,r4)
                r_max = max(r1,r2,r3,r4)

                # find corresponding min and max indices
                min_ind = int(r_min/self.dq)
                max_ind = int(r_max/self.dq)

                # expand ix,iy,size,area array for all indices
                for k in range(min_ind,max_ind+1):
                    self.ix[k].push_back(j)
                    self.iy[k].push_back(i)
                    self.size[k] = self.ix[k].size()  # current length of ix which is index of last value
                    self.area[k].push_back(0)

                # loop through sub-pixels and add fractional area
                for m in range(self.sub_pix):
                    for n in range(self.sub_pix):
                        # get the qr index
                        r = self.Radius(x+m*self.sub_size,y+n*self.sub_size)
                        r_ind = int(r/self.dq) # pixel index

                        # add sub pixel worth of area to area vector at proper index                        
                        self.area[r_ind][self.size[r_ind]-1] += self.frac


                        
    @cython.cdivision(True) 
    cpdef void Circ_Avg(self, double [:,:] PSD, np.int32_t index):
        """ Run the circular average """
        cdef Py_ssize_t i,j
        cdef double tot_area
        
        for i in range(self.structure_factor.shape[0]):
            tot_area = 0
            for j in range(self.size[i]):
                self.structure_factor[i,index] += self.area[i][j]*PSD[self.iy[i][j],self.ix[i][j]]
                tot_area += self.area[i][j]
            self.structure_factor[i,index] = self.structure_factor[i,index]/tot_area
            




                            


#----------------------------------------------------------------------------------------------------------------------------------------------------------
                    
#####################
###   Functions   ###
#####################
                
            
                
                
                

           

    
    
    
# End of Module