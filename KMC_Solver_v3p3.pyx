#cython: language_level=3

# distutils: language = c++

# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

#################################################################################################################
#################################################################################################################
#########################                                                               #########################
#########################            PLD Kinetic Monte Carlo Solver Module              #########################
#########################                                                               #########################
#################################################################################################################
#################################################################################################################

"""
Kinetic Monte Carlo - Solver Module

Low-Level Cython class implementing the kinetic monte-carlo solver algorithm.

Version: 3.3a

"""

# import modules
import numpy as np

# cimport modules
cimport numpy as np
cimport cython
from libc.math cimport log
from libcpp.vector cimport vector
from libcpp.set cimport set as Sset
from libcpp.algorithm cimport lower_bound
from cython.operator cimport dereference as deref, preincrement as inc

# cimport KMC modules
from KMC_Miscellaneous_Cy_v3p3 cimport Ran






######################################
###   Cython Class: Solver Class   ###
######################################
#--------------------------------------------------------------------------------------------------------------------------------------------------


cdef class Solver:
    """
    Solver Class: Kinetic Monte Carlo main algorithm
    
    Attributes
    ----------
    solver_type:  which solver type is used
    rates:        array of elementary process rates (in order)
    length:       number of elementary processes
    time:         simulation time
    time_diff:    time to next diffusion event
    ni:           total number of particles in each process list
    ri:           partial rates for each process list
    total_rate:   total rate for all processes
    max_ni:       largest size of each process list
    
    cummulative_rates:  array of cummulative rates
    process_list:       nested vector of process lists
    process_list_set:   nested vector of set containers
    
    seed:  integer seed for random number generator
    Ran:   Random Number Generator object
    
    
    Methods
    -------
    Return_Cummulative_Rates:  return cummulative_rates to python
    Return_Process_Lists:      return process_list tp python
    Calc_tdiff:                calculate time to next diffusion process
    Calc_Total_Processes:      calculate total number of process available
    Calc_Partial_Rates:        calculate the partial rates of each elementary process
    Calc_Cummulative_Rates:    calculate the cummulative rates over allelementary processes
    Select_Process:            choose a process to execute
    Select_Particle:           select a particle from the chosen process list
    Select_Particle_set:       select a particle from the chosen process list (with set data structure)
    Update_Process_Linear:     update the process list (add or remove particle) with linear search
    Update_Process_Binary:     update the process list (add or remove particle) with binary search
    Update_Process_Set:        update the process list (add or remove particle) with binary seach on set data structure
    
    """
    
    
    
    def __init__(self, np.int32_t solver_type, double[::1] rates, np.int32_t length, double[::1] time, double[::1] time_diff, np.int32_t[::1] ni, double[::1] ri, double[::1] total_rate, np.int32_t[::1] max_ni, np.uint64_t seed):
        
        # Constants
        self.solver_type = solver_type
        self.rates = rates    # Elementary Process rates in order      
        self.length = length    # length of the rates array
        
        # Times
        self.time = time   # Simulation Time
        self.time_diff = time_diff    # time to next diffusion event
        
        # Rates
        self.ni = ni    # Total particles in each elementary process list      
        self.ri = ri    # Partial rate for each elementary process list
        self.total_rate = total_rate   # Total rate for all processes
        
        # debug data structures
        self.max_ni = max_ni
        
        # Cummulative Partial Rates (make this a vector for binary search)
        for i in range(self.length):
            self.cummulative_rates.push_back(0.0)
        
        # Create Process Lists (Vector of Vector version)
        for i in range(self.length):
            self.process_list.push_back(self.store)

        # Create Process Lists (Vector of Sets version)
        for i in range(self.length):
            self.process_list_set.push_back(self.store_set)
            
        # Create Random Number Generator
        self.seed = seed
        self.Rand = Ran(self.seed)


        
    cpdef void Return_Cummulative_Rates(self, double [::1] c_rates):
        """ Export cummulative rate data from vector to array. """
        cdef Py_ssize_t i

        for i in range(self.length):
            c_rates[i] = self.cummulative_rates[i]
            

    
    cpdef list Return_Process_Lists(self, list store):
        """ Export data in Process_Lists. """
        cdef Py_ssize_t i
        cdef np.int32_t j

        if self.solver_type == 0 or self.solver_type == 1:
            for i in range(self.length):
                for j in self.process_list[i]:
                    store[i].append(j)
        elif self.solver_type == 2:
            for i in range(self.length):
                for j in self.process_list_set[i]:
                    store[i].append(j)
        return store
            

        
    @cython.cdivision(True)
    cdef void Calc_tdiff(self):
        """ calculate delta time for next diff event. """
        self.time_diff[0] = -(1.0/self.total_rate[0])*log(self.Rand.Float64())
        
        
        
    cdef np.int32_t Calc_Total_Processes(self):
        """ calculate how many elementary processes are available. """
        cdef Py_ssize_t i
        cdef np.int32_t tot
        
        tot = 0
        for i in range(self.length):
            tot += self.ni[i]
            
        return tot
    
    
    
    cdef void Calc_Partial_Rates(self):
        """ calculates Partial Rates for each Elementary Process. """        
        cdef Py_ssize_t i

        for i in range(self.length):                    
            self.ri[i] = self.ni[i]*self.rates[i]


            
    @cython.cdivision(True)        
    cdef void Calc_Cummulative_Rates(self):
        """ calculates cummulative rate and total rate. """        
        cdef Py_ssize_t i
                 
        self.cummulative_rates[0] = self.ri[0]
        
        for i in range(1,self.length):                   
            self.cummulative_rates[i] = self.cummulative_rates[i-1] + self.ri[i]
                
        self.total_rate[0] = self.cummulative_rates[self.length-1]
            

    
    cdef np.int32_t Select_Process(self):
        """ select a process. """
        cdef np.int32_t process
        cdef double i = self.Rand.Float64_n(self.total_rate[0])
        cdef vector[double].iterator it  # iterator for binary search of vector
        
        # Select Process
        it = lower_bound(self.cummulative_rates.begin(),self.cummulative_rates.end(),i)
        process = it - self.cummulative_rates.begin()
        
        return process


        
    cdef np.int32_t Select_Particle(self, np.int32_t process):
        """ select corresponding particle to move. """
        cdef np.int32_t atom
        cdef np.uint64_t randint, tot_particles
        
        tot_particles = self.ni[process]   # total number of available particles to move        
        randint = self.Rand.Int64_n(tot_particles)  # select particle at random 
        atom = self.process_list[process][randint]  # get particle id
        
        return atom
    

    
    cdef np.int32_t Select_Particle_Set(self, np.int32_t process):
        """ select corresponding particle to move for set data structure """
        cdef np.int32_t i
        cdef np.int32_t atom
        cdef np.uint64_t randint, tot_particles
        cdef Sset[np.int32_t].iterator it
        
        tot_particles = self.ni[process]   # total number of available particles to move
        randint = self.Rand.Int64_n(tot_particles)  # select particle at random
        it = self.process_list_set[process].begin()  # seems to be raising fault
        for i in range(randint-1):
            inc(it)
        atom = deref(it)
        
        return atom
                
 
    
    cdef void Update_Process_Linear(self, np.int32_t atom, np.int32_t process):
        """ Update Process Lists using linear search - O(N) 
        
        This method work on the vector of a vector data structure
        
        'process' is the process list index. 
        If atom is positive, then add to list. If it is negative, substract from list.
        
        """        
        cdef np.int32_t j,test
        cdef vector[np.int32_t].iterator it
        
        # new process added: O(1)
        if atom > 0:   
            self.process_list[process].push_back(atom)    # add atom to list
            self.ni[process] += 1    # increment the total particle counter
            
            # keep track of max size of Ni
            if self.ni[process] > self.max_ni[process]:
                self.max_ni[process] += 1
            
        # delete process from list: O(N)
        elif atom < 0:
            atom = -atom   # turn atom back to positive number for indexing
            it = self.process_list[process].begin()  # iterator to search list
            j = 0
            test = self.process_list[process][j]
            while test != atom:
                j += 1
                test = self.process_list[process][j]
            it = it + j   
            self.process_list[process].erase(it)  # remove atom from list                
            self.ni[process] -= 1   # decrement the total particle counter       

                
    
    cdef void Update_Process_Binary(self, np.int32_t atom, np.int32_t process):
        """ Update Process Lists using binary search - O(log N) 
        
        This method work on the vector of a vector data structure. Requires that vectors are ordered
        in order to perform binary search.
        
        """        
        cdef vector[np.int32_t].iterator it

        # new process added: O(N log N)
        if atom > 0:
            it = lower_bound(self.process_list[process].begin(),self.process_list[process].end(),atom)
            self.process_list[process].insert(it,atom)    # add atom to list
            self.ni[process] += 1    # increment the total particle counter
            
            # keep track of max size of Ni
            if self.ni[process] > self.max_ni[process]:
                self.max_ni[process] += 1
            
        # delete process from list: O(log N)
        elif atom < 0:
            atom = -atom   # turn it back to positive number for indexing
            it = lower_bound(self.process_list[process].begin(),self.process_list[process].end(),atom)  
            self.process_list[process].erase(it)  # remove atom from list                
            self.ni[process] -= 1   # decrement the total particle counter 
            
            
            
    cdef void Update_Process_Set(self, np.int32_t atom, np.int32_t process):
        """ Update Process Lists using binary search with std::set - O(log N) 
        
        This method works on the vector of sets data structure.
        
        """
        # new process added: O(log N)
        if atom > 0:
            self.process_list_set[process].insert(atom)    # add atom to list
            self.ni[process] += 1    # increment the total particle counter
            
            # keep track of max size of Ni
            if self.ni[process] > self.max_ni[process]:
                self.max_ni[process] += 1
            
        # delete process from list: O(log N)
        elif atom < 0:
            atom = -atom   # turn it back to positive number for indexing
            self.process_list_set[process].erase(atom)  # remove atom from list                
            self.ni[process] -= 1   # decrement the total particle counter
            
            
            
            
            
            
            
            
            
            
# End of Module