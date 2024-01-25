# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:58:48 2023

@author: jwurtz
"""
from postprocess.solution.base_solution import Solution
from postprocess.problem.k_maxcut import k_maxcut
from postprocess.utils.udg_graph_mapping import logical_to_physical

import postprocess.utils.observables as observables

import numpy as np
import networkx as nx
import scipy

from bloqade import start
from bloqade.ir import var

class spectral_kmaxcut_solution(Solution):
    """
    A spectral clustering solution to the kmaxcut or clustering problem,
    implementing type-2 postprocessing
    In this implementation, the eigenvalues of a connected correlation function
    measured from the quantum device are used as a feature for k-means clustering.
    """
    
    def __init__(self,problem:k_maxcut, backend):
        """
        Quantum clustering of some imput graph,
        based on spectral clustering of the connected correlation function
        
        Parameters
        ----------
        problem - a k-maxcut problem instance
        backend - the dictionary describing the quantum backend.
        """
        super().__init__(problem, backend)
        self.problem = problem
        self._backend = backend
        
    
    def ansatz(self, quantum_parameters:dict, classical_parameters:dict={}):
        """
        Construct an ansatz given quantum parameters.
        Evolution is simply a quench from the |0> state
        for variable times.
        
        quantum_parameters:dict with keys
         times:list[float] - Total times to evolve to
         detuning:float - Detuning that is fixed over the entire evolution

        """
        
        atom_positions = logical_to_physical(
                positions = self.problem.positions,
                graph = self.problem.graph,
                radius = 8.0)
        
        # [ screams internally ]
        atom_positions = [tuple(q) for q in atom_positions]
        
        program_unassigned = start.add_position(atom_positions) \
            .rydberg.rabi.amplitude.uniform.piecewise_linear(
                durations = [0.06,"t1",0.06],
                values = [0., 15, 15, 0.0])\
            .detuning.uniform.piecewise_linear(
                durations = [var("t1")+0.12],
                values = ["detuning", "detuning"])
            
        program_assigned = program_unassigned.batch_assign(
            t1 = np.array([quantum_parameters["time1"], 
                          quantum_parameters["time2"],
                          quantum_parameters["time3"]]),
            detuning = np.array([quantum_parameters["delta1"],
                                quantum_parameters["delta2"],
                                quantum_parameters["delta3"]]))
        
        return program_assigned
            
        
    def _get_features_centers_partition(self,quantum_parameters,classical_parameters):
        """
        Helper function to compute the eigenvectors of the correlation function
        and implement k-means clustering
        """
        bitstrings = self.get_bitstrings(quantum_parameters)
        
        # Compute the connected correlation functions
        #correlation_function = []
        #for data in bitstrings:
        #    correlation_function.append(observables.connected_correlation(data.T.astype(int)))
        
        correlation_function = 0
        for k,data in enumerate(bitstrings):
            cor_weight = classical_parameters["w"+str(k)]
            correlation_function += cor_weight * observables.connected_correlation(data.T.astype(int))
        
        # Next, compute the eigenspectra of the correlation functions
        # to extract features for clustering.
        # This might need a negative sign and normalization of the diagonal?
        # The opposite sign might give you an approximate coloring.
        eigees = np.linalg.eigh(correlation_function)
        
        
        # The lowest eigenvectors of each correlation function serve
        # as the feature vector for clustering
        features = eigees[1][:,0:self.problem.nclusters]
        
        # Then feed it into the same classical procedure
        centers = scipy.cluster.vq.kmeans(features,self.problem.nclusters)[0]
        
        
        partition = np.array([np.argmin(np.sum((ff - centers)**2,1)) for ff in features],dtype=int)
    
        return features,centers,partition,correlation_function
    
    def get_solution(self, quantum_parameters:dict, classical_parameters:dict = {}):
        """
        Computes a clustering solution using spectral methods of
         the connected correlation function.

        """
        features,centers,partition,cf = self._get_features_centers_partition(quantum_parameters,classical_parameters)
        
        return partition
    
    def objective(self,quantum_parameters:dict, classical_parameters:dict = {},nsamples=20):

        if self.penalty(quantum_parameters, classical_parameters):
            return 1E3
        else:
            solns = [self.get_solution(quantum_parameters,classical_parameters) for ctr in range(nsamples)]
            return np.average([self.problem.cost(soln) for soln in solns])
    
    @property
    def classical_parameters(self):
        return ["w0","w1","w2"]
    
    @property
    def quantum_parameters(self):
        return ["time1", "time2", "time3", "delta1", "delta2", "delta3"]
    

    def penalty(self, quantum_parameters:dict, classical_parameters:dict={}) -> bool:
   
        time1 = quantum_parameters["time1"]
        time2 = quantum_parameters["time2"]
        time3 = quantum_parameters["time3"]

        if np.any(np.array([time1, time2, time3]) < 0):
            return True
        else:
            return False