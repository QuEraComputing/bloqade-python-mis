# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:57:39 2023

@author: jwurtz
"""

from postprocess.solution.base_solution import Solution
from postprocess.problem.MIS_problem import unit_disk_maximum_independent_set
from postprocess.utils.maximal_IS import greedy_mIS, greedy_remove
from postprocess.utils.udg_graph_mapping import logical_to_physical

import numpy as np

from bloqade import start
from bloqade.ir import var

class greedy_MIS_solution(Solution):
    """
    A hybrid greedy solution to MIS, implementing a type-1 postprocessing.
    In this implementation, a measurement from a neutral atom computer is used
    as a warm start for a 1-local greedy algorithm to post-process the solution
    into a locally optimal and valid maximal independent set.
    """
    
    
    def __init__(self, problem:unit_disk_maximum_independent_set, backend):
        """
        Initialize the quantum_MIS object.

        Args:
            problem (problem_MIS.maximum_independent_set): The MIS problem instance.
            backend (dict): Backend information for quantum computation.
        """
        super().__init__(problem,backend)
        self.problem = problem
        self._backend = backend
    
    
    
    def ansatz(self, quantum_parameters:dict):

        """
        Generate a parameterized quantum program for the MIS problem.

        Args:
            quantum_parameters (dict): Dictionary containing quantum parameters.

        Returns:
            Program: A parameterized quantum program.
        """
        
        atom_positions = logical_to_physical(
                positions = self.problem.positions,
                graph = self.problem.graph,
                radius = 8.0)
        
        
        time = quantum_parameters["time"]
        initial_detuning = quantum_parameters["initial_detuning"]
        final_detuning = quantum_parameters["final_detuning"]
        
        program_unassigned = start.add_position([tuple(q) for q in atom_positions]) \
            .rydberg.rabi.amplitude.uniform.piecewise_linear(
                durations = [var("time")*0.2, var("time")*.6, 0.2*var("time")],
                values = [0., 15, 15, 0.0])\
            .detuning.uniform.piecewise_linear(
                durations = [var("time")*0.2, var("time")*.6, 0.2*var("time")],
                values = ["initial_detuning","initial_detuning", "final_detuning","final_detuning"])
            
        program_assigned = program_unassigned.assign(
            time=time,
            initial_detuning= initial_detuning,
            final_detuning  = final_detuning)
        
        return program_assigned
    
    
    def get_solution(self, quantum_parameters:dict, classical_parameters:dict = {}) -> list[np.array]:
        """
        Compute candidate solutions for the MIS problem.

        Args:
            quantum_parameters (dict): Dictionary containing quantum parameters.
            classical_parameters (dict, optional): Dictionary containing classical parameters. Defaults to an empty dictionary.

        Returns:
            list[np.array]: List of candidate solutions for the MIS problem.
        """
        bitstrings = self.get_bitstrings(quantum_parameters)
        
        solutions = []
        for bitstring in bitstrings[0]:
            for ctr in range(1):
                quantum = set(np.where(np.array(bitstring)==0)[0])
                removed = greedy_remove(self.problem.graph, quantum)
                maximal = greedy_mIS(self.problem.graph, removed)
                
                payload = np.zeros(len(self.problem.graph.nodes),dtype=int)
                payload[list(maximal)] = 1
                solutions.append(payload)
        
        return solutions
    
    def penalty(self, quantum_parameters:dict, classical_parameters:dict={}) -> bool:
        """
        Check if the given quantum parameters exceed the coherence time.

        Args:
            quantum_parameters (dict): Dictionary containing quantum parameters.
            classical_parameters (dict, optional): Dictionary containing classical parameters. Defaults to an empty dictionary.

        Returns:
            bool: True if quantum parameters exceed coherence time, False otherwise.
        """
        total_time = quantum_parameters["time"]

        if total_time >= 3.9:
            return True
        else:
            return False
    
    def objective(self, quantum_parameters:dict, classical_parameters:dict = {}) -> float:
        """
        Compute the objective function value for the MIS problem.

        Args:
            quantum_parameters (dict): Dictionary containing quantum parameters.
            classical_parameters (dict, optional): Dictionary containing classical parameters. Defaults to an empty dictionary.

        Returns:
            float: Objective function value.
        """
        penalty = 1E3

        if self.penalty(quantum_parameters, classical_parameters) == True:
            return penalty 
        else:
            solns = self.get_solution(quantum_parameters, classical_parameters)
            return -np.average([self.problem.cost(soln) for soln in solns])
    
    @property
    def classical_parameters(self) -> list:
        """
        List of keys representing classical parameters.

        Returns:
            list: An empty list as there are no classical parameters.
        """
        return []
    
    @property
    def quantum_parameters(self) -> list[str]:
        """
        List of keys representing quantum parameters.

        Returns:
            list[str]: A list containing keys for quantum parameters.
        """
        return ["time", "initial_detuning", "final_detuning"]
        

class classical_greedy_MIS_solution(greedy_MIS_solution):
    """
    A classical-only solution to MIS, implementing a type-1 postprocessing.
    In this implementation, an all-zeros bitstring is used
    as a warm start for a 1-local greedy algorithm to post-process the solution
    into a locally optimal and valid maximal independent set.
    """
    def get_bitstrings(self,quantum_parameters):
        nsamples = self._backend["num_shots"]
        
        return [[np.zeros(len(self.problem.graph)) for ctr in range(nsamples)]]
        
    