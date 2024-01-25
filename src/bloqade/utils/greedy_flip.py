# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:41:32 2023

@author: jwurtz
"""
import numpy as np

def single_flip(problem,sol):
    sol2 = np.copy(sol)
    val = problem.cost(sol)
    for ind in np.random.permutation(len(sol2)):
        for color in np.random.permutation(problem.nclusters):
            sol2 = np.copy(sol)
            sol2[ind] = color
            val_candidate = problem.cost(sol2)
            
            if val_candidate < val:
                sol_out = sol2
                val = val_candidate
                return sol_out
    return sol
    #return sol_out

def greedy_flip(problem,sol):
    
    val = problem.cost(sol)
    val_candidate = 1e99
    for ctr in range(len(sol)*problem.nclusters):
        sol = single_flip(problem,sol)
        val_candidate = problem.cost(sol)
        #print(val_candidate)
        if val_candidate == val:
            break
        val = val_candidate
    return sol