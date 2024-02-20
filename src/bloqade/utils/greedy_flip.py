# -*- coding: utf-8 -*-
"""
Greedy FLIP solvers for the k-maxcut problem

@author: jwurtz
"""
import numpy as np

from bloqade.postprocess.problem.k_maxcut import k_maxcut


def single_flip(problem: k_maxcut, sol: list):
    """
    Find a single-vertex update to a problem solution that increases the
    problem objective the most

    Parameters
    ----------
    problem: postprocess.problem.k_maxcut.k_maxcut problem
      A k-maxcut problem
    sol: A candidate solution to the k-maxcut problem, as a list of colors

    Returns
    -------
    sol: A new candidate solution with a single vertex flipped that increases
         the cost function by the largest amount
    """
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
    # return sol_out


def greedy_flip(problem: k_maxcut, sol):
    """
    An interative greedy flip (e.g. zero temperature monte carlo) solver to the
    k-maxcut problem, which finds a locally optimal solution where exchanging
    any vertex color with another will lead to an increase in the objective value.

    Parameters
    ----------
    problem: postprocess.problem.k_maxcut.k_maxcut problem
      A k-maxcut problem
    sol: A candidate initial solution to the k-maxcut problem, as a list of colors

    Returns
    -------
    sol: A candidate solution that is iteratively updated from the original
         solution, and has the property that changing any single vertex color
         will not decrease the cost function (eg "locally optimal")
    """
    val = problem.cost(sol)
    val_candidate = 1e99
    for ctr in range(len(sol) * problem.nclusters):
        sol = single_flip(problem, sol)
        val_candidate = problem.cost(sol)
        # print(val_candidate)
        if val_candidate == val:
            break
        val = val_candidate
    return sol
