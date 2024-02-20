# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:35:45 2023

@author: jwurtz
"""
from bloqade.postprocess.problem.base_problem import Problem
from bloqade.utils.visualize_graph import visualize_graph
import bloqade.postprocess as postprocess
from bloqade.utils.visualize_UDG import get_UDG

import networkx as nx
import numpy as np
import json


class maximum_independent_set(Problem):
    """
    A class defining the maximum independent set problem

    Methods
    ----------
    cost(solution) : The MIS cost of a particular solution
    show(solution): Render the problem instance
    from_file : Load an MIS problem instance from file
    to_file : Dump an MIS problem instance to file
    """

    def __init__(
        self,
        graph: nx.Graph,
        positions: list[tuple[float, float]] = [],
    ):
        """
        Parameters
        ----------
        graph : nx.Graph
            A graph describing the positions of each vertex
        positions : list[tuple[float,float]] (Optional)
            Positions of each vertex in the graph. This is optional
        """

        self.graph = graph

        self.positions = positions

    def cost(self, solution: list[int], penalty=-1e10):
        """
        Compute the cost of a candidate solution
        The cost is the size of the independent set

        Parameters
        ----------
        solution - list[int]: a binary (0 or 1) list indexing if a vertex is
                   in (1) or not in (0) the independent set
        pentalty - The penalty for an independent set violation. Note that any
                   non-independent set will not pass validation
        """

        self._validate(solution)

        solution = np.array(solution)

        soln_set = set(np.where(solution == 1)[0])

        subg = self.graph.subgraph(soln_set)

        return penalty * len(subg.edges) + len(soln_set)

    def show(self, solution: list[int] = None, ax=None):
        """
        Visualize the problem.
        Solution - A trial solution to the clustering problem
        ax - A plt.ax object to draw on

        returns
        ax - The axis that is drawn on.

        """

        if solution is None:
            solution = np.zeros(len(self.graph.nodes))
        # self._validate(solution)

        if len(self.graph.nodes) == len(self.positions):
            pos = dict(zip(self.graph.nodes, self.positions))
        else:
            pos = None

        ax = visualize_graph(self.graph, pos=pos, color_nodes=solution, scale=1.41)
        return ax

    def _validate(self, solution: list[int]):
        """
        Validate a candidate solution
        """
        assert len(solution) == len(
            self.graph.nodes
        ), "Solution must have the same size as the number of nodes"
        assert max(solution) <= 1, "Solution values must be integers between 0 and 1"
        assert min(solution) >= 0, "Solution values must be integers between 0 and 1"

        solution_set = set(np.where(np.array(solution) == 1)[0])
        assert (
            self.graph.subgraph(solution_set).number_of_edges() == 0
        ), "Graph is not an independent set"

        return True

    @staticmethod
    def from_file(record: postprocess.record.record):
        payload = json.loads(record.get(maximum_independent_set))
        graph = nx.Graph(payload["edges"])
        positions = np.array(payload["positions"])

        return maximum_independent_set(graph, positions)

    def to_file(self, record: postprocess.record.record):
        record.put(self)

    def __str__(self):
        payload = {
            "edges": [list(q) for q in self.graph.edges],
            "positions": np.array(self.positions, dtype=float).tolist(),
            "signature": "maximum_independent_set",
        }
        strout = json.dumps(payload)
        return strout


class unit_disk_maximum_independent_set(maximum_independent_set):
    """
    A maximum independent set problem whose graph has the property that
    it is a unit disk graph, shown constructively via a list of positions.
    The graph is directly constructed from positions and a unit disk radius.
    """

    def __init__(self, positions: list[tuple[float, float]], ud_radius: float):
        """
        Parameters
        ----------
        positions : list[tuple[float,float]]
            Positions of each vertex in the graph. This is optional
        ud_radius: float
            Unit disk radius of the graph, defined as the RADIUS and NOT diameter,
            eg if |x_i - x_j| < 2*R then edge; in other words, edge iff disks overlap.
        """

        self.positions = positions
        self.graph = get_UDG(positions, ud_radius)
        self.ud_radius = ud_radius

    @staticmethod
    def from_file(record: postprocess.record.record):
        payload = json.loads(record.get(unit_disk_maximum_independent_set))
        # graph = nx.Graph(payload["edges"])
        positions = np.array(payload["positions"])
        ud_radius = payload["ud_radius"]

        return unit_disk_maximum_independent_set(positions, ud_radius)

    def __str__(self):
        payload = {
            "edges": [list(q) for q in self.graph.edges],
            "positions": np.array(self.positions, dtype=float).tolist(),
            "ud_radius": self.ud_radius,
            "signature": "unit_disk_maximum_independent_set",
        }
        strout = json.dumps(payload)
        return strout
