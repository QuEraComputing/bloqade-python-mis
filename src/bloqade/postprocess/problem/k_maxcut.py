# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:59:12 2023

@author: jwurtz
"""

from bloqade.postprocess.problem.base_problem import Problem
from bloqade.utils.visualize_graph import visualize_graph
import bloqade.postprocess as postprocess


import networkx as nx
import numpy as np
import json


class k_maxcut(Problem):
    def __init__(
        self,
        graph: nx.Graph,
        positions: list[tuple[float, float]],
        nclusters: int,
        which: str,
    ):
        """
        A class defining k-maxcut and clustering, which seeks to minimize
        or maximize the number of edges in opposite partitions

        Parameters
        ----------
        graph : nx.Graph
            A graph describing the positions of each vertex
        positions : list[tuple[float,float]]
            Positions of each vertex in the graph. This should be
            close to a unit disk graph.
        nclusters : int
            Number of clusters to partition into
        which : string. "MIN" or "MAX"
            Switch to find the MINcut (clustering) or MAXcut (partition)

        Methods
        ----------
        cost(solution) : The k-maxcut cost of a particular solution
        show(solution): Render the problem instance
        from_file : Load a k-maxcut problem instance from file
        to_file : Dump a k-maxcut problem instance to file
        """

        self.graph = graph
        self.positions = positions
        self.nclusters = nclusters
        if which not in {"MIN", "MAX"}:
            raise KeyError("Bad which, must be MIN or MAX:", which)
        self.which = which

    def cost(self, solution: list[int]):
        """
        Compute the cost of a candidate solution
        The cost is the number of edges that need to be cut in order to
        separate the graph into [nclusters] disconnected subgraphs.
        """

        self._validate(solution)

        solution = np.array(solution)
        if self.which == "MIN":
            Nedges = len(self.graph.edges)
            for i in range(self.nclusters):
                Nedges -= len(self.graph.subgraph(np.where(solution == i)[0]).edges)

            return Nedges
        elif self.which == "MAX":
            Nedges = 0
            for i in range(self.nclusters):
                Nedges += len(self.graph.subgraph(np.where(solution == i)[0]).edges)

            return Nedges
        else:
            raise KeyError("Bad which, must be MIN or MAX:", self.which)

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
        self._validate(solution)

        colors = ["C{:0.0f}".format(p) for p in solution]

        nodes = list(self.graph.nodes)
        nodes.sort()
        ax = visualize_graph(
            self.graph,
            pos=dict(zip(nodes, self.positions)),
            color_nodes=colors,
            scale=1.41,
        )
        return ax

    def _validate(self, solution: list[int]):
        """
        Validate a candidate solution
        """
        assert len(solution) == len(
            self.graph.nodes
        ), "Solution must have the same size as the number of nodes"
        assert (
            max(solution) < self.nclusters
        ), "Solution values must be integers between 0 and [nclusters]-1"
        assert (
            min(solution) >= 0
        ), "Solution values must be integers between 0 and [nclusters]-1"

        return True

    def __str__(self):
        payload = {
            "edges": [list(q) for q in self.graph.edges],
            "positions": np.array(self.positions, dtype=float).tolist(),
            "nclusters": int(self.nclusters),
            "which": self.which,
            "signature": "problem_clustering",
        }

        strout = json.dumps(payload)
        return strout

    @classmethod
    def from_file(cls, record: postprocess.record.record):
        payload = json.loads(record.get(cls))
        graph = nx.Graph(payload["edges"])
        positions = np.array(payload["positions"])
        nclusters = payload["nclusters"]
        which = payload["which"]

        return cls(graph, positions, nclusters, which)

    def to_file(self, record: postprocess.record.record):
        record.put(self)
