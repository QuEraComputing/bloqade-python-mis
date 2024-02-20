# -*- coding: utf-8 -*-
"""
Utilities to compute maximal independent sets using greedy strategies

@author: jwurtz
"""
from numpy import *
import networkx as nx


def get_mIS(G: nx.Graph, ordering=None):
    """
    Computes the maximal independent set given some random ordering
    G - a networkx graph object
    ordering (optional, None) - An ordered list of nodes of G, which are
        sequentially added to the independent set.
    """
    if type(ordering) == type(None):
        ordering = random.permutation(list(G.nodes))

    soln = set()
    for ctr in ordering:
        adj = G.neighbors(ctr)
        if len(soln.intersection(adj)) == 0:
            soln = soln.union([ctr])

    return soln

    binout = zeros(len(G.nodes))
    binout[list(soln)] = 1
    return binout


def greedy_mIS(G, soln: set = set(), dumb=False):
    """
    Computes a greedy maximal independent set by choosing the vertices
    which block the fewest number of additional vertices.
    Slightly non-deterministic-- the ordering within each block (eg all vertices
      which block 5 others) os ramdomized.
    G - networkx graph to compute the solution to
    soln - The initial independent set solution; optional.
    dumb - Boolian. Default False
        True: do a "zero local" greedy that isn't aware of edge structure
        False: do a "one local" greedy that picks the vertices with the fewest
               adjacent vertices.
    """
    new_one = True
    wctr = 0

    CUTOFF__ = 10_000

    if dumb == True:
        notinset = set(list(G.nodes)).difference(soln)
        ordering = (
            random.permutation(list(soln)).tolist()
            + random.permutation(list(notinset)).tolist()
        )
        return get_mIS(G, ordering)

    reduce = lambda ind, graph: graph.subgraph(
        set(graph.nodes).difference(graph.neighbors(ind)).difference([ind])
    )
    degree = lambda graph: array([graph.degree(ind) for ind in graph.nodes])

    Gsub = nx.Graph(G)
    for ind in list(soln):
        if ind in Gsub:
            Gsub = reduce(ind, Gsub)

    while len(Gsub.nodes) > 0:
        # print(Gsub.nodes)
        deg = degree(Gsub) + 1e-5 * random.rand(len(Gsub.nodes))
        minind = list(Gsub.nodes)[argmin(deg)]
        Gsub = reduce(minind, Gsub)
        soln.add(minind)
        # print(minind)

    return soln


def greedy_weighted_mIS(G, weights, soln=set(), bias=1):
    """
    A 1-local greedy maximal independent set, which iteratively selects the vertex
    with the highest weight. This can be modified by a bias term, which alternatively
    weights vertices with low connectivity.
    G - networkx graph to compute the solution to
    weights - A list of weights to each vertex in the graph.
              !! Note that this implicitly assumes the graph labels are integers 0..N-1 !!
    soln - The initial independent set solution; optional.
    bias - The bias between choosing vertices with high weight, vs choosing vertices
           with low connectivity.
           For bias >> 1, the algorithm selects the vertex whose neighbors are the lowest weight
           For bias = 0, the algorithm selects the vertex that is the highest weight.
           For bias = 1 (default), the algorithm equally balances the two.
    """

    reduce = lambda ind, graph: graph.subgraph(
        set(graph.nodes).difference(graph.neighbors(ind)).difference([ind])
    )
    degree = lambda graph: array(
        [
            bias * sum(weights[list(graph.neighbors(ind))]) - weights[ind]
            for ind in graph.nodes
        ]
    )

    Gsub = nx.Graph(G)
    for ind in list(soln):
        if ind in Gsub:
            Gsub = reduce(ind, Gsub)

    while len(Gsub.nodes) > 0:
        # print(Gsub.nodes)

        deg = degree(Gsub) + 1e-5 * random.rand(len(Gsub.nodes))

        minind = list(Gsub.nodes)[argmin(deg)]
        Gsub = reduce(minind, Gsub)
        soln.add(minind)
        # print(minind)

    return soln


def greedy_remove(G, soln, dumb=False):
    """
    Greedily removes vertices that violate independent set constraint
    G - graph
    soln - indices which are part of the independent set. May have violations.
    """
    Gsub = G.subgraph(soln)
    return greedy_mIS(Gsub, dumb=dumb, soln=set())
