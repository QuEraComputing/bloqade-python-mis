# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:09:47 2023

@author: jwurtz
"""
from numpy import *
import networkx as nx

def get_mIS(G,ordering=None):
    """
    Computes the maximal independent set given some random ordering
    """
    if type(ordering)==type(None):
        ordering = random.permutation(list(G.nodes))
    
    soln = set()
    for ctr in ordering:
        adj = G.neighbors(ctr)
        if len(soln.intersection(adj))==0:
            soln = soln.union([ctr])
    
    return soln

    binout = zeros(len(G.nodes))
    binout[list(soln)] = 1
    return binout


def greedy_mIS(G,soln = set(),dumb = False):
    """
    Computes a greedy maximal independent set by choosing the vertices
    which block the fewest number of additional vertices.
    Slightly non-deterministic-- the ordering within each block (eg all vertices
      which block 5 others) os ramdomized.
    """
    new_one = True
    wctr = 0
    
    CUTOFF__ = 10_000
    
    if dumb==True:
        notinset = set(list(G.nodes)).difference(soln)
        ordering = random.permutation(list(soln)).tolist() + random.permutation(list(notinset)).tolist()
        return get_mIS(G,ordering)

    
    reduce = lambda ind,graph: graph.subgraph(set(graph.nodes).difference(graph.neighbors(ind)).difference([ind]))
    degree = lambda graph: array([graph.degree(ind) for ind in graph.nodes])
    
    Gsub = nx.Graph(G)
    for ind in list(soln):
        if ind in Gsub:
            Gsub = reduce(ind,Gsub)

    while len(Gsub.nodes)>0:
        #print(Gsub.nodes)
        deg = degree(Gsub) + 1e-5*random.rand(len(Gsub.nodes))
        minind = list(Gsub.nodes)[argmin(deg)]
        Gsub = reduce(minind,Gsub)
        soln.add(minind)
        #print(minind)

    return soln


def greedy_weighted_mIS(G,weights,soln = set(),bias = 1,verbose=False,pos=None):
    
    if type(pos)==type(None) and verbose==True:
        pos = nx.drawing.layout.kamada_kawai_layout(G)
    
    if verbose==True:
        colors = weights_to_colors(weights)
    
    reduce = lambda ind,graph: graph.subgraph(set(graph.nodes).difference(graph.neighbors(ind)).difference([ind]))
    degree = lambda graph: array([bias*sum(weights[list(graph.neighbors(ind))]) - weights[ind] for ind in graph.nodes])
    
    Gsub = nx.Graph(G)
    for ind in list(soln):
        if ind in Gsub:
            Gsub = reduce(ind,Gsub)

    while len(Gsub.nodes)>0:
        #print(Gsub.nodes)
        
        
        
        deg = degree(Gsub) + 1e-5*random.rand(len(Gsub.nodes))
        
        if verbose==True:
            
            visualize_graph(G,pos=pos,color_nodes = colors)
            for ctr,ind in enumerate(Gsub.nodes):
                text(pos[ind][0],pos[ind][1],"{:0.3f}".format(deg[ctr]),backgroundcolor="w")
        
        minind = list(Gsub.nodes)[argmin(deg)]
        Gsub = reduce(minind,Gsub)
        soln.add(minind)
        #print(minind)
        
        if verbose== True:
            colors[minind] = 2

    return soln


def greedy_remove(G,soln,dumb = False):
    """
    Greedily removes vertices that violate independent set constraint
    G - graph
    soln - indices which are part of the independent set. May have violations.
    """
    Gsub = G.subgraph(soln)
    return(greedy_mIS(Gsub,dumb=dumb,soln=set()))


