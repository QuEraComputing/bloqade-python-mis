# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:11:05 2022

A standardized graph visualization script.

@author: jwurtz
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib


def visualize_graph(
    G,
    ax=None,
    pos=None,
    label_nodes=False,
    color_nodes=3,
    color_edges=0,
    scale=1,
    vertex_edge=False,
):
    """
    Plots a networkx graph G with consistent formatting and coloring.
     This includes an option to choose the scale.
     The scale is choosen to fix the relative width of edges to vertices,
      and give the vertices a constant area on the plot.
     If the axis is undefined, the function will automatically choose an axis size
      such that each vertex is the same radius as any other plot. If the extent of
      the figure is too large (16x16), the script will automatically scale by
      powers of 2 until the graph fits within a small figure.

    This scaling option is crucial, as it allows for consistency across different graphs!

    G   - a networkx graph to plot.
    ax  - the figure axis to plot on. This is either
            matplotlib.axes._subplots.AxesSubplot : an axis to plot
            None : generate a new axis
    pos - positions of each vertex. This is either
            dict : keys are vertex labels, and values are positions.
            None : this generates a new set of postions based on the kamada_kawai_layout.
    label_nodes - If I should label vertices with their vertex labels.
            This should be used for debugging only, due to graph isomorphism properties
    color_nodes - The color for each vertex. This is either
            dict : keys are vertex labels, and values are color labels
                0   > Black.            Indicates an inert / ground state Rydberg atom
                1   > "Red"  #C2477F.   Indicates a Rydberg independent set excited atom
                2   > "Pink" #FFE5E5    Indicates an inert / ground state Rydberg atom against a dark background
                3   > "Blue" #6437FF    Default; used when a vertex is stateless
                str > a color descriptor
                      Unused:
                      "Yellow" #d4d45a
                      "Green"  #8ad679
            int  : A color label for every vertex
            str  : A color descriptor for every vertex
    color_edges - The line style for each edge. This is either
            dict : keys are edge labels, and values are style labels
                0   > Thick black line.   Used for geometric connections.
                1   > Thin grey line.     Used for nongeometric connections.
                2   > Thick pink line.   Used against dark backgrounds
               -1   > No line.
              tuple > (str, int). str indicates line color, int indicates line weight
            int  : A style label for every edge
            tuple: (str, int). str indicates line color, int indicates line weight
    scale - The relative size of vertices to the distance between vertices.
              The size of each edge stays in proportion to the size of each vertex.

    """

    # Define vertex colors from a pallite
    VERTEX_COLOR_LOOKUP = {0: "k", 2: "#FFE5E5", 3: "#00A2FF", 1: "#C2477F"}  # Old
    VERTEX_COLOR_LOOKUP = {
        0: "#333333",
        2: "#b5f8e6",
        3: "#6437FF",
        1: "#C2477F",
    }  # QuEra pallete

    # Define edge styles from a pallite
    EDGE_WIDTH_LOOKUP = {0: 3, 1: 1, 2: 3, -1: 0}
    EDGE_COLOR_LOOKUP = {0: "k", 1: "grey", 2: "#FFE5E5", -1: "white"}

    if type(pos) == type(None):
        pos = nx.drawing.layout.kamada_kawai_layout(G)
    X = np.array(list(pos.values()))
    position_extent = X.max(0) - X.min(0)

    # If no axis given, generate a new plot. The size will be chosen such that each
    #  vertex is a consistent size across plots.
    if ax == None:
        figsize = (
            2
            * position_extent
            * 1.2
            * np.sqrt(len(G.nodes))
            / scale
            / np.sqrt(np.prod(position_extent))
        )

        scaling = 0.5 ** np.ceil(np.log2(max(figsize) / 16))
        figsize *= scaling

        if scaling != 1:
            print("Figure scaled by 1/{:0.2f}x".format(1 / scaling))

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Determine scaling.
    ax.axis(
        [
            X.min(0)[0] - position_extent[0] * 0.1,
            X.max(0)[0] + position_extent[0] * 0.1,
            X.min(0)[1] - position_extent[1] * 0.1,
            X.max(0)[1] + position_extent[1] * 0.1,
        ]
    )
    ax.set_aspect("equal")
    ax.axis("off")

    pixelsize = ax.get_window_extent().size
    canvas_extent = np.sqrt(np.prod(pixelsize))

    # Choose vertex size to cover a fixed percentage of area at scale factor 1
    nodesize = 3 * scale * canvas_extent / np.sqrt(len(G.nodes))
    edgesize = 0.04 * scale * canvas_extent / np.sqrt(len(G.nodes)) / 3

    # Clean the node color list
    if type(color_nodes) == type(1):
        color_nodes2 = VERTEX_COLOR_LOOKUP[color_nodes]

    elif type(color_nodes) == type("string"):
        color_nodes2 = color_nodes
    elif len(color_nodes) >= len(G.nodes):
        color_nodes2 = []
        for node in G.nodes:
            color_nodes2.append(
                VERTEX_COLOR_LOOKUP.get(color_nodes[node], color_nodes[node])
            )
    else:
        print("WARNING: Color Nodes assignment not understood! Defaulting to black.")
        color_nodes2 = "k"

    # Clean the edge color list
    if type(color_edges) == type(1):
        color_edges2 = EDGE_COLOR_LOOKUP[color_edges]
        edge_weights = EDGE_WIDTH_LOOKUP[color_edges]
    elif type(color_edges) == type(("string", 1)):
        color_edges2 = color_edges[0]
        edge_weights = color_edges[1]
    elif len(color_edges) >= len(G.edges):
        color_edges2 = []
        edge_weights = []
        for edge in G.edges:
            if type(color_edges[edge]) == type(0):
                color_edges2.append(EDGE_COLOR_LOOKUP[color_edges[edge]])
                edge_weights.append(EDGE_WIDTH_LOOKUP[color_edges[edge]])
            elif type(color_edges[edge]) == type(("string", 1)):
                color_edges2.append(color_edges[edge][0])
                edge_weights.append(color_edges[edge][1])
            else:
                raise BaseException(
                    "Bad color_edges specifier! " + repr(color_edges[edge])
                )
    else:
        print("WARNING: Color Edges assignment not understood! Defaulting to black.")
        color_edges2 = EDGE_COLOR_LOOKUP[0]
        edge_weights = EDGE_WIDTH_LOOKUP[0]

    if vertex_edge == False:
        vertex_edge_radius = 0
    else:
        vertex_edge_radius = edgesize * np.array(edge_weights)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=nodesize**2 / 400,
        node_color=color_nodes2,
        linewidths=vertex_edge_radius,
        edgecolors="k",
    )
    nx.draw_networkx_edges(
        G, pos, width=edgesize * np.array(edge_weights), edge_color=color_edges2
    )

    if label_nodes == True:
        nx.draw_networkx_labels(G, pos)

    return ax


def weights_to_colors(weights, scheme="plasma"):
    """
    Converts a list of numerical weights to hex colors by matplotlib "plasma" colorscheme.
    The minimum weight is assigned the minimum color, and the maximum the maximum.
    """
    if np.isclose(min(weights), max(weights), atol=1e-10):
        weights_norm = 0.5 * np.ones(len(weights))
    else:
        weights_norm = (weights - min(weights)) / (max(weights) - min(weights))
    cmap = matplotlib.cm.get_cmap(scheme, 1024)
    color_nodes = [matplotlib.colors.to_hex(cmap(int(q * 1024))) for q in weights_norm]
    return color_nodes
