import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def visualize_UDG(
    positions,
    radii,
    ax=None,
    label_nodes=False,
    draw_disks=False,
    color_nodes=3,
    color_edges=0,
    scale=1,
    vertex_edge=False,
):
    """
    Draws a unit disk graph.
    positions - 2d positions of each vertex
    radii     - Float. radius of each vertex. Vertices are connected if the unit disks overlap.
                !! This is a factor of 2 from usual !!
    draw_disks - If unit disks should be drawn.
            True  : Draw disks for all vertices
            False : Do not draw disks
            list  : list of True False for each vertex
    color_nodes - The color for each vertex.
            int   : color assignment for all vertices.
                0   > Black.            Indicates an inert / ground state Rydberg atom
                1   > "Red"  #C2477F.   Indicates a Rydberg independent set excited atom
                2   > "Pink" #FFE5E5    Indicates an inert / ground state Rydberg atom against a dark background
                3   > "Blue" #6437FF    Default; used when a vertex is stateless
            str > a color descriptor for every vertex. eg #8ad679
            list  : color assignment for each vertex.
                     Each element may be an int or a string.

    ax          - axis to plot on. If None, generates a new axis that scales
                    so that every edge is a consistent size.
    color_edges - The color for each edge.
            dict : keys are edge labels, and values are style labels
                0   > Thick black line.   Used for geometric connections.
                1   > Thin grey line.     Used for nongeometric connections.
                2   > Thick pink line.   Used against dark backgrounds
               -1   > No line.
            tuple > (str, int). str indicates line color, int indicates line weight
            int  : A style label for every edge
            tuple: (str, int). str indicates line color, int indicates line weight
    label_nodes - label each vertex.
    scale       - scaling of vertex size. Smaller number means more area
    vertex_edge - Draw an edge around each vertex.
    """

    positions = np.array(positions)
    N = len(positions)

    # Construct the unit disk graph
    G = nx.Graph(
        (
            np.sqrt(
                (positions[:, 0:1] - positions[:, 0:1].T) ** 2
                + (positions[:, 1::] - positions[:, 1::].T) ** 2
            )
            + np.diag(np.ones(N)) * 1e99
        )
        < (2 * radii)
    )
    # Relabel vertices
    relabeling = {i: tuple(positions[i]) for i in range(N)}
    G = nx.relabel.relabel_nodes(G, relabeling)

    pos = {tuple(a): a for a in positions}

    if hasattr(color_nodes, "__len__"):
        color_nodes = {tuple(positions[i]): color_nodes[i] for i in range(N)}

    """
    ---------------------
    """
    X = positions
    position_extent = X.max(0) - X.min(0)

    # If no axis given, generate a new plot. The size will be chosen such that each
    #  edge is a consistent size across plots.
    if ax == None:
        figsize = position_extent / radii + 2

        scaling = 0.5 ** np.ceil(np.log2(max(figsize) / 16))
        figsize *= scaling

        if scaling != 1:
            print("Figure scaled by 1/{:0.2f}x".format(1 / scaling))

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        figsize = position_extent / radii + 2
        scaling = 0.5 ** np.ceil(np.log2(max(figsize) / 16))

    # Determine scaling.
    ax.axis(
        [
            X.min(0)[0] - radii,
            X.max(0)[0] + radii,
            X.min(0)[1] - radii,
            X.max(0)[1] + radii,
        ]
    )
    ax.set_aspect("equal")
    ax.axis("off")

    nodesize = 2.5 * 400 * scaling * scale
    edgesize = 4 * scaling * scale

    # Define vertex colors from a pallite
    VERTEX_COLOR_LOOKUP = {0: "k", 2: "#FFE5E5", 3: "#00A2FF", 1: "#C2477F"}
    VERTEX_COLOR_LOOKUP = {0: "#333333", 2: "#FF505D", 3: "#6437FF", 1: "#C2477F"}
    VERTEX_COLOR_LOOKUP = {
        0: "#333333",
        2: "#b5f8e6",
        3: "#6437FF",
        1: "#C2477F",
    }  # QuEra pallete

    # Define edge styles from a pallite
    EDGE_WIDTH_LOOKUP = {0: 3, 1: 1, 2: 3, -1: 0}
    EDGE_COLOR_LOOKUP = {0: "k", 1: "grey", 2: "#F0FF48", -1: "white"}
    # EDGE_COLOR_LOOKUP = {0:'#333333',1:'grey',2:'#E6E6E6',-1:'white'}

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

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=nodesize**2 / 400,
        node_color=color_nodes2,
        linewidths=vertex_edge_radius,
        edgecolors="k",
    )
    edges = nx.draw_networkx_edges(
        G, pos, width=edgesize * np.array(edge_weights), edge_color=color_edges2
    )

    if vertex_edge:
        nodes.set_edgecolor("w")
        nodes.set_linewidth(0.25 * edgesize * EDGE_WIDTH_LOOKUP[0])

    # Clean draw_disks variable
    if not hasattr(draw_disks, "__len__"):
        draw_disks = np.array([draw_disks] * N)

    for i in range(N):
        if draw_disks[i] == False:
            continue
        x1 = np.sin(np.linspace(0, 2 * np.pi, 101)) * radii
        y1 = np.cos(np.linspace(0, 2 * np.pi, 101)) * radii
        # print(x1)
        ax.plot(
            x1 + positions[i, 0],
            y1 + positions[i, 1],
            "--",
            color=VERTEX_COLOR_LOOKUP[1],
            linewidth=1,
            alpha=0.5,
        )
        ax.fill(
            x1 + positions[i, 0],
            y1 + positions[i, 1],
            color=VERTEX_COLOR_LOOKUP[1],
            zorder=-100,
            alpha=0.1,
        )

    # ax.fill_between([X.min(0)[0] - radii,X.max(0)[0] + radii],[X.min(0)[1] - radii]*2,[X.max(0)[1] + radii]*2,color="#fbfbfb",zorder=-1000)

    return ax, G


def get_UDG(positions, radii):
    positions = np.array(positions)
    N = len(positions)

    # Construct the unit disk graph
    G = nx.Graph(
        (
            np.sqrt(
                (positions[:, 0:1] - positions[:, 0:1].T) ** 2
                + (positions[:, 1::] - positions[:, 1::].T) ** 2
            )
            + np.diag(np.ones(N)) * 1e99
        )
        < (2 * radii)
    )
    return G
