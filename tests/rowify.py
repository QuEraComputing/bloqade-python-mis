from bloqade.mis.types import MISProblem
import networkx as nx
import numpy as np

# 5
# | \
# 2 - 3 - 4
# | X | /
# 0 - 1


graph = nx.Graph()
graph.add_edges_from(
    [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (3, 0),
        (1, 2),
        (3, 4),
        (1, 4),
        (2, 5),
        (5, 3),
    ]
)

positions = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [0, 2]])
problem = MISProblem.create(positions, graph)

print(problem)
