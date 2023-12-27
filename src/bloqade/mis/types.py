from dataclasses import dataclass
import numpy as np
import networkx as nx
from beartype import beartype


@dataclass(frozen=True)
class MISProblem:
    positions: np.ndarray[np.float64, 2]
    graph: nx.Graph
    udg_radius: float

    @staticmethod
    @beartype
    def create(positions: np.ndarray[np.float64, 2], graph: nx.Graph) -> "MISProblem":
        dists = dict()
        min_radius = 0
        max_radius = np.inf

        for i, pos_i in enumerate(positions[:]):
            for j, pos_j in enumerate(positions[i + 1 :], i + 1):
                dist = dists.get((i, j), np.linalg.norm(pos_i - pos_j))

                if (i, j) in graph.edges:
                    min_radius = max(min_radius, dist)
                else:
                    max_radius = min(max_radius, dist)

                if min_radius > max_radius:
                    raise ValueError("Provided graph is not a unit disk graph!")

        return MISProblem(
            positions=positions,
            graph=graph,
            udg_radius=np.sqrt(min_radius * max_radius),
        )
