from dataclasses import dataclass
from beartype import beartype
import numpy as np
import networkx as nx


@dataclass(frozen=True)
class MISProblem:
    positions: np.ndarray[np.float64, 2]
    graph: nx.Graph
    udg_radius: float

    @staticmethod
    @beartype
    def create(positions: np.ndarray[np.float64, 2], graph: nx.Graph) -> "MISProblem":
        min_radius = 0
        max_radius = np.inf

        for i, pos_i in enumerate(positions[:]):
            for j, pos_j in enumerate(positions[i + 1 :], i + 1):
                dist = np.linalg.norm(pos_i - pos_j)

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
