from dataclasses import dataclass
from decimal import Decimal
import warnings
from beartype import beartype
import numpy as np
import networkx as nx
from bloqade.atom_arrangement import ListOfLocations


@dataclass(frozen=True)
class MISProblem:
    positions: np.ndarray[Decimal, 2]
    graph: nx.Graph
    udg_radius: Decimal

    @staticmethod
    @beartype
    def create(positions: np.ndarray, graph: nx.Graph) -> "MISProblem":
        from bloqade.mis.rowify import find_UDG_radius

        if positions.shape[0] != graph.number_of_nodes():
            raise ValueError(
                "Number of positions must be equal to number of nodes in graph!"
            )

        if positions.shape[1] != 2:
            raise ValueError("Positions must be 2D!")

        def convert(x):
            return Decimal(str(x))

        positions = np.asarray(list(map(convert, positions.ravel()))).reshape(
            positions.shape
        )

        udg_radius, rmin, rmax = find_UDG_radius(positions, graph)

        return MISProblem(
            positions=positions,
            graph=graph,
            udg_radius=udg_radius,
        )

    def rowify(
        self,
        row_threshold: Decimal = Decimal("4.0"),
        horizontal_threshold: Decimal = Decimal("4.0"),
        precision: Decimal = Decimal("0.01"),
    ) -> "MISProblem":
        from bloqade.mis.rowify import rowify

        positions_out, _, _, rms_error = rowify(
            self.positions, row_threshold, horizontal_threshold, precision
        )

        if rms_error >= 0.1:
            warnings.warn(f"RMS error of rowification is {rms_error}. ")

        if positions_out.shape != self.positions.shape:
            warnings.warn(
                "Rowification failed, points are too close together to be "
                "rowified with the given thresholds. "
            )

        return MISProblem(
            positions=positions_out, graph=self.graph, udg_radius=self.udg_radius
        )

    def to_register(self) -> ListOfLocations:
        reg = ListOfLocations()
        for pos in self.positions:
            reg = reg.add_position((pos[0], pos[1]))

        return reg


@dataclass(frozen=True)
class NearlyMISProblem:
    positions: np.ndarray[float, 2]
    graph: nx.Graph

    def create(positions: np.ndarray, graph: nx.Graph) -> "NearlyMISProblem":
        positions = np.asarray(positions, dtype=float)

        if positions.shape[0] != graph.number_of_nodes():
            raise ValueError(
                "Number of positions must be equal to number of nodes in graph!"
            )

        if positions.shape[1] != 2:
            raise ValueError("Positions must be 2D!")

        return NearlyMISProblem(
            positions=positions,
            graph=graph,
        )

    def force_to_mis(self) -> MISProblem:
        raise NotImplementedError()

    def to_register(self) -> ListOfLocations:
        reg = ListOfLocations()
        for pos in self.positions:
            reg = reg.add_position((pos[0], pos[1]))

        return reg
