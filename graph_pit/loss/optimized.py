from dataclasses import dataclass
from functools import cached_property
from typing import Union, Callable
import torch
import numpy as np

import padertorch as pt

from graph_pit.loss.base import GraphPITBase, LossModule
from graph_pit.assignment import graph_assignment_solvers

__all__ = [
    'OptimizedGraphPITSourceAggregatedSDRLoss',
    'OptimizedGraphPITSourceAggregatedSDRLossModule',
    'optimized_graph_pit_source_aggregated_sdr_loss',
]


@dataclass
class OptimizedGraphPITSourceAggregatedSDRLoss(GraphPITBase):
    """
    It's called SDR3 because this is the third SDR variant I
    defined. It aggregates the energies of the target and error
    signals in the fraction.
    """
    permutation_solver: Union[Callable, str] = 'optimal_brute_force'

    def __post_init__(self):
        super().__post_init__()

        if not callable(self.permutation_solver):
            self.permutation_solver = (
                graph_assignment_solvers[self.permutation_solver]
            )(minimize=False)

    @cached_property
    def similarity_matrix(self):
        """
        # TODO: can this be optimized?

        Returns:
            Shape: (num_targets, num_estimates)
        """
        v = []
        for idx, (target, (start, stop)) in enumerate(zip(
                self.targets, self.segment_boundaries
        )):
            v.append(torch.sum(
                self.estimate[..., start:stop] * target[..., None, :],
                dim=-1
            ))
        similarity_matrix = torch.stack(v)
        return similarity_matrix

    @cached_property
    def best_coloring(self):
        similarity_matrix = self.similarity_matrix
        x = self.permutation_solver(
            pt.utils.to_numpy(similarity_matrix, detach=True), self.graph
        )
        assert x is not None
        return x

    @cached_property
    def loss(self):
        # Compute target and estimate energies
        target_energy = torch.sum(
            torch.stack([torch.sum(t ** 2) for t in self.targets])
        )
        estimate_energy = torch.sum(self.estimate ** 2)

        # Solve permutation problem based on similarity matrix
        similarity = torch.sum(
            self.similarity_matrix[
                np.arange(len(self.targets)), self.best_coloring
            ]
        )

        # Compute the final SDR as (|s|^2)/(|s|^2 + |shat|^2 + sum(matmul(s, shat)))
        sdr = target_energy / (target_energy + estimate_energy - 2 * similarity)
        return -10 * torch.log10(sdr)


def optimized_graph_pit_source_aggregated_sdr_loss(
        estimate, targets, segment_boundaries, permutation_solver
):
    return OptimizedGraphPITSourceAggregatedSDRLoss(
        estimate, targets, segment_boundaries,
        permutation_solver=permutation_solver
    ).loss


class OptimizedGraphPITSourceAggregatedSDRLossModule(LossModule):
    def __init__(self, permutation_solver):
        super().__init__()
        self.permutation_solver = permutation_solver

    def get_loss_object(self, estimate, targets, segment_boundaries):
        return OptimizedGraphPITSourceAggregatedSDRLoss(
            estimate, targets, segment_boundaries,
            permutation_solver=self.permutation_solver
        )

    def extra_repr(self) -> str:
        return f'{self.permutation_solver}, '
