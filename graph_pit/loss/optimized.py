from dataclasses import dataclass
from functools import cached_property
from typing import Union, Callable, List, Tuple
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
    The sa-SDR with "optimized" assignment algorithms from [1]. The assignment
    solver can be selected by setting `permutation_solver` to one of
     - 'optimal_brute_force'
     - 'optimal_branch_and_bound'
     - 'optimal_dynamic_programming'
     - 'dfs'
     - 'greedy_cop'

    The loss aggregates the energies of the target and error
    signals in the fraction.

    TODO: Fix references when paper is published on IEEExplore

    References:
        [1] Speeding up Permutation Invariant Training
    """
    assignment_solver: Union[Callable, str] = 'optimal_brute_force'

    def __post_init__(self):
        super().__post_init__()

        if not callable(self.assignment_solver):
            self.assignment_solver = (
                graph_assignment_solvers[self.assignment_solver]
            )(minimize=False)

    @cached_property
    def similarity_matrix(self) -> torch.Tensor:
        """
        # TODO: can this be optimized?

        This computes the matrix-matrix product but ignores anything that lies
        outside the segment boundaries (and is by definition 0).

        TODO: Could a sparse tensor be faster?

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
    def best_coloring(self) -> tuple:
        similarity_matrix = self.similarity_matrix
        x = self.assignment_solver(
            pt.utils.to_numpy(similarity_matrix, detach=True), self.graph
        )
        assert x is not None
        return tuple(x)

    @cached_property
    def loss(self) -> torch.Tensor:
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
        estimate: torch.Tensor, targets: torch.Tensor,
        segment_boundaries: List[Tuple[int, int]],
        assignment_solver: Union[Callable, str] = OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver
) -> torch.Tensor:
    """
    Function form of the sa-SDR Graph-PIT loss.
    """
    return OptimizedGraphPITSourceAggregatedSDRLoss(
        estimate, targets, segment_boundaries,
        assignment_solver=assignment_solver
    ).loss


class OptimizedGraphPITSourceAggregatedSDRLossModule(LossModule):
    def __init__(
            self,
            assignment_solver: Union[Callable, str] = OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver
    ):
        """
        The sa-SDR Graph-PIT loss as a `Torch.nn.Module`. Can be used in a torch
        module.
        """
        super().__init__()
        self.assignment_solver = assignment_solver

    def get_loss_object(
            self, estimate: torch.Tensor, targets: torch.Tensor,
            segment_boundaries: List[Tuple[int, int]],
    ) -> OptimizedGraphPITSourceAggregatedSDRLoss:
        return OptimizedGraphPITSourceAggregatedSDRLoss(
            estimate, targets, segment_boundaries,
            assignment_solver=self.assignment_solver
        )

    def extra_repr(self) -> str:
        return f'{self.assignment_solver}, '
