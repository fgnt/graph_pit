from dataclasses import dataclass
from cached_property import cached_property
from typing import Union, Callable, List, Tuple
import torch
import numpy as np

from graph_pit.loss.base import GraphPITBase, LossModule
from graph_pit.assignment import graph_assignment_solvers

__all__ = [
    'OptimizedGraphPITSourceAggregatedSDRLoss',
    'OptimizedGraphPITSourceAggregatedSDRLossModule',
    'optimized_graph_pit_source_aggregated_sdr_loss',
]

from graph_pit.utils import to_numpy


@dataclass
class OptimizedGraphPITLoss(GraphPITBase):
    """
    Base class for optimized variants of Graph-PIT losses. Can be used to
    define custom losses. Overwrite
     - `similarity_matriy` to compute the matrix "M" from [1]
     - `compute_f` to compute the function "f" from [1]

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
        """ The matrix "M" from the paper. """
        raise NotImplementedError()

    def compute_f(self, x) -> torch.Tensor:
        """The function "f" from the paper."""
        raise NotImplementedError()

    @cached_property
    def best_coloring(self) -> tuple:
        similarity_matrix = self.similarity_matrix
        x = self.assignment_solver(
            to_numpy(similarity_matrix, detach=True), self.graph
        )
        assert x is not None
        return tuple(x)

    @cached_property
    def max_similarity(self):
        return torch.sum(
            self.similarity_matrix[
                np.arange(len(self.targets)), self.best_coloring
            ]
        )

    @cached_property
    def loss(self) -> torch.Tensor:
        return self.compute_f(self.max_similarity)


@dataclass
class OptimizedGraphPITSourceAggregatedSDRLoss(OptimizedGraphPITLoss):
    """
    The sa-SDR with "optimized" assignment algorithms from [1]. The assignment
    solver can be selected by setting `assignment_solver` to one of
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

    def compute_f(self, x) -> torch.Tensor:
        # Compute target and estimate energies
        target_energy = torch.sum(
            torch.stack([torch.sum(t ** 2) for t in self.targets])
        )
        estimate_energy = torch.sum(self.estimate ** 2)

        # Compute the final SDR as
        # (|s|^2)/(|s|^2 + |shat|^2 + sum(matmul(s, shat)))
        sdr = target_energy / (target_energy + estimate_energy - 2 * x)
        return -10 * torch.log10(sdr)


def optimized_graph_pit_source_aggregated_sdr_loss(
        estimate: torch.Tensor, targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        assignment_solver: Union[Callable, str] =
        OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver
) -> torch.Tensor:
    """
    Function form of the sa-SDR Graph-PIT loss.
    """
    return OptimizedGraphPITSourceAggregatedSDRLoss(
        estimate, targets, segment_boundaries,
        assignment_solver=assignment_solver
    ).loss


class OptimizedGraphPITLossModule(LossModule):
    """
    Base class for loss modules for Graph-PIT losses. Sub-class can set
    `loss_class` to get a module for that loss.
    """
    loss_class = None

    def __init__(
            self,
            assignment_solver: Union[Callable, str] =
            OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver
    ):
        """
        The sa-SDR Graph-PIT loss as a `Torch.nn.Module`. Can be used in a
        torch module.
        """
        super().__init__()
        self.assignment_solver = assignment_solver

    def get_loss_object(
            self, estimate: torch.Tensor, targets: torch.Tensor,
            segment_boundaries: List[Tuple[int, int]],
            **kwargs,
    ) -> loss_class:
        return self.loss_class(
            estimate, targets, segment_boundaries,
            assignment_solver=self.assignment_solver
        )

    def extra_repr(self) -> str:
        return f'{self.assignment_solver}, '


class OptimizedGraphPITSourceAggregatedSDRLossModule(
    OptimizedGraphPITLossModule
):
    """
    The sa-SDR Graph-PIT loss as a `Torch.nn.Module`. Can be used in a
    torch module.
    """
    loss_class = OptimizedGraphPITSourceAggregatedSDRLoss
