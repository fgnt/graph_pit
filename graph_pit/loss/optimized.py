from dataclasses import dataclass
from cached_property import cached_property
from typing import Union, Callable, List, Tuple
import torch
import numpy as np

import graph_pit.assignment
from graph_pit.loss.base import GraphPITBase, LossModule

__all__ = [
    'OptimizedGraphPITSourceAggregatedSDRLoss',
    'OptimizedGraphPITSourceAggregatedSDRLossModule',
    'optimized_graph_pit_source_aggregated_sdr_loss',
]

from graph_pit.utils import to_numpy
import graph_pit


def compute_mse_similarity_matrix(estimate, targets, segment_boundaries):
    """
    # TODO: can this be optimized?

    This computes the matrix-matrix product but ignores anything that lies
    outside the segment boundaries (and is by definition 0).
    """
    v = []
    for target, (start, stop) in zip(targets, segment_boundaries):
        v.append(torch.sum(
            estimate[:, start:stop, ...] * target[None, :, ...],
            dim=tuple(range(1, len(estimate.shape)))
        ))
    similarity_matrix = torch.stack(v)
    assert len(similarity_matrix.shape) == 2, similarity_matrix.shape
    return similarity_matrix


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
    assignment_solver: Union[Callable, str] = 'optimal_dynamic_programming'
    reduction: str = 'sum'

    def __post_init__(self):
        super().__post_init__()
        assert self.reduction in ('sum', 'mean'), self.reduction

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
        x = graph_pit.assignment.solve_assignment(
            to_numpy(similarity_matrix, detach=True), self.graph,
            minimize=False, algorithm=self.assignment_solver
        )
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
        loss = self.compute_f(self.max_similarity)
        if self.reduction == 'mean':
            loss = loss / np.prod(self.estimate.shape)
        return loss


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
    sdr_max: float = None

    def __post_init__(self):
        super().__post_init__()
        assert self.reduction == 'sum', (
            f'Only sum reduction makes sense for SA-SDR, not {self.reduction}'
        )

    @cached_property
    def _threshold(self):
        if self.sdr_max is None:
            return None
        return 10 ** (-self.sdr_max / 10)

    @cached_property
    def similarity_matrix(self) -> torch.Tensor:
        """
        Returns:
            Shape: (num_targets, num_estimates)
        """
        return compute_mse_similarity_matrix(
            self.estimate, self.targets, self.segment_boundaries
        )

    def compute_f(self, x) -> torch.Tensor:
        # Compute target and estimate energies
        target_energy = torch.sum(
            torch.stack([torch.sum(t ** 2) for t in self.targets])
        )
        estimate_energy = torch.sum(self.estimate ** 2)

        # Compute the final SDR
        sdr = (estimate_energy - 2 * x) / target_energy + 1
        if self.sdr_max is not None:
            sdr = sdr + self._threshold
        return 10 * torch.log10(sdr)


def optimized_graph_pit_source_aggregated_sdr_loss(
        estimate: torch.Tensor, targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        graph_segment_boundaries: List[Tuple[int, int]] = None,
        assignment_solver: Union[Callable, str] =
        OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver
) -> torch.Tensor:
    """
    Function form of the sa-SDR Graph-PIT loss.
    """
    return OptimizedGraphPITSourceAggregatedSDRLoss(
        estimate, targets, segment_boundaries,
        graph_segment_boundaries=graph_segment_boundaries,
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
            OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver,
            reduction: str = 'sum',
    ):
        """
        The sa-SDR Graph-PIT loss as a `Torch.nn.Module`. Can be used in a
        torch module.
        """
        super().__init__()
        self.assignment_solver = assignment_solver
        self.reduction = reduction

    def get_loss_object(
            self, estimate: torch.Tensor, targets: torch.Tensor,
            segment_boundaries: List[Tuple[int, int]],
            **kwargs,
    ) -> loss_class:
        return self.loss_class(
            estimate, targets, segment_boundaries,
            assignment_solver=self.assignment_solver,
            reduction=self.reduction
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

    def __init__(
            self,
            assignment_solver: Union[Callable, str] = OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver,
            sdr_max: float = None,
    ):
        super().__init__(assignment_solver, reduction='sum')
        self.sdr_max = sdr_max

    def get_loss_object(
            self, estimate: torch.Tensor, targets: torch.Tensor,
            segment_boundaries: List[Tuple[int, int]],
            graph_segment_boundaries: List[Tuple[int, int]] = None,
            **kwargs,
    ) -> loss_class:
        return self.loss_class(
            estimate, targets, segment_boundaries,
            graph_segment_boundaries=graph_segment_boundaries,
            assignment_solver=self.assignment_solver,
            sdr_max=self.sdr_max,
        )


@dataclass
class OptimizedGraphPITMSELoss(OptimizedGraphPITLoss):
    reduction: str = 'mean'

    @cached_property
    def similarity_matrix(self):
        return compute_mse_similarity_matrix(
            self.estimate, self.targets, self.segment_boundaries
        )

    def compute_f(self, x):
        target_energy = torch.sum(
            torch.stack([torch.sum(t ** 2) for t in self.targets])
        )
        estimate_energy = torch.sum(self.estimate ** 2)
        return target_energy + estimate_energy - 2 * x


class OptimizedGraphPITMSELossModule(OptimizedGraphPITLossModule):
    loss_class = OptimizedGraphPITMSELoss

    def __init__(
            self,
            assignment_solver: Union[Callable, str] = OptimizedGraphPITMSELoss.assignment_solver,
            reduction: str = 'mean'
    ):
        super().__init__(assignment_solver, reduction)


def optimized_graph_pit_mse_loss(
        estimate: torch.Tensor, targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        graph_segment_boundaries: List[Tuple[int, int]] = None,
        assignment_solver: Union[Callable, str] =
        OptimizedGraphPITSourceAggregatedSDRLoss.assignment_solver,
) -> torch.Tensor:
    """
    Function form of the sa-SDR Graph-PIT loss.
    """
    return OptimizedGraphPITMSELoss(
        estimate, targets, segment_boundaries,
        graph_segment_boundaries=graph_segment_boundaries,
        assignment_solver=assignment_solver
    ).loss


@dataclass
class OptimizedGraphPITBCEWithLogitsLoss(OptimizedGraphPITLoss):
    """
    Computes the binary cross entropy with logits as
    `torch.nn.functional.binary_cross_entropy_with_logits`
    """
    reduction: str = 'mean'

    @cached_property
    def similarity_matrix(self) -> torch.Tensor:
        return compute_mse_similarity_matrix(
            self.estimate, self.targets, self.segment_boundaries
        )

    def compute_f(self, x) -> torch.Tensor:
        """
        Computes the loss value from the similarity score `x` using
        the log-sum-exp trick for numerical stabilization.
        """
        m = torch.clamp_min(-self.estimate, 0)
        return -x + torch.sum(
            m + + self.estimate +
            torch.log(torch.exp(-m) + torch.exp(-m - self.estimate))
        )


class OptimizedGraphPITBCEWithLogits(OptimizedGraphPITLossModule):
    loss_class = OptimizedGraphPITBCEWithLogitsLoss

    def __init__(
            self,
            assignment_solver: Union[Callable, str] = OptimizedGraphPITBCEWithLogitsLoss.assignment_solver,
            reduction: str = 'mean'
    ):
        super().__init__(assignment_solver, reduction)


def optimized_graph_pit_bce_with_logits_loss(
        estimate: torch.Tensor, targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        graph_segment_boundaries: List[Tuple[int, int]] = None,
        assignment_solver: Union[Callable, str] =
        OptimizedGraphPITBCEWithLogitsLoss.assignment_solver
):
    return OptimizedGraphPITBCEWithLogitsLoss(
        estimate, targets, segment_boundaries,
        graph_segment_boundaries=graph_segment_boundaries,
        assignment_solver=assignment_solver
    ).loss
