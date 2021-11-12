from dataclasses import dataclass
from typing import List, Tuple, Callable
import torch
from cached_property import cached_property

from graph_pit.graph import Graph
from graph_pit.loss.base import GraphPITBase, LossModule


__all__ = [
    'GraphPITLoss',
    'GraphPITLossModule',
    'graph_pit_loss',
]


def solve_graph_pit(
        estimate: torch.Tensor,
        targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        graph: Graph,
        loss_fn: Callable
) -> Tuple[torch.Tensor, Tuple[int, ...], torch.Tensor]:
    colorings = list(graph.enumerate_graph_colorings(
        max_num_colors=estimate.shape[0]))
    if len(colorings) == 0:
        raise ValueError(f'No coloring found for graph! graph: {graph}')

    best_loss = None
    best_coloring = None
    best_target_sum = None

    for coloring in colorings:
        # Construct targets matching the current coloring
        target_sum = target_sum_from_target_list(
            estimate, targets, segment_boundaries, coloring
        )
        loss = loss_fn(estimate, target_sum)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_coloring = coloring
            best_target_sum = target_sum

    return best_loss, best_coloring, best_target_sum


@dataclass
class GraphPITLoss(GraphPITBase):
    """
    A class variant of the Graph-PIT loss. This class computes the loss for a
    single instance of estimates and targets. It gives access to the
    intermediate states (e.g., `best_coloring`) that would not be accessible
    with a plain function. This simplifies testing and is easy to extend
    because individual steps of the loss computation are factored out into
    different methods that can easily be overwritten.
    """
    loss_fn: Callable

    def __post_init__(self):
        # Check inputs
        num_estimates = self.estimate.shape[0]
        num_targets = len(self.targets)
        if num_estimates > 30:
            raise ValueError(f'Are you sure? num_estimates={num_estimates}')

        if num_targets != len(self.segment_boundaries):
            raise ValueError(
                f'The number of targets doesn\'t match the number of segment '
                f'boundaries! '
                f'num targets: {num_targets}, '
                f'num segment_boundaries: {len(self.segment_boundaries)}'
            )

    @property
    def loss(self) -> torch.Tensor:
        return self._loss[0]

    @property
    def best_coloring(self) -> Tuple[int]:
        return self._loss[1]

    @property
    def best_target_sum(self) -> torch.Tensor:
        return self._loss[2]

    @cached_property
    def _loss(self):
        return solve_graph_pit(
            self.estimate, self.targets, self.segment_boundaries, self.graph,
            self.loss_fn
        )


def graph_pit_loss(
        estimate: torch.Tensor,
        targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        loss_fn: Callable,
) -> torch.Tensor:
    """
    Graph-PIT loss function.

    A function wrapper around the GraphPITLoss class for a simpler interface to
    the loss.

    Args:
        estimate (n_out time [...]): Estimations, one for each output channel of the
            separator.
        targets (n_target time [...]): List of targets, one for each utterance. The
            targets are not zero-padded. The length of the target must match
            `end - start` of the corresponding segment boundary.
        segment_boundaries (n_target): Start and stop times of targets relative
            to estimate.
        loss_fn: Callable loss function to apply to the sum signal. Has to have
            the signature `loss_fn(estimate, target)`.

    Returns:
        loss
    """
    return GraphPITLoss(estimate, targets, segment_boundaries, loss_fn).loss


class GraphPITLossModule(LossModule):
    """
    A `torch.nn.Module`-based interface to the loss object. This is useful if
    `loss_fn` itself is a module and the loss should be displayed in the print
    representation of the module. Also, this is required for an object-oriented
    approach for the model, e.g., with `pt.Configurable`.
    """
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def get_loss_object(
            self,
            estimate: torch.Tensor,
            targets: List[torch.Tensor],
            segment_boundaries: List[Tuple[int, int]],
            **kwargs,   # unused here
    ) -> GraphPITLoss:
        return GraphPITLoss(
            estimate, targets, segment_boundaries,
            loss_fn=self.loss_fn,
        )


def target_sum_from_target_list(
        estimate: torch.Tensor,
        targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        permutation: List[int],
) -> torch.Tensor:
    """
    Constructs target sum signals from utterances (`target`) and the assignment
    of utterances to output channels (`permutation`)

    Args:
        estimate (n_out time [...]): Only used to ensure the correct shape and type.
        targets (n_targets time [...]): The target signals, one for each utterance.
        segment_boundaries: Segment boundaries
        permutation:

    Returns:

    """
    if len(segment_boundaries) != len(targets):
        raise ValueError(
            f'Length of segment_boundaries does not match length of targets! '
            f'num segment_boundaries: {len(segment_boundaries)} '
            f'num targets: {len(targets)}'
        )

    target_sum = torch.zeros_like(estimate)
    for idx, ((start, stop), (target_index, estimate_index)) in enumerate(zip(
            segment_boundaries, enumerate(permutation)
    )):
        if targets[target_index].shape[0] != stop - start:
            raise ValueError(
                f'Length mismatch between target and segment_boundaries at '
                f'target {idx}: '
                f'target shape: {targets[target_index].shape} '
                f'segment_boundaries: {start, stop}'
            )
        if start < 0 or stop > target_sum.shape[1]:
            raise ValueError(
                f'Length mismatch between estimation and targets / '
                f'segment_boundaries at {idx}: '
                f'estimation shape: {estimate.shape} '
                f'segment_boundaries: {start, stop}'
            )
        target_sum[estimate_index, start:stop] += targets[target_index]
    return target_sum
