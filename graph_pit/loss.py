from typing import List, Tuple, Callable
import torch

from graph_pit.graph import EdgeListGraph
from graph_pit.utils import get_overlaps_from_segment_boundaries


def graph_pit_loss(
        estimate: torch.Tensor,
        targets: List[torch.Tensor],
        segment_boundaries: List[Tuple[int, int]],
        loss_fn: Callable,
) -> torch.Tensor:
    """
    Graph-PIT loss function.

    Args:
        estimate (n_out time): Estimations, one for each output channel of the
            separator.
        targets (n_target time): List of targets, one for each utterance. The
            targets are not zero-padded. The length of the target must match
            `end - start` of the corresponding segment boundary.
        segment_boundaries (n_target): Start and stop times of targets relative
            to estimate.
        loss_fn:

    Returns:
        loss
    """
    # Check inputs
    num_estimates = estimate.shape[0]
    num_targets = len(targets)
    if num_estimates > 30:
        raise ValueError(f'Are you sure? num_estimates={num_estimates}')

    if num_targets != len(segment_boundaries):
        raise ValueError(
            f'The number of targets doesn\'t match the number of segment '
            f'boundaries! '
            f'num targets: {num_targets}, '
            f'num segment_boundaries: {len(segment_boundaries)}'
        )

    # Construct graph from overlaps, based on segment_boundaries
    graph = get_overlap_graph(segment_boundaries)

    # Enumerate all colorings
    colorings = list(graph.enumerate_graph_colorings(
        max_num_colors=num_estimates))
    if len(colorings) == 0:
        raise ValueError(f'No coloring found for graph! graph: {graph}')

    candidates = []
    for coloring in colorings:
        # Construct targets matching the current coloring
        candidates.append(loss_fn(
            estimate,
            target_sum_from_target_list(
                estimate, targets, segment_boundaries, coloring
            )
        ))

    min_loss, idx = torch.min(torch.stack(candidates), dim=0)

    return min_loss


def get_overlap_graph(segment_boundaries: List[Tuple[int, int]]):
    edges = get_overlaps_from_segment_boundaries(segment_boundaries)
    graph = EdgeListGraph(len(segment_boundaries), edges)
    assert graph.num_vertices == len(segment_boundaries), (
        graph, segment_boundaries
    )
    return graph


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
        estimate (n_out time): Only used to ensure the correct shape and type.
        targets: The target signals, one for each utterance.
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
        target_sum[estimate_index, start:stop] += targets[target_index]
    return target_sum
