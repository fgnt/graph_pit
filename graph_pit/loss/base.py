from dataclasses import dataclass
from typing import List, Tuple

import torch
from cached_property import cached_property

from ..graph import Graph, get_overlap_graph


@dataclass
class GraphPITBase:
    estimate: torch.Tensor
    targets: List[torch.Tensor]
    segment_boundaries: List[Tuple[int, int]]

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

    @cached_property
    def graph(self) -> Graph:
        """The graph constructed from the segment boundaries"""
        return get_overlap_graph(self.segment_boundaries)

    @property
    def loss(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def best_coloring(self) -> Tuple[int]:
        raise NotImplementedError()


class LossModule(torch.nn.Module):
    def get_loss_object(
            self,
            estimate: torch.Tensor,
            targets: List[torch.Tensor],
            segment_boundaries: List[Tuple[int, int]],
            **kwargs,   # for additional arguments for modified losses
    ) -> GraphPITBase:
        raise NotImplementedError()

    def forward(
            self,
            estimate: torch.Tensor,
            targets: List[torch.Tensor],
            segment_boundaries: List[Tuple[int, int]],
            **kwargs,   # for additional arguments for modified losses
    ) -> torch.Tensor:
        return self.get_loss_object(estimate, targets, segment_boundaries,
                                    **kwargs).loss
