from dataclasses import dataclass
from typing import List, Tuple

import torch
from cached_property import cached_property

from ..graph import Graph, get_overlap_graph
from ..utils import validate_inputs


@dataclass
class GraphPITBase:
    estimate: torch.Tensor
    targets: List[torch.Tensor]
    segment_boundaries: List[Tuple[int, int]]   # Used to efficiently compute the loss
    graph_segment_boundaries: List[Tuple[int, int]] = None  # Used for constructing the overlap graph

    def __post_init__(self):
        # Check inputs
        validate_inputs(self.estimate, self.targets, self.segment_boundaries)

        if self.graph_segment_boundaries is None:
            self.graph_segment_boundaries = self.segment_boundaries

        if len(self.segment_boundaries) != len(self.graph_segment_boundaries):
            raise ValueError(
                f'The number of graph segment boundaries does not match the '
                f'number of targets! '
                f'num targets: {len(self.targets)} '
                f'num graph_segment_boundaries: {len(self.graph_segment_boundaries)}'
            )

    @cached_property
    def graph(self) -> Graph:
        """The graph constructed from the segment boundaries"""
        return get_overlap_graph(self.graph_segment_boundaries)

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
            graph_segment_boundaries: List[Tuple[int, int]] = None,
            **kwargs,   # for additional arguments for modified losses
    ) -> GraphPITBase:
        raise NotImplementedError()

    def forward(
            self,
            estimate: torch.Tensor,
            targets: List[torch.Tensor],
            segment_boundaries: List[Tuple[int, int]],
            graph_segment_boundaries: List[Tuple[int, int]] = None,
            **kwargs,   # for additional arguments for modified losses
    ) -> torch.Tensor:
        return self.get_loss_object(estimate, targets, segment_boundaries,
                                    graph_segment_boundaries,
                                    **kwargs).loss
