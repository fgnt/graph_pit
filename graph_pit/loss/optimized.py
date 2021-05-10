from dataclasses import dataclass, field
from functools import cached_property
from typing import Union, Callable
import torch
import numpy as np

import padertorch as pt

from graph_pit.loss import GraphPITBase
from graph_pit.permutation_solving import graph_permutation_solvers


@dataclass
class OptimizedGraphPITSDR3Loss(GraphPITBase):
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
                graph_permutation_solvers[self.permutation_solver]
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
        return torch.stack(v)

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
