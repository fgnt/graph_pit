import functools

from graph_pit.loss.optimized import OptimizedGraphPITSourceAggregatedSDRLoss, OptimizedGraphPITMSELoss
from graph_pit.loss.regression import sdr_loss
from graph_pit.loss.unoptimized import GraphPITLoss
from graph_pit.assignment import NoSolutionError
import torch
import pytest
import numpy as np


@pytest.mark.parametrize(
    'seed,num_samples,num_estimates,segment_boundaries,algorithm,unoptimized_loss_fn,optimized_loss_class',
    [(seed, num_samples, num_estimates, segment_boundaries, algorithm, unoptimized_loss_fn, optimized_loss_class)
     for seed, num_samples, num_estimates, segment_boundaries in [
         (0, 500, 2, [(0, 100), (150, 350), (300, 450)]),
         (0, 8000 * 16, 2, [(0, 8000), (4000, 12000), (8000, 16000)]),
         (0, 8000 * 16, 3, [(0, 8000), (4000, 12000), (8000, 16000)]),
         (0, 8000 * 16, 4, [(0, 8000), (4000, 12000), (4000, 16000)]),
     ] for algorithm in ['optimal_brute_force', 'optimal_branch_and_bound']
     for unoptimized_loss_fn, optimized_loss_class in [
         (functools.partial(sdr_loss, aggregation='in_fraction', reduction='sum'), OptimizedGraphPITSourceAggregatedSDRLoss),
         (torch.nn.functional.mse_loss, OptimizedGraphPITMSELoss),
     ]],
)
def test_optimized_graph_pit(
        seed, num_samples, num_estimates, segment_boundaries, algorithm,
        unoptimized_loss_fn, optimized_loss_class
):
    torch.manual_seed(seed)
    targets = [
        torch.rand(stop - start, dtype=torch.float64)
        for start, stop in segment_boundaries
    ]
    estimate = torch.rand(num_estimates, num_samples, dtype=torch.float64)

    # The optimized version uses in fraction loss aggregation, so we have to
    # test against that
    unoptimized_graph_pit = GraphPITLoss(
        estimate, targets, segment_boundaries, loss_fn=unoptimized_loss_fn
    )

    optimized_graph_pit = optimized_loss_class(
        estimate, targets, segment_boundaries, assignment_solver=algorithm
    )

    # Check actual loss value
    np.testing.assert_allclose(
        unoptimized_graph_pit.loss, optimized_graph_pit.loss,
    )

    # Check coloring
    np.testing.assert_equal(
        unoptimized_graph_pit.best_coloring, optimized_graph_pit.best_coloring
    )


@pytest.mark.parametrize(
    'optimized_loss_class',
    [
         OptimizedGraphPITSourceAggregatedSDRLoss,
         OptimizedGraphPITMSELoss
     ],
)
def test_optimized_graph_pit_exceptions(
        optimized_loss_class
):
    torch.manual_seed(0)

    with pytest.raises(
            NoSolutionError, match='No valid coloring could be found.'
    ):
        optimized_loss_class(
            estimate=torch.zeros(2, 100),
            targets=torch.zeros(3, 100),
            segment_boundaries=[(0, 100), (0, 100), (0, 100)]
        ).loss

    with pytest.raises(
            ValueError,
            match='The number of segment_boundaries does not match the number '
                  'of targets!'
    ):
        optimized_loss_class(torch.zeros(2, 100), torch.zeros(3, 100),
                       [(0, 100), (0, 100)]).loss

    with pytest.raises(
            ValueError,
            match='Length mismatch between target and segment_boundaries at '
                  'target'
    ):
        optimized_loss_class(torch.zeros(3, 100), torch.zeros(2, 100),
                       [(0, 50), (0, 100)]).loss

