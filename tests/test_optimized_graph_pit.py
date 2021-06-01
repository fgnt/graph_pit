from graph_pit.loss.optimized import OptimizedGraphPITSourceAggregatedSDRLoss
from graph_pit.loss.regression import sdr_loss
from graph_pit.loss.unoptimized import GraphPITLoss
import torch
import pytest
import paderbox as pb


@pytest.mark.parametrize(
    'seed,num_samples,num_estimates,segment_boundaries,algorithm',
    [(seed, num_samples, num_estimates, segment_boundaries, algorithm)
     for seed, num_samples, num_estimates, segment_boundaries in [
        (0, 500, 2, [(0, 100), (150, 350), (300, 450)]),
        (0, 8000 * 16, 2, [(0, 8000), (4000, 12000), (8000, 16000)]),
        (0, 8000 * 16, 3, [(0, 8000), (4000, 12000), (8000, 16000)]),
        (0, 8000 * 16, 4, [(0, 8000), (4000, 12000), (4000, 16000)]),
    ] for algorithm in ['optimal_brute_force', 'optimal_branch_and_bound']],
)
def test_optimized_graph_pit(
        seed, num_samples, num_estimates, segment_boundaries, algorithm
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
        estimate, targets, segment_boundaries, loss_fn=sdr_loss(
            aggregation='in_fraction', reduction='sum'
        )
    )

    optimized_graph_pit = OptimizedGraphPITSourceAggregatedSDRLoss(
        estimate, targets, segment_boundaries, permutation_solver=algorithm
    )

    # Check actual loss value
    pb.testing.assert_allclose(
        unoptimized_graph_pit.loss, optimized_graph_pit.loss,
    )

    # Check permutation
    pb.testing.assert_equal(
        unoptimized_graph_pit.best_coloring, optimized_graph_pit.best_coloring
    )
