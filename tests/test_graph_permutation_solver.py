from collections import deque
from typing import Tuple

import pytest

from graph_pit.graph import Graph
import numpy as np
import paderbox as pb
from graph_pit import permutation_solving as gps


def _random_graph(num_vertices, window_size):
    window = deque(maxlen=window_size)
    edge_list = set()
    for v in range(num_vertices):
        if window:
            num_neighbors = np.random.randint(0, len(window) + 1)
            if num_neighbors:
                neighbors = np.random.choice(window, num_neighbors)

                for neighbor in neighbors:
                    edge_list.add((neighbor, v))
        window.append(v)
    graph = Graph.from_edge_list(num_vertices, edge_list)
    return graph


def _check_valid_solution(coloring: Tuple, graph: Graph):
    num_colors = max(coloring) + 1
    valid_colorings = list(map(tuple, graph.enumerate_graph_colorings(num_colors)))
    assert tuple(coloring) in valid_colorings


@pytest.mark.parametrize(
    'permutation_solver,num_targets,num_estimates,minimize',
    [
        (permutation_solver, num_targets, num_estimates, minimize)
        for permutation_solver in gps.graph_permutation_solvers.keys()
        for num_targets, num_estimates in [(3, 2), (10, 2), (5, 3), (5, 5), (3, 15)]
        for minimize in [True, False]
    ]
)
def test_permutation_solver_valid(
        permutation_solver,
        num_targets, num_estimates,
        minimize, trials=5,
):
    """Test if all permutation solvers return a valid coloring of the graph"""
    permutation_solver_fn = gps.graph_permutation_solvers[permutation_solver](
        minimize=minimize
    )
    for _ in range(trials):
        score_matrix = np.random.randn(num_targets, num_estimates)
        graph = _random_graph(num_targets, (num_estimates + 1) // 2)
        best_coloring = permutation_solver_fn(score_matrix, graph)
        if permutation_solver == 'greedy_cop':
            # This greedy variant does not always find a solution, so it should
            # pass the test if the output is None
            if best_coloring is None:
                continue
        _check_valid_solution(best_coloring, graph)


@pytest.mark.parametrize(
    'permutation_solver,num_targets,num_estimates,minimize',
    [
        (permutation_solver, num_targets, num_estimates, minimize)
        for permutation_solver in ['optimal_brute_force', 'optimal_branch_and_bound']
        for num_targets, num_estimates in [(3, 2), (10, 2), (5, 3), (5, 5)]
        for minimize in [True, False]
    ]
)
def test_permutation_optimal(
        permutation_solver, num_targets, num_estimates, minimize, trials=10
):
    """
    Test if the different optimal permutation solvers give the same result.
    These are at the moment:
     - Unoptimized brute force
     - brute force (the optimized version)
     - branch_and_bound
    """
    optimal_permutation_solver = gps.graph_permutation_solvers['optimal_brute_force'](
        minimize=minimize, optimize_connected_components=False
    )
    permutation_solver = gps.graph_permutation_solvers[permutation_solver](minimize=minimize)
    for _ in range(trials):
        score_matrix = np.random.randn(num_targets, num_estimates)
        graph = _random_graph(num_targets, (num_estimates + 1) // 2)
        best_coloring = permutation_solver(score_matrix, graph)

        _check_valid_solution(best_coloring, graph)
        optimal_coloring = optimal_permutation_solver(score_matrix, graph)
        assert tuple(best_coloring) == tuple(optimal_coloring)


# Sometimes this test fails when the randomly generated graph has too small
# connected components
def test_runtime(num_targets=15, num_estimates=3):
    """Test that the branch-and-bound, greedy and dfs are faster than
    brute-force"""
    timer = pb.utils.timer.TimerDict()

    score_matrix = np.random.randn(num_targets, num_estimates)
    graph = _random_graph(num_targets, (num_estimates + 1) // 2)

    # Unoptimized
    permutation_solver = gps.OptimalBruteForceGraphPermutationSolver(
        optimize_connected_components=False
    )
    with timer['unoptimized']:
        permutation_solver(score_matrix, graph)

    # Optimized brute force
    permutation_solver = gps.OptimalBruteForceGraphPermutationSolver()
    with timer['optimized_brute_force']:
        permutation_solver(score_matrix, graph)

    # DFS
    permutation_solver = gps.DFSGraphPermutationSolver()
    with timer['dfs']:
        permutation_solver(score_matrix, graph)

    # Greedy
    permutation_solver = gps.GreedyCOPGraphPermutationSolver()
    with timer['greedy']:
        permutation_solver(score_matrix, graph)

    # Branch and Bound
    permutation_solver = gps.OptimalBranchAndBoundGraphPermutationSolver()
    with timer['branch_and_bound']:
        permutation_solver(score_matrix, graph)

    times = timer.as_dict
    assert times['unoptimized'] > times['optimized_brute_force']
    assert times['dfs'] < times['optimized_brute_force']
    assert times['greedy'] < times['optimized_brute_force']
    assert times['branch_and_bound'] < times['optimized_brute_force']

