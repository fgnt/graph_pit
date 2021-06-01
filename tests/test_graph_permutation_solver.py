from collections import deque
from typing import Tuple

import pytest

from graph_pit.graph import Graph
import numpy as np
import paderbox as pb
from graph_pit import assignment as gpa


def _random_graph(num_vertices, window_size):
    window = deque(maxlen=window_size)
    edge_list = set()
    for v in range(num_vertices):
        if window:
            discard = np.random.randint(0, len(window) + 1)
            for i in range(discard):
                window.popleft()

            for neighbor in window:
                edge_list.add((neighbor, v))
        window.append(v)
    graph = Graph.from_edge_list(num_vertices, edge_list)
    return graph


def _check_valid_solution(coloring: Tuple, graph: Graph):
    num_colors = max(coloring) + 1
    valid_colorings = list(map(tuple, graph.enumerate_graph_colorings(num_colors)))
    assert tuple(coloring) in valid_colorings


@pytest.mark.parametrize(
    'assignment_solver,num_targets,num_estimates,minimize',
    [
        (assignment_solver, num_targets, num_estimates, minimize)
        for assignment_solver in gpa.graph_assignment_solvers.keys()
        for num_targets, num_estimates in [(3, 2), (10, 2), (5, 3), (5, 5), (3, 15)]
        for minimize in [True, False]
    ]
)
def test_permutation_solver_valid(
        assignment_solver,
        num_targets, num_estimates,
        minimize, trials=5,
):
    """Test if all permutation solvers return a valid coloring of the graph"""
    assignment_fn = gpa.graph_assignment_solvers[assignment_solver](
        minimize=minimize
    )
    for _ in range(trials):
        score_matrix = np.random.randn(num_targets, num_estimates)
        graph = _random_graph(num_targets, (num_estimates + 1) // 2)
        best_coloring = assignment_fn(score_matrix, graph)
        if assignment_solver == 'greedy_cop':
            # This greedy variant does not always find a solution, so it should
            # pass the test if the output is None
            if best_coloring is None:
                continue
        _check_valid_solution(best_coloring, graph)


@pytest.mark.parametrize(
    'assignment_solver,num_targets,num_estimates,minimize',
    [
        (assignment_solver, num_targets, num_estimates, minimize)
        for assignment_solver in [
            'optimal_brute_force', 'optimal_branch_and_bound',
            'optimal_dynamic_programming'
        ]
        for num_targets, num_estimates in [
        (3, 2), (10, 2),
        (5, 3),
        (5, 5)
    ]
        for minimize in [True, False]
    ]
)
def test_permutation_optimal(
        assignment_solver, num_targets, num_estimates, minimize, trials=10
):
    """
    Test if the different optimal permutation solvers give the same result.
    These are at the moment:
     - Unoptimized brute force
     - brute force (the optimized version)
     - branch_and_bound
    """
    optimal_permutation_solver = gpa.graph_assignment_solvers['optimal_brute_force'](
        minimize=minimize, optimize_connected_components=False
    )
    permutation_solver = gpa.graph_assignment_solvers[assignment_solver](minimize=minimize)
    for _ in range(trials):
        score_matrix = np.random.randn(num_targets, num_estimates)
        graph = _random_graph(num_targets, (num_estimates + 1) // 2)
        best_coloring = permutation_solver(score_matrix, graph)

        _check_valid_solution(best_coloring, graph)
        optimal_coloring = optimal_permutation_solver(score_matrix, graph)
        assert tuple(best_coloring) == tuple(optimal_coloring), (
            list(score_matrix), graph
        )


# Sometimes this test fails when the randomly generated graph has too small
# connected components
def test_runtime(num_targets=15, num_estimates=3):
    """Test that the branch-and-bound, greedy and dfs are faster than
    brute-force"""
    timer = pb.utils.timer.TimerDict()

    score_matrix = np.random.randn(num_targets, num_estimates)
    graph = _random_graph(num_targets, (num_estimates + 1) // 2)

    # Unoptimized
    assignment_solver = gpa.OptimalBruteForceGraphAssignmentSolver(
        optimize_connected_components=False
    )
    with timer['unoptimized']:
        assignment_solver(score_matrix, graph)

    # Optimized brute force
    assignment_solver = gpa.OptimalBruteForceGraphAssignmentSolver()
    with timer['brute_force']:
        assignment_solver(score_matrix, graph)

    # DFS
    assignment_solver = gpa.DFSGraphAssignmentSolver()
    with timer['dfs']:
        assignment_solver(score_matrix, graph)

    # Greedy
    assignment_solver = gpa.GreedyCOPGraphAssignmentSolver()
    with timer['greedy']:
        assignment_solver(score_matrix, graph)

    # Branch and Bound
    assignment_solver = gpa.OptimalBranchAndBoundGraphAssignmentSolver()
    with timer['branch_and_bound']:
        assignment_solver(score_matrix, graph)

    # Dynamic Programming
    assignment_solver = gpa.OptimalDynamicProgrammingAssignmentSolver()
    with timer['dynamic_programming']:
        assignment_solver(score_matrix, graph)

    times = timer.as_dict
    assert times['unoptimized'] > times['brute_force']
    assert times['dfs'] < times['brute_force']
    assert times['greedy'] < times['brute_force']
    assert times['dynamic_programming'] < times['brute_force']
    # These can fail occasionally because B&B's runtime is not deterministic
    assert times['branch_and_bound'] < times['brute_force']
    assert times['dynamic_programming'] < times['branch_and_bound']

