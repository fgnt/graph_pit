"""
This file contains algorithms for permutation solving similar to
tvn.algorithm.permutation_solving but with cannot-link constraints described by
a cannot-link graph.

TODO: Rename! The name "permutation_solving" is actually wrong because this
    finds assignments instead of permutations. The name is taken from the uPIT
    case.
"""
from dataclasses import dataclass

import numpy as np

import paderbox as pb
from paderbox.utils.functional import partial_decorator
from .graph import Graph

import logging

logger = logging.getLogger('graph-permutation-solver')


def _find_mapping_apply_connected_components(
        permutation_solver, score_matrix, cannot_link_graph, **kwargs
):
    num_targets, num_estimates = score_matrix.shape
    assert num_targets == cannot_link_graph.num_vertices, (
        num_targets, cannot_link_graph.num_vertices
    )

    mapping = np.zeros(num_targets, dtype=np.int)

    for connected_component in cannot_link_graph.connected_components:
        cc_score_matrix = score_matrix[(connected_component.labels, slice(None))]
        partial_solution = permutation_solver(
            cc_score_matrix, connected_component, **kwargs
        )
        if partial_solution is None:
            return None
        mapping[connected_component.labels] = partial_solution

    return mapping


@dataclass
class GraphPermutationSolver:
    minimize: bool = False
    optimize_connected_components: bool = True

    def __call__(self, score_matrix, cannot_link_graph):
        score_matrix = np.asanyarray(score_matrix)
        assert np.issubdtype(score_matrix.dtype, np.floating), score_matrix.dtype
        num_targets, num_estimates = score_matrix.shape
        assert num_targets == cannot_link_graph.num_vertices, (
            num_targets, cannot_link_graph.num_vertices
        )

        if self.optimize_connected_components:
            coloring = _find_mapping_apply_connected_components(
                self.solve_permutation, score_matrix, cannot_link_graph
            )
        else:
            coloring = self.solve_permutation(score_matrix, cannot_link_graph)

        if coloring is None:
            logger.debug(
                f'Couldn\'t find a solution because there is no graph coloring '
                f'for the graph {cannot_link_graph} with {num_estimates} colors'
            )
        return coloring

    def solve_permutation(self, score_matrix, cannot_link_graph):
        raise NotImplementedError()


class OptimalBruteForceGraphPermutationSolver(GraphPermutationSolver):
    """
    Examples:
        >>> score_matrix = np.array([[10., 7., 5.], [11., 3., 6.]]).T
        >>> graph = Graph.from_edge_list(3, [(0, 1), (1, 2)])
        >>> OptimalBruteForceGraphPermutationSolver()(score_matrix, graph)
        array([1, 0, 1])

        >>> score_matrix = np.array([[10., 7., 8.], [11., 3., 5.]]).T
        >>> OptimalBruteForceGraphPermutationSolver()(score_matrix, graph)
        array([1, 0, 1])

        Fully connected graph does not have a solution, so the function returns None.
        >>> score_matrix = np.array([[10., 7., 5.], [11., 3., 6.]]).T
        >>> OptimalBruteForceGraphPermutationSolver()(score_matrix, Graph.from_edge_list(
        ...     3, [(0, 1), (1, 2), (0, 2)]))
    """
    def solve_permutation(self, score_matrix, cannot_link_graph):
        num_targets, num_estimates = score_matrix.shape
        colorings = list(
            cannot_link_graph.enumerate_graph_colorings(num_estimates)
        )

        if not colorings:
            return None

        # This is the old loop-based implementation. I'd like to keep it here in
        # case the indexing-based variant becomes too memory demanding
        #
        # best = RunningValue('min' if minimize else 'max')
        #
        # for permutation in colorings:
        #     assert len(permutation) == num_targets, permutation
        #     best(
        #         score_matrix[(range(num_targets), permutation)].sum(),
        #         permutation
        #     )
        # return best.data

        # The following lines create the same index as
        # np.arange(num_targets)[None].repeat(len(colorings), axis=0)
        # but without copying the elements of the arange
        x = np.arange(num_targets)
        first_index = np.lib.stride_tricks.as_strided(
            x, (len(colorings), x.size), (0, x.itemsize)
        )
        # Do some intelligent indexing for speed up over for loop. The advanced
        # indexing creates copies of the matrix, so it could potentially become
        # large.
        scores = np.sum(score_matrix[(first_index, colorings)], axis=-1)

        if self.minimize:
            idx = np.argmin(scores)
        else:
            idx = np.argmax(scores)

        return colorings[idx]


@dataclass
class GreedyCOPGraphPermutationSolver(GraphPermutationSolver):
    """
      Greedy algorithm.

      If used in the constrained k-means algorithm, it results in the
      COP-k-means [1].

      Has a runtime of O(N**2 log(N))
          - sort values of NxN matrix: O(N**2 log(N))
          - Go through the sorted values and select the best ones respecting the
              constraints: O(N**2)

      Note:
          This algorithm does not always find a solution. It returns `None` if
          no solution is found.

      Returns:
          Permutation with shape (N). Each entry in the returned array is
          an index along the K axis.

      Examples:
          >>> score_matrix = np.array([[10., 7., 5.], [11., 3., 6.]]).T
          >>> graph = Graph.from_edge_list(3, [(0, 1), (1, 2)])
          >>> solve_permutation_graph_greedy_cop(score_matrix, graph)
          array([1, 0, 1])

          The following example isn't solvable by this greedy approach
          >>> score_matrix = np.array([[10., 7., 8.], [11., 3., 5.]]).T
          >>> solve_permutation_graph_greedy_cop(score_matrix, graph)

      References:
          [1] Wagstaff, Kiri, Claire Cardie, Seth Rogers, and Stefan Schroedl.
              “Constrained K-Means Clustering with Background Knowledge,”
      """
    def solve_permutation(self, score_matrix, cannot_link_graph):
        if not self.minimize:
            score_matrix = -score_matrix

        N, K = score_matrix.shape
        coloring = -np.ones(N, dtype=np.int)

        # Copy the score matrix because we are going to modify it
        score_matrix: np.ndarray = score_matrix.copy()
        # score_matrix_flat is a view in score_matrix
        # -> changing score_matrix also changes score_matrix_flat
        score_matrix_flat = score_matrix.reshape(N * K)

        mask = np.zeros(score_matrix.shape, dtype=bool)
        mask_flat = mask.reshape(N * K)

        # argmax does not support axis=(-2, -1)
        # -> use score_matrix_flat
        for idx in np.argsort(score_matrix_flat):
            if mask_flat[idx]:
                # This means the whole matrix is filled with -inf.
                # There is no solution using this scheme.
                continue

            i, j = np.unravel_index(idx, score_matrix.shape)

            # Remove the usual candidates (same row)
            mask[i, :] = True

            # Remove candidates from the cannot-link graph
            # score_matrix[:, j] += adjacency_matrix[i]
            for k in cannot_link_graph.neighbors_of(i):
                mask[(k, j)] = True

            coloring[i] = j

        if np.any(coloring == -1):
            return None

        return coloring


@dataclass
class DFSGraphPermutationSolver(GraphPermutationSolver):
    """
    A depth-first search algorithm for greedy permutation solving.

    This greedy permutation solving algorithm always finds a solution
    if there is one (compared to the greedy COP variant), but in the
    worst case tries every possible combination. This algorithm is not
    guaranteed to find the optimal solution.

    Algorithm:
        1. Pick the best available score
        2. Eliminate all scores that are invalid given the selection from (1)
            and the cannot-link graph
        3. Go back to 1. If 1 does not succeed, backtrack.

    Note:
        In the worst case, this function performs the same operation as the
        optimal solver (i.e., tries every possible combination), but it often
        finds a good (but not necessarily optimal)
        solution faster than the optimal algorithm.
        Depending on the use-case and motivation, this "greedy" (it's not 100%
        greedy but I don't know how else to call it) solution might
        even be a better solution than the "optimal" one.

    Args:
        score_matrix (N K):
        cannot_link_graph: Graph with N vertices that describes cannot-link
            dependencies
        minimize: If `True`, minimizes the score

    Returns:
        Permutation with shape (N). Each entry in the returned array is
        an index along the K axis.

    Examples:
        >>> score_matrix = np.array([[10., 7., 5.], [11., 3., 6.]]).T
        >>> graph = Graph.from_edge_list(3, [(0, 1), (1, 2)])
        >>> solve_permutation_graph_dfs(score_matrix, graph)
        array([1, 0, 1])

        The following example isn't solvable by this COP greedy approach,
        but the DFS approach finds a solution
        >>> score_matrix = np.array([[10., 7., 8.], [11., 3., 5.]]).T
        >>> solve_permutation_graph_dfs(score_matrix, graph)
        array([1, 0, 1])

        >>> score_matrix = np.array([[10., 7., 8.], [11., 3., 5.]]).T
        >>> solve_permutation_graph_dfs(score_matrix, graph, minimize=True)
        array([0, 1, 0])

        Fully connected graph does not have a solution, so the function returns None.
        >>> solve_permutation_graph_dfs(score_matrix, Graph.from_edge_list(
        ...     3, [(0, 1), (1, 2), (0, 2)]))

        Also works for N < K
        >>> score_matrix = np.array([[10., 7., 8.], [11., 3., 5.]])
        >>> graph = Graph.from_edge_list(2, [(0, 1)])
        >>> solve_permutation_graph_dfs(score_matrix, graph)
        array([2, 0])
    """
    def solve_permutation(self, score_matrix, cannot_link_graph):
        N, K = score_matrix.shape

        if self.minimize:
            score_matrix = -score_matrix

        def find_permutation(score_matrix, depth):
            score_matrix_flat = score_matrix.reshape(N * K)
            for idx in np.argsort(score_matrix_flat)[::-1]:
                if score_matrix_flat[idx] == float('-inf'):
                    # This means the whole matrix is filled with -inf.
                    # There is no solution on this path.
                    return None

                i, j = np.unravel_index(idx, score_matrix.shape)

                _score_matrix = score_matrix.copy()
                # Remove the usual candidates (same row)
                _score_matrix[(i, slice(None))] = float('-inf')

                # Remove candidates from the cannot-link graph
                for k in cannot_link_graph.neighbors_of(i):
                    _score_matrix[(k, j)] = float('-inf')

                permutation = find_permutation(_score_matrix, depth + 1)
                if permutation is None:
                    if depth < N:
                        # No solution for this path
                        continue
                    permutation = -np.ones(N, dtype=np.int)
                assert depth <= N
                permutation[i] = j
                return permutation

        return find_permutation(score_matrix, 1)


class OptimalBranchAndBoundGraphPermutationSolver(GraphPermutationSolver):
    """
    A bran-and-bound algorithm to find the optimal solution to the graph
    permutation problem.

    Runtime:
        Has the same worst-case complexity as the brute-force variant, but is in
        many cases a lot faster. The better the scores are, the faster the
        algorithm becomes.
    """
    def solve_permutation(self, score_matrix, cannot_link_graph):
        N, K = score_matrix.shape

        if not self.minimize:
            score_matrix = -score_matrix

        # Make sure there are no negative values in the matrix
        # We can subtract the minimum from each column (or is this a row?). This
        # could make things easier to compute
        score_matrix = score_matrix - np.min(score_matrix, axis=-1, keepdims=True)
        assert np.all(score_matrix >= 0), score_matrix

        best_cost = None

        def find_best_permutation(cost_matrix, cost, depth):
            nonlocal best_cost
            current_vertex_costs = cost_matrix[0]

            best_perm = None
            for idx in np.argsort(current_vertex_costs):
                current_cost = current_vertex_costs[idx] + cost

                if best_cost is not None and current_cost >= best_cost:
                    return best_perm

                if depth == N - 1:
                    # We are at a leaf (or, one before leaf node)
                    best_cost = current_cost
                    return (idx,)

                # We are not at a leaf
                _cost_matrix = cost_matrix[1:].copy()
                for k in cannot_link_graph.neighbors_of(depth):
                    k = k - depth - 1
                    if k >= 0:
                        _cost_matrix[(k, idx)] = float('inf')

                perm = find_best_permutation(_cost_matrix, current_cost, depth + 1)

                if perm is not None:
                    best_perm = (idx,) + perm
            return best_perm

        return find_best_permutation(score_matrix, 0, 0)


# Dispatchers for permutation solving using a cannot-link graph
graph_permutation_solvers = pb.utils.mapping.Dispatcher({
    'optimal_brute_force': OptimalBruteForceGraphPermutationSolver,
    'optimal_branch_and_bound': OptimalBranchAndBoundGraphPermutationSolver,
    'dfs': DFSGraphPermutationSolver,
    'greedy_cop': GreedyCOPGraphPermutationSolver,
})
