"""
This file contains algorithms for finding graph-pit assignments similar to
tvn.algorithm.permutation_solving but with cannot-link constraints described by
a cannot-link graph.

References:
    [1] Speeding Up Permutation Invariant Training for Source Separation
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

import paderbox as pb
from .graph import Graph

import logging

logger = logging.getLogger('graph-assignment-solver')


def _find_mapping_apply_connected_components(
        assignment_solver, score_matrix, cannot_link_graph, **kwargs
):
    num_targets, num_estimates = score_matrix.shape
    assert num_targets == cannot_link_graph.num_vertices, (
        num_targets, cannot_link_graph.num_vertices
    )

    mapping = np.zeros(num_targets, dtype=np.int)

    for connected_component in cannot_link_graph.connected_components:
        cc_score_matrix = score_matrix[(connected_component.labels, slice(None))]
        partial_solution = assignment_solver(
            cc_score_matrix, connected_component, **kwargs
        )
        if partial_solution is None:
            return None
        mapping[connected_component.labels] = partial_solution

    return mapping


@dataclass
class GraphAssignmentSolver:
    """
    Base class for graph-based assignment solvers.

    Attributes:
        minimize: If `True`, find the minimum. Otherwise, find the maximum
        optimize_connected_components: If `True`, the assignment algorithm is
            applied to each connected component of the graph individually.
    """
    minimize: bool = False
    optimize_connected_components: bool = True

    def __call__(self, score_matrix, cannot_link_graph):
        score_matrix = np.asanyarray(score_matrix)
        self._check_inputs(score_matrix, cannot_link_graph)

        if self.optimize_connected_components:
            coloring = _find_mapping_apply_connected_components(
                self.find_assignment, score_matrix, cannot_link_graph
            )
        else:
            coloring = self.find_assignment(score_matrix, cannot_link_graph)

        if coloring is None:
            num_targets, num_estimates = score_matrix.shape
            logger.debug(
                f'Couldn\'t find a solution with the permutation solver '
                f'{self.__class__.__name__}. This could mean that the '
                f'cannot_link_graph is not colorable with {num_estimates} '
                f'colors.'
            )
        return coloring

    def _check_inputs(self, score_matrix, cannot_link_graph):
        num_targets, num_estimates = score_matrix.shape

        if not np.issubdtype(score_matrix.dtype, np.floating):
            raise TypeError(
                f'The score matrix must be floating point, not '
                f'{score_matrix.dtype}!'
            )
        if num_targets != cannot_link_graph.num_vertices:
            raise ValueError(
                f'The shape of score_matrix and number of vertices in the'
                f' cannot_link_graph must match, but '
                f'score_matrix.shape={score_matrix.shape}, '
                f'cannot_link_graph.num_vertices={cannot_link_graph.num_vertices}'
            )

    def find_assignment(self, score_matrix: np.ndarray, cannot_link_graph: Graph):
        """
        Runs the assignment algorithm.

        Args:
            score_matrix (num_targets num_estimates): The score matrix to use
            cannot_link_graph: The graph that describes the overlaps / entries
                in `score_matrix` that cannot be used together. Has to have
                `num_targets` vertices.
        """
        raise NotImplementedError()


class OptimalBruteForceGraphAssignmentSolver(GraphAssignmentSolver):
    """
    A brute-force assignment algorithm. Tests every possible permitted
    assignment and returns the one with the smallest (or largest, if
    `minimize`=False) score.
    """
    def find_assignment(self, score_matrix, cannot_link_graph):
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
class GreedyCOPGraphAssignmentSolver(GraphAssignmentSolver):
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

      References:
          [1] Wagstaff, Kiri, Claire Cardie, Seth Rogers, and Stefan Schroedl.
              “Constrained K-Means Clustering with Background Knowledge,”
      """
    def find_assignment(self, score_matrix, cannot_link_graph):
        if not self.minimize:
            score_matrix = -score_matrix

        num_targets, num_estimates = score_matrix.shape
        coloring = -np.ones(num_targets, dtype=np.int)

        # Copy the score matrix because we are going to modify it
        score_matrix: np.ndarray = score_matrix.copy()
        # score_matrix_flat is a view in score_matrix
        # -> changing score_matrix also changes score_matrix_flat
        score_matrix_flat = score_matrix.reshape(num_targets * num_estimates)

        mask = np.zeros(score_matrix.shape, dtype=bool)
        mask_flat = mask.reshape(num_targets * num_estimates)

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
class DFSGraphAssignmentSolver(GraphAssignmentSolver):
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

    Returns:
        Permutation with shape (N). Each entry in the returned array is
        an index along the K axis.
    """

    def find_assignment(self, score_matrix, cannot_link_graph):
        stack = deque()
        N, K = score_matrix.shape

        if not self.minimize:
            score_matrix = -score_matrix

        score_matrix_flat = score_matrix.reshape(N * K)
        sorted_indices = np.argsort(score_matrix_flat)

        # a state is (coloring, sorted_index, mask, depth)
        # Saving a boolean mask saves some memory compared to saving the full
        # modified score matrix.
        stack.append(
            (-np.ones(N, dtype=np.int), 0,
             np.zeros_like(score_matrix, dtype=np.bool), 0)
        )

        while stack:
            coloring, sorted_index, mask, depth = stack.pop()
            mask_flat = mask.reshape(N * K)

            for idx in range(sorted_index, score_matrix_flat.shape[0]):
                sidx = sorted_indices[idx]
                if mask_flat[sidx]:
                    continue

                # Push current state
                i, j = np.unravel_index(sidx, score_matrix.shape)
                stack.append((coloring.copy(), idx + 1, mask.copy(), depth))

                # update state
                coloring[i] = j
                mask[i, :] = True
                for k in cannot_link_graph.adjacency_list[i]:
                    mask[k, j] = True
                depth = depth + 1
                if depth == N:
                    return coloring


class OptimalBranchAndBoundGraphAssignmentSolver(GraphAssignmentSolver):
    """
    A bran-and-bound algorithm to find the optimal solution to the graph
    permutation problem.

    Runtime:
        Has the same worst-case complexity as the brute-force variant, but is in
        many cases a lot faster. The better the scores are, the faster the
        algorithm becomes.
    """
    def find_assignment(self, score_matrix, cannot_link_graph):
        num_targets, num_estimates = score_matrix.shape

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

                if depth == num_targets - 1:
                    # We are at a leaf (or, one before leaf node)
                    best_cost = current_cost
                    return (idx,)

                # We are not at a leaf
                _cost_matrix = cost_matrix[1:].copy()
                for k in cannot_link_graph.neighbors_of(depth):
                    k = k - depth - 1
                    if k >= 0:
                        _cost_matrix[(k, idx)] = float('inf')

                perm = find_best_permutation(
                    _cost_matrix, current_cost, depth + 1
                )

                if perm is not None:
                    best_perm = (idx,) + perm
            return best_perm

        return find_best_permutation(score_matrix, 0, 0)


@dataclass
class OptimalDynamicProgrammingAssignmentSolver(GraphAssignmentSolver):
    """
    An assignment algorithm that runs in num_colors**num_colors * num_vertices
    time, so it is linear in the number of vertices (but exponential in the
    number of colors).

    Warnings:
        Assumes that the nodes are sorted so that traversing them in order never
        leaves more than num_colors nodes in a partial state (i.e., some of its
        neighbors are colored and some are not). If this is violated, an
        assertion error is raised.
    """
    def find_assignment(self, score_matrix, cannot_link_graph: Graph):
        # TODO: Check if adjacency list is valid (i.e., sorted)
        adjacency_list = [
            [n for n in neighbors if n < node]
            for node, neighbors in enumerate(cannot_link_graph.adjacency_list)
        ]
        num_colors = len(score_matrix[0])
        all_colors = set(range(num_colors))

        if not self.minimize:
            score_matrix = -score_matrix
        score_matrix -= np.min(score_matrix)

        candidates = {
            (i,):  # currently relevant nodes color sequence
                (
                    (i,),  # coloring
                    score,  # score
                )
            for i, score in enumerate(score_matrix[0])
        }

        for node in range(1, score_matrix.shape[0]):
            neighbors = adjacency_list[node]

            state_nodes = tuple(neighbors) + (node,)

            candidates_old = candidates.copy()
            candidates = {}

            for state, (coloring, score) in candidates_old.items():

                blocked_colors = {coloring[n] for n in neighbors}
                allowed_colors = all_colors - blocked_colors

                for c in allowed_colors:
                    new_coloring = coloring + (c,)
                    new_state = tuple([new_coloring[n] for n in state_nodes])
                    new_score = score + score_matrix[node][c]

                    if new_state in candidates:
                        if new_score >= candidates[new_state][1]:
                            continue

                    candidates[new_state] = (new_coloring, new_score)
        return min(candidates.items(), key=lambda x: x[1][1])[1][0]


# Dispatchers for permutation solving using a cannot-link graph
graph_assignment_solvers = pb.utils.mapping.Dispatcher({
    'optimal_brute_force': OptimalBruteForceGraphAssignmentSolver,
    'optimal_branch_and_bound': OptimalBranchAndBoundGraphAssignmentSolver,
    'optimal_dynamic_programming': OptimalDynamicProgrammingAssignmentSolver,
    'dfs': DFSGraphAssignmentSolver,
    'greedy_cop': GreedyCOPGraphAssignmentSolver,
})
