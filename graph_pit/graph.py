import itertools
from cached_property import cached_property
from typing import List, Tuple, Optional, Any, Set


class Graph:
    def __init__(self, num_vertices: int, labels: Optional[List[Any]] = None):
        self.num_vertices = num_vertices
        assert labels is None or isinstance(labels, (list, tuple))
        self.labels = labels

    @classmethod
    def from_edge_list(cls, num_vertices, edge_list, labels=None) -> 'Graph':
        """
        Expects `edge_list` to be a list of tuples of integers indexing into
            num_vertices.

        Note:
            `edge_list` does _not_ contain the labels, but the indices of the
            vertices to connect.
        """
        return EdgeListGraph(num_vertices, edge_list, labels)

    @classmethod
    def from_adjacency_list(cls, num_vertices, edge_list) -> 'Graph':
        raise NotImplementedError()

    @classmethod
    def from_adjacency_matrix(cls, adjacency_matrix) -> 'Graph':
        # For now, convert the adjacency matrix into an edge list
        assert adjacency_matrix.ndim == 2, adjacency_matrix.shape
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], (
            adjacency_matrix.shape
        )
        num_vertices = adjacency_matrix.shape[0]
        edge_list = set()
        for i, x in enumerate(adjacency_matrix):
            for j, v in enumerate(x):
                edge_list.add(tuple(sorted((i, j))))
        return cls.from_edge_list(num_vertices, edge_list)

    @property
    def vertices(self) -> List[int]:
        """List of vertices of this graph."""
        return list(range(self.num_vertices))

    @property
    def adjacency_list(self) -> List[Set[int]]:
        """Adjancency list representation of this graph."""
        raise NotImplementedError()

    @property
    def adjacency_matrix(self):
        """Adjacency matrix representation of this graph."""
        raise NotImplementedError()

    @property
    def edge_list(self) -> List[Tuple[int, int]]:
        """
        Edge list representation of this graph.

        Returns:
            A list of tuples of length two where each entry in the list
            represents one edge.
        """
        raise NotImplementedError()

    @cached_property
    def cycle_free(self):
        """
        Checks if the graph is cycle-free.

        Returns:
            `True` if graph is cycle-free and `False` if it contains a cycle.

        Examples:
            >>> EdgeListGraph(3, [(0, 1), (1, 2)]).cycle_free
            True
            >>> EdgeListGraph(3, [(0, 1), (1, 2), (0, 2)]).cycle_free
            False

        """
        visited = set()

        # For all unvisited vertices:
        for vertex in self.vertices:
            if vertex in visited:
                continue

            vertices = [(vertex, None)]

            while vertices:
                vertex, parent = vertices.pop()
                visited.add(vertex)
                for neighbor in self.neighbors_of(vertex):
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        return False
                    vertices.append((neighbor, vertex))

        return True

    def has_coloring(self, max_num_colors: int):
        """
        Checks if this graph is `max_num_colors`-colorable.

        Args:
            max_num_colors:

        Returns:

        """
        for _ in self.enumerate_graph_colorings(max_num_colors):
            return True
        return False

    def indices_to_labels(self, indices):
        assert self.labels is not None
        return [self.labels[i] for i in indices]

    def enumerate_graph_colorings(
            self,
            max_num_colors: int,
            min_num_colors: int = 1
    ):
        """
        Greedy brute-force enumeration of all graph colorings with `num_nodes`
        vertices, given the `edges`, with the maximum number of colors
        `num_colors`.

        Notes:
            This is not an particularly efficient implementation, but for its
            purpose this should be efficient enough.

        Args:
            num_nodes: Number of nodes in graph
            edges: List of edges. Each edge is represented as a tuple of
                vertices between which to insert an edge
            num_colors: Number colors to use

        Yields:
            List of lists of int of length `self.num_vertices`, each integer
            entry represents one color.

        Example:
            For a fully connected graph, this is equivalient to enumerating all
            permutations
            >>> list(Graph.from_edge_list(3, [(0, 1), (0, 2), (1, 2)])
            ...      .enumerate_graph_colorings(3))
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
            >>> list(itertools.permutations(range(3)))
            [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

        """

        def _can_color(vertex, color, coloring):
            """Check if vertex can be colored with color given the (incomplete)
            coloring coloring"""
            for r in self.neighbors_of(vertex):
                if len(coloring) > r and coloring[r] == color:
                    return False
            return True

        def color_vertex(i):
            if i == 0:
                # We can choose any color for the first vertex
                for i in range(max_num_colors):
                    yield [i]
            else:
                # For the following vertices we have to check if the color is
                # valid
                for coloring in color_vertex(i - 1):
                    for color in range(max_num_colors):
                        if _can_color(i, color, coloring):
                            yield coloring + [color]

        for coloring in color_vertex(self.num_vertices - 1):
            if len(set(coloring)) >= min_num_colors:
                yield coloring

    @cached_property
    def chromatic_number(self) -> int:
        """
        Minimum number of colors required to color the graph.

        This is cached because it is NP-hard to compute.
        """
        # Finding the chromatic number is NP-hard, so we can simply enumerate
        # all colorings and find the one with the smallest number of colors.
        chromatic_number = 0
        for component in self.connected_components:
            local_chromatic_number = self.num_vertices

            found = False

            while not found:
                found = True
                for coloring in component.enumerate_graph_colorings(
                        local_chromatic_number
                ):
                    c = len(set(coloring))
                    if c < local_chromatic_number:
                        # If we found a coloring with less than
                        # local_chromatic_number colors, we can re-start the
                        # enumeration process with this number of colors
                        local_chromatic_number = c
                        if local_chromatic_number > 1:
                            found = False
                        break
            if local_chromatic_number > chromatic_number:
                chromatic_number = local_chromatic_number
        return chromatic_number

    def neighbors_of(self, vertex: int) -> Set[int]:
        """Neighbors of the vertex `vertex`."""
        return self.adjacency_list[vertex]

    @cached_property
    def num_connected_components(self) -> int:
        """
        The number of connected components of this graph.
        """
        return len(self.connected_components)

    @cached_property
    def connected_components(self) -> List['Graph']:
        """
        A list of all connected components of this graph.
        """
        return list(self.enumerate_connected_components())

    def enumerate_connected_components(self):
        """
        Enumerates all connected components of this graph.

        Yields:
            Graph
            All connected components
        """
        vertices = list(range(self.num_vertices))

        while vertices:
            # Find all vertices in a connected component
            i = 0
            component = [vertices.pop(0)]

            while i < len(component):
                neighbors = self.neighbors_of(component[i]) - set(component)
                for neighbor in neighbors:
                    if neighbor in vertices:
                        vertices.remove(neighbor)
                component.extend(neighbors)
                i += 1

            # Build the new adjacency list. We have to make sure that the new
            # adjacency list starts to index at 0 (i.e., index into component)
            adjacency_list = [{
                component.index(n) for n in self.adjacency_list[c]
            } for c in component]

            if self.labels is None:
                labels = component
            else:
                labels = self.indices_to_labels(component)

            yield AdjacencyListGraph(len(component), adjacency_list, labels)


class EdgeListGraph(Graph):
    """
    Graph defined using an edge-list
    """
    def __init__(
            self, num_vertices: int, edge_list: List[Tuple[int, int]],
            labels=None
    ):
        super().__init__(num_vertices, labels)
        self._edge_list = list(edge_list)
        if not all(0 <= e < num_vertices
                   for e in itertools.chain.from_iterable(edge_list)):
            raise ValueError(
                f'Invalid index in edge list {self._edge_list} for graph with '
                f'{num_vertices} vertices!'
            )

    @cached_property
    def adjacency_list(self):
        return [
            set(itertools.chain.from_iterable(
                [e for e in self.edge_list if n in e]
            )) - {n} for n in range(self.num_vertices)
        ]

    @cached_property
    def adjacency_matrix(self):
        raise NotImplementedError('TODO')

    @property
    def edge_list(self):
        return self._edge_list

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'{self.num_vertices}, {self.edge_list})'


class AdjacencyListGraph(Graph):
    """
    Graph defined using an adjacency list.
    """
    def __init__(self, num_vertices, adjacency_list, labels=None):
        super().__init__(num_vertices, labels)
        self._adjacency_list = adjacency_list

    @cached_property
    def adjacency_list(self):
        return self._adjacency_list

    @cached_property
    def edge_list(self):
        return list(set(
            tuple(sorted((x, y)))
            for x, l in enumerate(self.adjacency_list)
            for y in l
        ))
