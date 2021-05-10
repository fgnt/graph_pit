import itertools

import pytest

from graph_pit.graph import Graph


@pytest.mark.parametrize('num_vertices', [2, 10, 15,  25])
def test_num_connected_components_no_edges(num_vertices):
    # Without edges we have as many components as vertices
    graph = Graph.from_edge_list(num_vertices, [])
    assert len(graph.connected_components) == num_vertices


@pytest.mark.parametrize('num_vertices,num_colors', [
    (3, 2), (2, 3), (5, 3)
])
def test_coloring_no_edges(num_vertices, num_colors):
    graph = Graph.from_edge_list(num_vertices, [])
    num_colorings = len(list(graph.enumerate_graph_colorings(num_colors)))

    assert num_colorings == num_colors ** num_vertices


@pytest.mark.parametrize(
    'num_vertices,edges,num_colors,num_colorings',
    [
        (3, [], 2, 2**3),               # No edges
        (3, [(0, 1), (1, 2)], 2, 2),    # Fully connected
    ]
)
def test_coloring(num_vertices, edges, num_colors, num_colorings):
    graph = Graph.from_edge_list(num_vertices, edges)
    colorings = list(graph.enumerate_graph_colorings(num_colors))

    # Assert that all colorings are unique
    assert len(colorings) == len(set(map(tuple, colorings)))

    # Assert that we use the correct color representation
    all_colors = set(itertools.chain.from_iterable(colorings))
    assert min(all_colors) >= 0
    assert max(all_colors) < num_colors

    # Assert number of colorings is correct
    assert len(colorings) == num_colorings


if __name__ == '__main__':
    pytest.main()
