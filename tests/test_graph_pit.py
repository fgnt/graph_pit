import pytest
import torch
import numpy as np

from graph_pit import graph_pit_loss


def test_graph_pit():
    rng = np.random.default_rng(42)

    # Create three target utterances and two estimated signals
    targets = [
        torch.tensor(rng.random(100)),
        torch.tensor(rng.random(200)),
        torch.tensor(rng.random(150))
    ]
    segment_boundaries = [(0, 100), (150, 350), (300, 450)]
    estimate = torch.tensor(rng.random((2, 500)))

    # Compute loss
    loss = graph_pit_loss(
        estimate, targets, segment_boundaries,
        torch.nn.functional.mse_loss
    )
    np.testing.assert_allclose(float(loss), 0.26560837)

    # Loss should be 0 for perfect input
    targets = [torch.ones(100), torch.ones(200), torch.ones(150)]
    estimate = torch.zeros(2, 500)
    estimate[0, 0:100] = 1
    estimate[1, 150:350] = 1
    estimate[0, 300:450] = 1
    loss = graph_pit_loss(
        estimate, targets, segment_boundaries,
        torch.nn.functional.mse_loss
    )
    np.testing.assert_allclose(float(loss), 0)


def test_graph_pit_exceptions():
    with pytest.raises(
            ValueError, match='No coloring found for graph!'
    ):
        graph_pit_loss(
            torch.zeros(2, 100), torch.zeros(3, 100),
            [(0, 100), (0, 100), (0, 100)], torch.nn.functional.mse_loss
        )

    with pytest.raises(
            ValueError,
            match='The number of targets doesn\'t match the number of '
                  'segment boundaries!'
    ):
        graph_pit_loss(torch.zeros(2, 100), torch.zeros(3, 100),
                       [(0, 100), (0, 100)], torch.nn.functional.mse_loss)

    with pytest.raises(
            ValueError,
            match='Length mismatch between target and segment_boundaries'
    ):
        graph_pit_loss(torch.zeros(3, 100), torch.zeros(2, 100),
                       [(0, 50), (0, 100)], torch.nn.functional.mse_loss)


if __name__ == '__main__':
    pytest.main()
