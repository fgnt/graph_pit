import torch


__all__ = [
    'sdr_loss',
    'SDRLoss',
    'thresholded_sdr_loss',
    'ThresholdedSDRLoss',
]


def _reduce(array, reduction):
    if reduction is None or reduction == 'none':
        return array
    if reduction == 'sum':
        return torch.sum(array)
    elif reduction == 'mean':
        return torch.mean(array)
    else:
        raise ValueError(
            f'Unknown reduction: {reduction}. Choose from "sum", "mean".')


def sdr_loss(estimate: torch.Tensor, target: torch.Tensor,
             reduction: str = 'mean', aggregation: str = 'none'):
    """
    The (scale dependent) SDR or SNR loss.

    Copied and modified from pt.ops.losses.regression.sdr_loss

    Args:
        estimate (... x T): The estimated signal
        target (... x T, same as estimate): The target signal
        reduction: 'mean', 'sum' or 'none'/None for batch dimensions

    Returns:

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> sdr_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(-6.5167)
    >>> sdr_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([-9.8528, -3.1806])

    """
    target_energy = torch.norm(target, dim=-1) ** 2
    error_energy = torch.norm(estimate - target, dim=-1) ** 2
    if aggregation == 'in_fraction':
        target_energy = torch.sum(target_energy, dim=-1)
        error_energy = torch.sum(error_energy, dim=-1)
        sdr = target_energy / error_energy
        sdr = 10 * torch.log10(sdr)
    elif aggregation == 'in_log':
        sdr = target_energy / error_energy
        sdr = torch.sum(sdr, dim=-1)
        sdr = 10 * torch.log10(sdr)
    elif aggregation == 'outside_log':
        sdr = target_energy / error_energy
        sdr = 10 * torch.log10(sdr)
        sdr = torch.sum(sdr, dim=-1)
    elif aggregation == 'none':
        sdr = target_energy / error_energy
        sdr = 10 * torch.log10(sdr)
    else:
        raise ValueError(f'Unknown aggregation type {aggregation}')

    return -_reduce(sdr, reduction=reduction)


class SDRLoss(torch.nn.Module):
    """
    `partial_decorator` doesn't work with configurable (yet?), so we need this
    class wrapper
    """
    def __init__(self, aggregation='outside_log', reduction='sum'):
        super().__init__()
        self.aggregation = aggregation
        self.reduction = reduction

    def forward(self, estimate, target):
        sdr_loss(
            estimate, target,
            aggregation=self.aggregation,
            reduction=self.reduction,
        )


def thresholded_sdr_loss(
        estimate: torch.Tensor,
        target: torch.Tensor,
        threshold: float,
        epsilon: float = 0,
        reduction: str = 'sum',
) -> torch.Tensor:
    """
    Thresholded SDR loss as defined in [1], eq. 2. If `epsilon > 0`, this
    function computes the epsilon-tSDR defined in [2], eq. ?.

    TODO: fill out the details when the paper is accepted

    Args:
        estimate:
        target:
        threshold: The threshold tau in [1], _not_ the max SDR! The max SDR can
            be set with `threshold = 10**(-sdr_max / 10)`
        epsilon: Epsilon from the epsilon-tSDR loss [2]. Makes the function
            robust against perfect silence reconstruction.
        reduction:

    Returns:
        The loss value

    References:
        [1] Wisdom, Scott, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss,
            Kevin Wilson, and John R. Hershey. “Unsupervised Speech Separation
            Using Mixtures of Mixtures.” In ICML 2020 Workshop on
            Self-Supervision for Audio and Speech, 2020.
            https://openreview.net/pdf?id=qMMzJGRPT2d

        [2] Graph-PIT: Generalized permutation invariant training for
            continuous separation of arbitrary numbers of speakers
    """
    target_power = torch.norm(target, dim=-1) ** 2 + epsilon
    sdr = 10 * torch.log10(
        target_power /
        (
                torch.norm(estimate - target, dim=-1) ** 2
                + threshold * target_power
        )
    )
    return -_reduce(sdr, reduction)


class ThresholdedSDRLoss(torch.nn.Module):
    def __init__(
            self,
            max_sdr: float,
            epsilon: float = 0,
            reduction: str = 'sum'
    ):
        """
        Class variant of the thresholded SDR loss so that it can be used with
        `pt.Configurable`. See docstring of `thresholded_sdr_loss` for details.

        Args:
            max_sdr: Used to set the threshold so that the minimum value of the
                loss is limited to `-max_sdr`.
        """
        super().__init__()
        self.max_sdr = max_sdr
        self.threshold = 10 ** (-max_sdr / 10)
        self.epsilon = epsilon
        self.reduction = reduction

    def __call__(
            self, estimate: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return thresholded_sdr_loss(
            estimate, target, self.threshold, self.epsilon, self.reduction
        )

    def extra_repr(self) -> str:
        return (
            f'max_sdr={self.max_sdr}, epsilon={self.epsilon}, '
            f'reduction={self.reduction}'
        )
