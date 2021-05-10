import torch
from padertorch.ops.losses.regression import _reduce
import paderbox as pb


@pb.utils.functional.partial_decorator
def sdr_loss(estimate: torch.Tensor, target: torch.Tensor,
             reduction: str = 'mean', aggregation: str = 'outside_log'):
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
