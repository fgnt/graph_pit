from typing import Tuple, Union

import torch
from einops import rearrange

from padertorch.contrib.examples.source_separation.tasnet.tas_coders import \
    TasEncoder as _TasEncoder, TasDecoder as _TasDecoder


class ConvolutionalEncoder(_TasEncoder):
    """Same as TasEncoder, but returns shape (B T F)"""
    def forward(
            self,
            x: torch.Tensor,
            sequence_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        out, sequence_lengths = super().forward(x, sequence_lengths)
        return rearrange(out, '... f t -> ... t f'), sequence_lengths


class ConvolutionalDecoder(_TasDecoder):
    """Same as TasDecoder, but expects shape (B T F)"""
    def forward(
            self,
            x: torch.Tensor,
            sequence_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        return super().forward(rearrange(x, '... t f -> ... f t'))
