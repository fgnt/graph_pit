import torch
from einops import rearrange

import padertorch as pt

from padertorch.modules.dual_path_rnn import apply_examplewise
from padertorch.ops.mappings import ACTIVATION_FN_MAP

from graph_pit.examples.tasnet.coders import ConvolutionalEncoder, \
    ConvolutionalDecoder


class SeparationNetwork(pt.Module):
    def __init__(
            self,
            network,
            num_speakers,
            input_norm=None,
            input_projection=None,
            output_projection=None,
            output_nonlinearity='sigmoid',
    ):
        """
        TODO: rename num_speakers -> num_outputs
        """
        super().__init__()
        self.input_norm = input_norm
        self.input_projection = input_projection
        self.separator_network = network
        self.output_nonlinearity = ACTIVATION_FN_MAP[output_nonlinearity]()
        self.output_prelu = torch.nn.PReLU()
        self.output_projection = output_projection
        self.num_speakers = num_speakers

    def forward(self, signal, sequence_lengths):
        """
        Args:
            signal (B L N): Input signal (B: batch, L: length, N: features)
            sequence_lengths:

        Returns:
            (B K L N)
        """
        if self.input_norm is not None:
            signal = apply_examplewise(
                self.input_norm, signal, sequence_lengths
            )

        if self.input_projection:
            signal = rearrange(signal, 'b l n -> b n l')
            signal = self.input_projection(signal)
            signal = rearrange(signal, 'b n l -> b l n')

        separated = self.separator_network(signal, sequence_lengths)
        separated = rearrange(separated, 'b l n -> b n l')
        separated = self.output_prelu(separated)
        separated = self.output_projection(separated)
        separated = torch.stack(torch.chunk(
            separated, self.num_speakers, dim=1
        ), dim=-1)
        separated = self.output_nonlinearity(separated)
        separated = separated[:, :, :signal.shape[1]]
        separated = rearrange(separated, 'b n l k -> b k l n')
        assert separated.shape[2:] == signal.shape[1:], (
            separated.shape, signal.shape
        )
        assert separated.shape[1] == self.num_speakers, (
            separated.shape, self.num_speakers
        )
        return separated


class MaskingSeparationNetwork(SeparationNetwork):
    def forward(self, signal, sequence_lengths):
        masks = super().forward(signal, sequence_lengths)
        return masks * signal.unsqueeze(1)


class SourceSeparator(pt.Module):
    def __init__(
            self,
            encoder,
            separator,
            decoder,
            correct_output_mean=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder
        self.correct_output_mean = correct_output_mean

    def forward(self, signal, sequence_length=None):
        """

        Args:
            signal (B T): (B: batch, T: time length)
            sequence_length:

        Returns:
            (B K T): The separated signals
        """
        encoded_signal, encoded_sequence_length = self.encoder(
            signal, sequence_length
        )
        separated = self.separator(encoded_signal, encoded_sequence_length)

        reconstructed = rearrange(self.decoder(
            rearrange(separated, 'b k t f -> (b k) t f'),
            encoded_sequence_length
        ), '(b k) t -> b k t',
            b=signal.shape[0],
            k=self.separator.num_speakers
        )

        # The length can be slightly longer than the input length
        reconstructed = reconstructed[..., :signal.shape[-1]]

        # This is necessary if an offset-invariant loss fn (e.g.,
        # SI-SNR from the TasNet paper) but an offset-variant evaluation metric
        # (e.g., SI-SDR) is used.
        # TODO: Fix the loss fn and remove this
        if self.correct_output_mean:
            reconstructed = reconstructed - torch.mean(
                reconstructed, dim=-1, keepdim=True
            )

        return reconstructed


class DPRNNTasNetSeparator(SourceSeparator):
    @classmethod
    def finalize_dogmatic_config(
            cls, config, feature_size=64, hidden_size=64,
            encoder_window_size=16
    ):
        num_speakers = 2

        config['encoder'] = {
            'factory': ConvolutionalEncoder,
            'window_length': encoder_window_size,
            'feature_size': feature_size,
        }

        config['separator'] = {
            'factory': MaskingSeparationNetwork,
            'num_speakers': num_speakers,
            'input_norm': {
                'factory': torch.nn.LayerNorm,
                'normalized_shape': feature_size,
            },
            'input_projection': {
                'factory': torch.nn.Conv1d,
                'in_channels': feature_size,
                'out_channels': hidden_size,
                'kernel_size': 1,
            },
            'network': {
                'factory': pt.modules.dual_path_rnn.DPRNN,
                'input_size': config['encoder']['feature_size'],
                'rnn_size': 128,
                'window_length': 100,
                'hop_size': 50,
                'num_blocks': 6,
            }
        }

        config['separator']['output_projection'] = {
            'factory': torch.nn.Conv1d,
            'in_channels': hidden_size,
            'out_channels': feature_size * config['separator']['num_speakers'],
            'kernel_size': 1,
        }

        config['decoder'] = {
            'factory': ConvolutionalDecoder,
            'window_length': encoder_window_size,
            'feature_size': feature_size,
        }
