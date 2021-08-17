import torch
import padertorch as pt
from padertorch.contrib.cb.summary import ReviewSummary

from graph_pit.examples.tasnet.modules import DPRNNTasNetSeparator
from graph_pit.loss import OptimizedGraphPITSourceAggregatedSDRLossModule
from graph_pit.loss.base import LossModule


class GraphPITTasNetModel(pt.Model):
    """
    A Time-Domain Audio Separation Model.

    Expects a time-domain signal as input and outputs a time-domain signal.
    """

    def __init__(
            self,
            source_separator: DPRNNTasNetSeparator,
            loss: LossModule,
            sample_rate: int = 8000,
    ):
        super().__init__()
        self.source_separator = source_separator
        self.loss = loss
        self.sample_rate = sample_rate

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """Default config: sa-SDR with DP solver"""
        config['loss'] = {
            'factory': OptimizedGraphPITSourceAggregatedSDRLossModule,
            'assignment_solver': 'optimal_dynamic_programming',
        }

    def example_to_device(self, example, device=None):
        for k in ('s', 'y'):
            example[k] = pt.data.example_to_device(example[k], device=device)
        return example

    def forward(self, example: dict):
        sequence = pt.pad_sequence(example['y'], batch_first=True)
        sequence_lengths = example['num_samples']
        if not torch.is_tensor(sequence_lengths):
            sequence_lengths = torch.tensor(sequence_lengths)
        return self.source_separator(sequence, sequence_lengths)

    def review(self, inputs: dict, outputs: dict):
        review = ReviewSummary(sampling_rate=self.sample_rate)

        for (
                # Required for loss
                num_samples,
                model_out,
                targets,
                utterance_boundaries,
                # Required for reporting
                num_speakers,
                observation,
        ) in zip(
            inputs['num_samples'],
            outputs,
            inputs['s'],
            inputs['utterance_boundaries'],
            inputs['num_speakers'],
            inputs['y'],
        ):
            # Aggregate loss as sum
            loss = self.loss(
                model_out[:, :num_samples], targets, utterance_boundaries
            )

            review.add_to_loss(loss)
            review.add_histogram('loss_', loss)

            # Report losses with number of speakers for easier analysis
            review.add_scalar(f'loss/{num_speakers}spk', loss)

            # Report histograms
            review.add_histogram('num_speakers', num_speakers)
            review.add_histogram('num_samples', num_samples)

            # Report audios with number of speakers
            review.add_audio(f'{num_speakers}spk/observation', observation)
            for idx, estimate in enumerate(model_out):
                review.add_audio(f'{num_speakers}spk/estimate_{idx}', estimate)
            # Targets don't make too much sense because their number differs
            # between examples so they get messed up in tensorboard

        return review
