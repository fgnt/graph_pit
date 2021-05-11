from typing import Mapping, Optional

import torch

from graph_pit import graph_pit_loss

from padertorch.contrib.cb.summary import ReviewSummary
from padertorch.contrib.examples.source_separation.tasnet.model import TasNet


class GraphPITTasNet(TasNet):
    def __init__(
            self,
            encoder: torch.nn.Module,
            separator: torch.nn.Module,
            decoder: torch.nn.Module,
            loss: torch.nn.Module,
            mask: bool = True,
            output_nonlinearity: Optional[str] = 'sigmoid',
            num_speakers: int = 2,
            additional_out_size: int = 0,
            sample_rate: int = 8000,
    ):
        super().__init__(
            encoder, separator, decoder, mask, output_nonlinearity,
            num_speakers, additional_out_size, sample_rate,
        )
        self.loss = loss

    def review(self, inputs: dict, outputs: dict) -> Mapping:
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
            outputs['out'],
            inputs['s'],
            inputs['utterance_boundaries'],
            inputs['num_speakers'],
            inputs['y'],
        ):
            # Aggregate loss as sum
            loss = self.loss(
                model_out[:, :num_samples], targets[:, :num_samples],
                utterance_boundaries
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

