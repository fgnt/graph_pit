import copy
import numpy as np
import paderbox as pb
from sms_wsj.database.utils import extract_piece, get_white_noise_for_signal


def cut_segment(
        source_example: dict, start: int, stop: int,
) -> dict:
    """
    Cut one segment from an example. Can be applied before or after
    `load_example`.
    """
    if stop is None:
        stop = source_example['num_samples']['observation']

    if not isinstance(start, int) or not isinstance(stop, int):
        raise TypeError(
            f'start and stop must be integers, but got start={start!r} '
            f'({type(start)}) and stop={stop!r} ({type(stop)})'
        )

    # Compute which "utterances" (i.e., entry in original_source) are active in
    # this segment
    active_utterances = [
        onset_ < stop and start < offset_
        for onset_, offset_ in zip(
            source_example['speech_onset'],
            source_example['speech_offset']
        )
    ]

    if all(not a for a in active_utterances):
        return None

    # Remove inactive entries. Do the copy in here and only copy those
    # entries that are needed (huge speedup compared to a deepcopy on the
    # full example)
    def copy_value(key, value):
        if not key.startswith('audio_data'):
            return copy.deepcopy(value)

        if key != 'audio_data.rirs' and key != 'audio_data.original_source':
            value = value[..., start:stop].copy()

        return value

    def remove_inactive(k, value):
        if isinstance(value, list):
            return [
                copy_value(k, v)
                for v, c_active in zip(value, active_utterances) if c_active
            ]
        elif isinstance(value, np.ndarray) and value.ndim >= 2:
            if value.ndim >= 2:
                return np.stack([
                    copy_value(k, v)
                    for v, c_active in zip(value, active_utterances)
                    if c_active
                ])
        return copy_value(k, value)

    flattened = pb.utils.nested.flatten(source_example)
    flattened = {k: remove_inactive(k, v) for k, v in flattened.items()}
    example = pb.utils.nested.deflatten(flattened)

    # Set some additional info of the example
    example['start'] = start
    example['stop'] = stop
    example['source_example_id'] = source_example['example_id']
    example['num_speakers'] = len(set(example['speaker_id']))

    # Shift offsets
    example['offset'] = [o - start for o in example['offset']]
    example['num_samples']['observation'] = stop - start
    if 'speech_onset' in example:
        example['speech_onset'] = [o - start for o in example['speech_onset']]
        example['speech_offset'] = [
            o - start for o in example['speech_offset']
        ]

    return example


def load_example(example):
    example['audio_data'] = {
        'original_source': [
            pb.io.load_audio(path)
            for path in example['audio_path']['speech_source']
        ]
    }

    # Create meeting from loaded source signals. This creates the
    # observation and padded target signals. Offsets can be negative, then
    # the start of the utterance lies before the beginning of the meeting
    example = single_channel_scenario_map_fn(example)

    return example


def get_scale(log_weights, signals):
    std = np.maximum(
        np.std(signals, axis=-1, keepdims=True),
        np.finfo(signals.dtype).tiny,
    )
    log_weights = np.asarray(log_weights)

    # Bring into the correct shape
    log_weights = log_weights.reshape((-1,) + (1,) * (signals.ndim - 1))

    scale = (10 ** (log_weights / 20)) / std

    # divide by 71 to ensure that all values are between -1 and 1 (WHY 71?)
    scale /= 71

    return scale


def add_microphone_noise(example, snr_range):
    if snr_range is not None:
        example_id = example['example_id']
        rng = pb.utils.random_utils.str_to_random_generator(example_id)
        example["snr"] = snr = rng.uniform(*snr_range)

        rng = pb.utils.random_utils.str_to_random_generator(example_id)
        mix = example['audio_data']['observation']
        n = get_white_noise_for_signal(mix, snr=snr, rng_state=rng)
        example['audio_data']['noise_image'] = n
        mix += n
        example['audio_data']['observation'] = mix


def single_channel_scenario_map_fn(
        example,
        *,
        snr_range: tuple = (20, 30),
        normalize_sources: bool = True,
):
    """
    Constructs the observation and scaled speech source signals for `example`
    for the single-channel no reverberation case.
    """
    T = example['num_samples']['observation']
    s = example['audio_data']['original_source']
    offset = example['offset']

    # In some databases (e.g., WSJ) the utterances are not mean normalized.
    # This leads to jumps when padding with zeros or concatenating recordings.
    # We mean-normalize here to eliminate these jumps
    if normalize_sources:
        s = [s_ - np.mean(s_) for s_ in s]

    # Move and pad speech source to the correct position
    x = [extract_piece(s_, offset_, T) for s_, offset_ in zip(s, offset)]
    x = np.stack(x)

    # Scale the sources by log_weights
    x *= get_scale(example['log_weights'], x)

    # The mix is now simply the sum over the speech sources
    mix = np.sum(x, axis=0)

    example['audio_data']['observation'] = mix
    example['audio_data']['speech_image'] = x

    # Add noise if snr_range is specified. RNG depends on example ID (
    # deterministic)
    add_microphone_noise(example, snr_range)

    return example
