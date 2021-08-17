"""
This is a copy of padertorch.contrib.examples.source_separation.tasnet.train

Example call on NT infrastructure:

export STORAGE_ROOT=<your desired storage root>
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m padertorch.contrib.examples.source_separation.tasnet.train with database_json=${paths to your JSON}
"""
# sacred uses things that flake8 doesn't like
# flake8: noqa
import os

import numpy as np
import paderbox as pb
import sacred.commands
import torch
from lazy_dataset.database import JsonDatabase
from pathlib import Path
from sacred import Experiment
from sacred.observers.file_storage import FileStorageObserver
from sacred.utils import InvalidConfigError, MissingConfigError
from paderbox.io.new_subdir import NameGenerator

import padertorch as pt

from graph_pit.examples.tasnet.data import cut_segment, single_channel_scenario_map_fn
from graph_pit.examples.tasnet.model import GraphPITTasNetModel
from graph_pit.examples.tasnet.modules import DPRNNTasNetSeparator

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False
experiment_name = "tasnet-graph-pit"
ex = Experiment(experiment_name)

JSON_BASE = os.environ.get('NT_DATABASE_JSONS_DIR', None)


@ex.config
def config():
    debug = False
    batch_size = 4  # Runs on 4GB GPU mem. Can safely be set to 12 on 12 GB (e.g., GTX1080)
    chunk_size = 32000  # 4s chunks @8kHz

    train_dataset = 'train'
    validate_dataset = 'dev'
    target = 'speech_source'
    load_model_from = None
    database_json = None
    if database_json is None and JSON_BASE:
        database_json = Path(JSON_BASE) / 'sms_wsj_meeting_1ch_8k_kaldi_vad_58spk.json'

    if database_json is None:
        raise MissingConfigError(
            'You have to set the path to the database JSON!', 'database_json')
    if not Path(database_json).exists():
        raise InvalidConfigError('The database JSON does not exist!',
                                 'database_json')

    feat_size = 64
    encoder_window_size = 16
    trainer = {
        "model": {
            "factory": GraphPITTasNetModel,
            'source_separator': {
                'factory': DPRNNTasNetSeparator,
            },
        },
        "storage_dir": None,
        "optimizer": {
            "factory": pt.optimizer.Adam,
            "gradient_clipping": 1
        },
        "summary_trigger": (1000, "iteration"),
        "stop_trigger": (100, "epoch"),
        "loss_weights": {
            "si-sdr": 1.0,
            "log-mse": 0.0,
            "log1p-mse": 0.0,
        }
    }
    pt.Trainer.get_config(trainer)
    if trainer['storage_dir'] is None:
        trainer['storage_dir'] = pt.io.get_new_storage_dir(
            experiment_name, id_naming=NameGenerator(('adjectives', 'animals'))
        )

    ex.observers.append(FileStorageObserver(
        Path(trainer['storage_dir']) / 'sacred')
    )


@ex.capture
def pre_batch_transform(inputs):
    s = np.ascontiguousarray(inputs['audio_data']['speech_image'], np.float32)
    num_samples = inputs['num_samples']['observation']
    utterance_boundaries = [
        (max(start, 0), min(stop, num_samples))
        for start, stop in zip(inputs['speech_onset'], inputs['speech_offset'])
    ]
    s = [s_[start:stop] for s_, (start, stop) in zip(s, utterance_boundaries)]
    assert all(s_.shape[0] > 0 for s_ in s), (s, utterance_boundaries)
    return {
        's': s,
        'y': np.ascontiguousarray(
            inputs['audio_data']['observation'],
            np.float32
        ),
        'num_samples': num_samples,
        'example_id': inputs['example_id'],
        'utterance_boundaries': utterance_boundaries,
        'num_speakers': len(set(inputs['speaker_id'])),
    }


def load_audio(example):
    example['audio_data'] = {
        'original_source': [
            pb.io.load_audio(p)
            for p in example['audio_path']['speech_source']
        ]
    }
    example = single_channel_scenario_map_fn(example)
    return example


def prepare_dataset(
        dataset, batch_size, chunk_size, shuffle=True,
        prefetch=True, dataset_slice=None,
):
    """
    This is re-used in the evaluate script
    """
    if dataset_slice is not None:
        dataset = dataset[dataset_slice]

    def segment(example):
        segment_boundaries = pt.data.segment.get_segment_boundaries(
            num_samples=example['num_samples']['observation'],
            length=chunk_size, shift=chunk_size,
            anchor='random' if shuffle else 'left',
        )

        segments = [
            cut_segment(example, int(start), int(stop))
            for start, stop in segment_boundaries
        ]
        segments = [s for s in segments if s is not None]

        return segments

    dataset = dataset.map(load_audio)
    dataset = dataset.map(segment)

    if shuffle:
        dataset = dataset.shuffle(reshuffle=True)

    dataset = dataset.batch_map(pre_batch_transform)
    # Filter out invalid examples
    dataset = dataset.map(lambda x: [x_ for x_ in x if x_ is not None])

    # FilterExceptions are only raised inside the chunking code if the
    # example is too short. If chunk_size == -1, no filter exception is raised.
    if prefetch:
        dataset = dataset.prefetch(8, 16, catch_filter_exception=True)
    else:
        dataset = dataset.catch()

    dataset = dataset.unbatch()

    if shuffle:
        dataset = dataset.shuffle(reshuffle=True, buffer_size=128)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(pt.data.batch.Sorter('num_samples'))
    dataset = dataset.map(pt.data.utils.collate_fn)

    return dataset


@ex.capture
def prepare_dataset_captured(
        database_obj, dataset, batch_size, debug, chunk_size,
        shuffle, dataset_slice=None,
):
    if dataset_slice is None:
        if debug:
            dataset_slice = slice(0, 100, 1)

    return prepare_dataset(
        database_obj.get_dataset(dataset), batch_size, chunk_size,
        shuffle=shuffle,
        prefetch=not debug,
        dataset_slice=dataset_slice,
    )


@ex.capture
def dump_config_and_makefile(_config):
    """
    Dumps the configuration into the experiment dir and creates a Makefile
    next to it. If a Makefile already exists, it does not do anything.
    """
    experiment_dir = Path(_config['trainer']['storage_dir'])
    makefile_path = Path(experiment_dir) / "Makefile"

    if not makefile_path.exists():
        from padertorch.contrib.examples.source_separation.tasnet.templates \
            import MAKEFILE_TEMPLATE_TRAIN

        config_path = experiment_dir / "config.json"
        pt.io.dump_config(_config, config_path)

        makefile_path.write_text(
            MAKEFILE_TEMPLATE_TRAIN.format(
                main_python_path=pt.configurable.resolve_main_python_path(),
                experiment_name=experiment_name,
                eval_python_path=('.'.join(
                    pt.configurable.resolve_main_python_path().split('.')[:-1]
                ) + '.evaluate')
            )
        )


@ex.command(unobserved=True)
def init(_config, _run):
    """Create a storage dir, write Makefile. Do not start any training."""
    sacred.commands.print_config(_run)
    dump_config_and_makefile()

    print()
    print('Initialized storage dir. Now run these commands:')
    print(f"cd {_config['trainer']['storage_dir']}")
    print(f"make train")
    print()
    print('or')
    print()
    print(f"cd {_config['trainer']['storage_dir']}")
    print('make ccsalloc')


@ex.capture
def prepare_and_train(_run, _log, trainer, train_dataset, validate_dataset,
                      load_model_from, database_json):
    trainer = get_trainer(trainer, load_model_from, _log)

    db = JsonDatabase(database_json)

    train_dataset = prepare_dataset_captured(db, train_dataset, shuffle=True)
    validate_dataset = prepare_dataset_captured(
        db, validate_dataset, shuffle=False
    )

    # Perform a test run to check if everything works
    trainer.test_run(train_dataset, validate_dataset)

    # Register hooks and start the actual training
    trainer.register_validation_hook(validate_dataset)
    trainer.train(train_dataset, resume=trainer.checkpoint_dir.exists())


def get_trainer(trainer_config, load_model_from, _log):
    trainer = pt.Trainer.from_config(trainer_config)

    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'
    if load_model_from is not None and not checkpoint_path.is_file():
        _log.info(f'Loading model weights from {load_model_from}')
        checkpoint = torch.load(load_model_from)
        trainer.model.load_state_dict(checkpoint['model'])

    return trainer


@ex.command
def test_run(_run, _log, trainer, train_dataset, validate_dataset,
             load_model_from, database_json):
    trainer = get_trainer(trainer, load_model_from, _log)

    db = JsonDatabase(database_json)

    # Perform a test run to check if everything works
    trainer.test_run(
        prepare_dataset_captured(db, train_dataset, shuffle=True),
        prepare_dataset_captured(db, validate_dataset, shuffle=True),
    )


@ex.main
def main(_config, _run):
    """Main does resume directly.

    It also writes the `Makefile` and `config.json` again, even when you are
    resuming from an initialized storage dir. This way, the `config.json` is
    always up to date. Historic configuration can be found in Sacred's folder.
    """
    sacred.commands.print_config(_run)
    dump_config_and_makefile()
    prepare_and_train()


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
