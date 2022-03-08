import itertools
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Segment:
    index: int
    start: int
    end: int

    @classmethod
    def segments_from_segment_boundaries(cls, segment_boundaries):
        return [
            cls(idx, start, stop)
            for idx, (start, stop) in enumerate(segment_boundaries)
        ]

    @staticmethod
    def overlaps(r1: 'Segment', r2: 'Segment'):
        return r1.start < r2.end and r2.start < r1.end


def get_overlaps_from_segment_boundaries(
        segment_boundaries: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    A totally not optimized function that outputs tuples of indices into
    `segment_boundaries` where these segments overlap.

    Examples:
        >>> segment_boundaries = [
        ...    (0, 10), (11, 14), (12, 17), (16, 20), (21, 30)
        ... ]
        >>> sorted(get_overlaps_from_segment_boundaries(segment_boundaries))
        [(1, 2), (2, 3)]
    """
    segments = Segment.segments_from_segment_boundaries(segment_boundaries)

    overlaps = set()
    for r1, r2 in itertools.combinations(segments, 2):
        if Segment.overlaps(r1, r2):
            overlaps.add(tuple(sorted((r1.index, r2.index))))

    return list(overlaps)


def to_numpy(array, detach: bool = False, copy: bool = False) -> np.ndarray:
    """
    Transforms `array` to a numpy array. `array` can be anything that
    `np.asarray` can handle and torch tensors.

    Copied from padertorch to minimize depenencies

    Args:
        array: The array to transform to numpy
        detach: If `True`, `array` gets detached if it is a `torch.Tensor`.
            This has to be enabled explicitly to prevent unintentional
            truncation of a backward graph.
        copy: If `True`, the array gets copied. Otherwise, it becomes read-only
            to prevent unintened changes on the input array or tensor by
            altering the output.

    Returns:
        `array` as a numpy array.

    >>> t = torch.zeros(2)
    >>> t
    tensor([0., 0.])
    >>> to_numpy(t), np.zeros(2, dtype=np.float32)
    (array([0., 0.], dtype=float32), array([0., 0.], dtype=float32))

    >>> t = torch.zeros(2, requires_grad=True)
    >>> t
    tensor([0., 0.], requires_grad=True)
    >>> to_numpy(t, detach=True), np.zeros(2, dtype=np.float32)
    (array([0., 0.], dtype=float32), array([0., 0.], dtype=float32))

    """
    # if isinstance(array, torch.Tensor):
    try:
        array = array.cpu()
    except AttributeError:
        pass
    else:
        if detach:
            array = array.detach()

    try:
        # torch only supports np.asarray for cpu tensors
        if copy:
            return np.array(array)
        else:
            array = np.asarray(array)
            array.setflags(write=False)
            return array
    except TypeError as e:
        raise TypeError(type(array), array) from e
    except RuntimeError as e:
        import sys
        raise type(e)(str(e) + (
            '\n\n'
            'It is likely, that you are evaluating a model in train mode.\n'
            'You may want to call `model.eval()` first and use a context\n'
            'manager, which disables gradients: `with torch.no_grad(): ...`.\n'
            'If you want to detach anyway, use `detach=True` as argument.'
            )
        ) from e


class DispatchError(KeyError):
    def __str__(self):
        if len(self.args) == 2 and isinstance(self.args[0], str):
            item, keys = self.args
            import difflib
            # Suggestions are sorted by their similarity.
            suggestions = difflib.get_close_matches(
                item, keys, cutoff=0, n=100
            )
            return f'Invalid option {item!r}.\n' \
                   f'Close matches: {suggestions!r}.'
        else:
            return super().__str__()


class Dispatcher(dict):
    """
    Is basically a dict with a better error message on key error.
    >>> d = Dispatcher(abc=1, bcd=2)
    >>> d['acd']  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    paderbox.utils.mapping.DispatchError: Invalid option 'acd'.
    Close matches: ['bcd', 'abc'].

    Copied from paderbox to minimize depenencies.
    """

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError as e:
            raise DispatchError(item, self.keys()) from None


def validate_inputs(estimate, targets, segment_boundaries):
    num_estimates = estimate.shape[0]
    num_targets = len(targets)

    if num_estimates > 30:
        raise ValueError(f'Are you sure? num_estimates={num_estimates}')

    if len(segment_boundaries) != num_targets:
        raise ValueError(
            f'The number of segment_boundaries does not match the number of '
            f'targets! '
            f'num segment_boundaries: {len(segment_boundaries)} '
            f'num targets: {len(targets)}'
        )

    for idx, ((start, stop), target) in enumerate(zip(
            segment_boundaries, targets
    )):
        if target.shape[0] != stop - start:
            raise ValueError(
                f'Length mismatch between target and segment_boundaries at '
                f'target {idx}: '
                f'target shape: {target.shape} '
                f'segment_boundaries: {start, stop}'
            )
        if start < 0 or stop > estimate.shape[1]:
            raise ValueError(
                f'Length mismatch between estimation and targets / '
                f'segment_boundaries at {idx}: '
                f'estimation shape: {estimate.shape} '
                f'segment_boundaries: {start, stop}'
            )
