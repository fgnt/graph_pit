import itertools
from dataclasses import dataclass
from typing import List, Tuple


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
        >>> segment_boundaries = [(0, 10), (11, 14), (12, 17), (16, 20), (21, 30)]
        >>> sorted(get_overlaps_from_segment_boundaries(segment_boundaries))
        [(1, 2), (2, 3)]
    """
    segments = Segment.segments_from_segment_boundaries(segment_boundaries)

    overlaps = set()
    for r1, r2 in itertools.combinations(segments, 2):
        if Segment.overlaps(r1, r2):
            overlaps.add(tuple(sorted((r1.index, r2.index))))

    return list(overlaps)
