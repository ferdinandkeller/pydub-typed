"""Various functions for finding/manipulating silence in AudioSegments."""

import itertools

from .audio_segment import AudioSegment
from .utils import db_to_power


def detect_silence(
    audio_segment: AudioSegment, *, min_silence_len: int = 1000, silence_thresh: float = -16, seek_step: int = 1
) -> list[tuple[int, int]]:
    """
    Returns a list of all silent sections [start, end] in milliseconds of audio_segment.

    Inverse of detect_nonsilent().
    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    seg_len = len(audio_segment)

    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    silence_thresh = db_to_power(silence_thresh) * audio_segment.max_possible_amplitude

    # find silence and add start and end indicies to the to_cut list
    silence_starts: list[int] = []

    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        audio_slice = audio_segment[i : i + min_silence_len]
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)

    # short circuit when there is no silence
    if not silence_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    silent_ranges: list[tuple[int, int]] = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = silence_start_i == prev_i + seek_step

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append((current_range_start, prev_i + min_silence_len))
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append((current_range_start, prev_i + min_silence_len))

    return silent_ranges


def detect_nonsilent(
    audio_segment: AudioSegment, *, min_silence_len: int = 1000, silence_thresh: float = -16, seek_step: int = 1
) -> list[tuple[int, int]]:
    """
    Returns a list of all nonsilent sections [start, end] in milliseconds of audio_segment.

    Inverse of detect_silent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    silent_ranges = detect_silence(
        audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=seek_step
    )
    len_seg = len(audio_segment)

    # if there is no silence, the whole thing is nonsilent
    if not silent_ranges:
        return [(0, len_seg)]

    # short circuit when the whole audio segment is silent
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges: list[tuple[int, int]] = []
    end_i = None
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append((prev_end_i, start_i))
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append((prev_end_i, len_seg))

    if nonsilent_ranges[0] == (0, 0):
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def _fix_overlapping_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fix overlapping ranges by splitting overlaps at the midpoint."""
    for i in range(len(ranges) - 1):
        last_range = ranges[i]
        next_range = ranges[i + 1]
        last_end = last_range[1]
        next_start = next_range[0]
        if next_start < last_end:
            mid = (last_end + next_start) // 2
            ranges[i] = (last_range[0], mid)
            ranges[i + 1] = (mid, next_range[1])
    return ranges


def split_on_silence[T: AudioSegment](
    audio_segment: T,
    *,
    min_silence_len: int = 1000,
    silence_thresh: float = -16,
    keep_silence: int = 100,
    seek_step: int = 1,
) -> list[T]:
    """
    Returns list of audio segments from splitting audio_segment on silent sections.

    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS

    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms

    seek_step - step size for interating over the segment in ms
    """
    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges: list[tuple[int, int]] = [
        (start - keep_silence, end + keep_silence)
        for (start, end) in detect_nonsilent(
            audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=seek_step
        )
    ]

    _fix_overlapping_ranges(output_ranges)

    return [audio_segment[max(start, 0) : min(end, len(audio_segment))] for start, end in output_ranges]


def detect_leading_silence(sound: AudioSegment, *, silence_threshold: float = -50.0, chunk_size: int = 10) -> int:
    """
    Returns the millisecond/index that the leading silence ends.

    audio_segment - the segment to find silence in
    silence_threshold - the upper bound for how quiet is silent in dFBS
    chunk_size - chunk size for interating over the segment in ms
    """
    trim_ms = 0  # ms

    # to avoid infinite loop
    if chunk_size <= 0:
        msg = 'Chunk Size should be strictly greater than zero.'
        raise ValueError(msg)

    while sound[trim_ms : trim_ms + chunk_size].dbfs < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    # if there is no end it should return the length of the segment
    return min(trim_ms, len(sound))
