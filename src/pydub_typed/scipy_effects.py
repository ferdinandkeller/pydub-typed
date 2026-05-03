"""Scipy effects module.

This module provides scipy versions of high_pass_filter, and low_pass_filter
as well as an additional band_pass_filter.

Of course, you will need to install scipy for these to work.

When this module is imported the high and low pass filters from this module
will be used when calling audio_segment.high_pass_filter() and
audio_segment.high_pass_filter() instead of the slower, less powerful versions
provided by pydub.effects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from scipy.signal import butter, sosfilt  # pyright: ignore[reportMissingModuleSource, reportUnknownVariableType]

from .audio_segment import AudioSegment
from .effects import apply_mono_filter_to_each_channel
from .exceptions import PydubError


def _mk_butter_filter[T: AudioSegment](
    freq: float | list[float],
    filter_type: Literal['lowpass', 'highpass', 'band'],
    order: int,
) -> Callable[[T], T]:
    """Make butter filter.

    Args:
        freq: The cutoff frequency for highpass and lowpass filters. For
            band filters, a list of [low_cutoff, high_cutoff]
        filter_type: "lowpass", "highpass", or "band"
        order: nth order butterworth filter (default: 5th order). The
            attenuation is -6dB/octave beyond the cutoff frequency (for 1st
            order). A Higher order filter will have more attenuation, each level
            adding an additional -6dB (so a 3rd order butterworth filter would
            be -18dB/octave).

    Returns:
        function which can filter a mono audio segment
    """

    def filter_fn(seg: T) -> T:
        if seg.channels != 1:
            msg = 'The filter only works on mono audio.'
            PydubError(msg)

        nyq = 0.5 * seg.frame_rate
        freqs = [f / nyq for f in freq] if isinstance(freq, list) else freq / nyq

        sos = butter(order, freqs, btype=filter_type, output='sos')  # pyright: ignore[reportUnknownVariableType]
        y = sosfilt(sos, seg.get_array_of_samples())  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

        return seg._spawn(y.astype(seg.array_type))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportPrivateUsage] # noqa: SLF001

    return filter_fn


def band_pass_filter[T: AudioSegment](seg: T, low_cutoff_freq: float, high_cutoff_freq: float, order: int = 5) -> T:
    """Band pass filter."""
    filter_fn = _mk_butter_filter([low_cutoff_freq, high_cutoff_freq], 'band', order=order)
    return seg.pipe(apply_mono_filter_to_each_channel, filter_fn)


def high_pass_filter[T: AudioSegment](seg: T, cutoff_freq: float, order: int = 5) -> T:
    """High pass filter."""
    filter_fn = _mk_butter_filter(cutoff_freq, 'highpass', order=order)
    return seg.pipe(apply_mono_filter_to_each_channel, filter_fn)


def low_pass_filter[T: AudioSegment](seg: T, cutoff_freq: float, order: int = 5) -> T:
    """Low pass filter."""
    filter_fn = _mk_butter_filter(cutoff_freq, 'lowpass', order=order)
    return seg.pipe(apply_mono_filter_to_each_channel, filter_fn)


def eq[T: AudioSegment](
    seg: T,
    focus_freq: float,
    bandwidth: int = 100,
    filter_mode: Literal['peak', 'high_shelf', 'low_shelf'] = 'peak',
    gain_db: float = 0,
    order: int = 2,
) -> T:
    """Equalize.

    Args:
        seg:
            Audio segment.
        focus_freq:
            middle frequency or known frequency of band (in Hz)
        bandwidth:
            range of the equalizer band
        filter_mode:
            Mode of Equalization(Peak/Notch(Bell Curve),High Shelf, Low Shelf)
        gain_db:
            dB gain
        order:
            Rolloff factor(1 - 6dB/Octave 2 - 12dB/Octave)

    Returns:
        Equalized/Filtered AudioSegment
    """
    filt_mode = ['peak', 'low_shelf', 'high_shelf']
    if filter_mode not in filt_mode:
        msg = 'Incorrect Mode Selection'
        raise ValueError(msg)

    if gain_db >= 0:
        if filter_mode == 'peak':
            sec = band_pass_filter(seg, focus_freq - bandwidth / 2, focus_freq + bandwidth / 2, order=order)
            return seg.overlay(sec - (3 - gain_db))

        if filter_mode == 'low_shelf':
            sec = low_pass_filter(seg, focus_freq, order=order)
            return seg.overlay(sec - (3 - gain_db))

        if filter_mode == 'high_shelf':
            sec = high_pass_filter(seg, focus_freq, order=order)
            return seg.overlay(sec - (3 - gain_db))

    if gain_db < 0:
        if filter_mode == 'peak':
            sec = high_pass_filter(seg, focus_freq - bandwidth / 2, order=order)
            seg = seg.overlay(sec - (3 + gain_db)) + gain_db
            sec = low_pass_filter(seg, focus_freq + bandwidth / 2, order=order)
            return seg.overlay(sec - (3 + gain_db)) + gain_db

        if filter_mode == 'low_shelf':
            sec = high_pass_filter(seg, focus_freq, order=order)
            return seg.overlay(sec - (3 + gain_db)) + gain_db

        if filter_mode == 'high_shelf':
            sec = low_pass_filter(seg, focus_freq, order=order)
            return seg.overlay(sec - (3 + gain_db)) + gain_db

    msg = 'Unknown configuration.'
    raise PydubError(msg)
