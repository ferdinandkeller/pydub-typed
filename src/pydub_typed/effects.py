"""Effects library."""

import array
import math
from collections.abc import Callable

from .audio_segment import AudioSegment
from .exceptions import InvalidDurationError, PydubError
from .silence import split_on_silence
from .utils import audioop, db_to_power, get_min_max_value, make_chunks, power_to_db


def apply_mono_filter_to_each_channel[T: AudioSegment](seg: T, filter_fn: Callable[[T], T]) -> T:
    """Apply a mono filter to each channel."""
    n_channels = seg.channels

    channel_segs = seg.split_to_mono()
    channel_segs = [filter_fn(channel_seg) for channel_seg in channel_segs]

    out_data = seg.get_array_of_samples()
    for channel_i, channel_seg in enumerate(channel_segs):
        for sample_i, sample in enumerate(channel_seg.get_array_of_samples()):
            index = (sample_i * n_channels) + channel_i
            out_data[index] = sample

    return seg._spawn(out_data.tobytes())  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def normalize[T: AudioSegment](seg: T, *, headroom: float = 0.1) -> T:
    """Headroom is how close to the maximum volume to boost the signal up to (specified in dB)."""
    peak_sample_val = seg.max

    # if the max is 0, this audio segment is silent, and can't be normalized
    if peak_sample_val == 0:
        return seg

    target_peak = seg.max_possible_amplitude * db_to_power(-headroom)

    needed_boost = power_to_db(target_peak / peak_sample_val)
    return seg.apply_gain(volume_change=needed_boost)


def speedup[T: AudioSegment](seg: T, *, playback_speed: float = 1.5, chunk_size: int = 150, crossfade: int = 25) -> T:
    """Speedup."""
    # we will keep audio in 150ms chunks since one waveform at 20Hz is 50ms long
    # (20 Hz is the lowest frequency audible to humans)

    # portion of AUDIO TO KEEP. if playback speed is 1.25 we keep 80% (0.8) and
    # discard 20% (0.2)
    atk = 1.0 / playback_speed

    if playback_speed < 2.0:  # noqa: PLR2004
        # throwing out more than half the audio - keep 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size * (1 - atk) / atk)
    else:
        # throwing out less than half the audio - throw out 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size)
        chunk_size = int(atk * chunk_size / (1 - atk))

    # the crossfade cannot be longer than the amount of audio we're removing
    crossfade = min(crossfade, ms_to_remove_per_chunk - 1)

    chunks = make_chunks(seg, chunk_size + ms_to_remove_per_chunk)
    if len(chunks) < 2:  # noqa: PLR2004
        msg = (
            f'Could not speed up AudioSegment, it was too short {seg.duration_seconds:0.2f}s '
            'for the current settings:\n{chunk_size}ms chunks at {playback_speed:0.1f}x speedup'
        )
        raise PydubError(msg)

    # we'll actually truncate a bit less than we calculated to make up for the
    # crossfade between chunks
    ms_to_remove_per_chunk -= crossfade

    # we don't want to truncate the last chunk since it is not guaranteed to be
    # the full chunk length
    last_chunk = chunks[-1]
    chunks = [chunk[:-ms_to_remove_per_chunk] for chunk in chunks[:-1]]

    out = chunks[0]
    for chunk in chunks[1:]:
        out = out.append(chunk, crossfade=crossfade)

    out += last_chunk
    return out


def strip_silence[T: AudioSegment](
    seg: T, *, silence_len: int = 1000, silence_thresh: float = -16, padding: int = 100
) -> T:
    """Strip silence."""
    if padding > silence_len:
        msg = 'padding cannot be longer than silence_len'
        raise InvalidDurationError(msg)

    chunks = split_on_silence(seg, min_silence_len=silence_len, silence_thresh=silence_thresh, keep_silence=padding)
    crossfade = padding // 2

    if not len(chunks):
        return seg[0:0]

    seg = chunks[0]
    for chunk in chunks[1:]:
        seg = seg.append(chunk, crossfade=crossfade)

    return seg


def compress_dynamic_range[T: AudioSegment](
    seg: T, *, threshold: float = -20.0, ratio: float = 4.0, attack: float = 5.0, release: float = 50.0
) -> T:
    """Compress dynamic range.

    Args:
        seg:
            An audio segment.

        threshold:
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio:
            Compression ratio. Audio louder than the threshold will be
            reduced to 1/ratio the volume. A ratio of 4.0 is equivalent to
            a setting of 4:1 in a pro-audio compressor like the Waves C1.

        attack:
            Attack in milliseconds. How long it should take for the compressor
            to kick in once the audio has exceeded the threshold.

        release:
            Release in milliseconds. How long it should take for the compressor
            to stop compressing after the audio has falled below the threshold.


    For an overview of Dynamic Range Compression, and more detailed explanation
    of the related terminology, see:

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """
    thresh_rms = seg.max_possible_amplitude * db_to_power(threshold)

    look_frames = int(seg.frame_count(ms=attack))

    def rms_at(frame_i: int) -> int:
        return seg.get_sample_slice(start_sample=frame_i - look_frames, end_sample=frame_i).rms

    def db_over_threshold(rms: int) -> float:
        if rms == 0:
            return 0.0
        db = power_to_db(rms / thresh_rms)
        return max(db, 0)

    output: list[bytes] = []

    # amount to reduce the volume of the audio by (in dB)
    attenuation = 0.0

    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in range(int(seg.frame_count())):
        rms_now = rms_at(i)

        # with a ratio of 4.0 this means the volume will exceed the threshold by
        # 1/4 the amount (of dB) that it would otherwise
        max_attenuation = (1 - (1.0 / ratio)) * db_over_threshold(rms_now)

        attenuation_inc = max_attenuation / attack_frames
        attenuation_dec = max_attenuation / release_frames

        if rms_now > thresh_rms and attenuation <= max_attenuation:
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else:
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)

        frame = seg.get_frame(i)
        if attenuation != 0.0:
            frame = audioop.mul(frame, seg.sample_width, db_to_power(-attenuation))

        output.append(frame)

    return seg._spawn(data=b''.join(output))  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def invert_phase(seg: AudioSegment, *, channels: tuple[int, int] = (1, 1)) -> AudioSegment:
    """Channels- specifies which channel (left or right) to reverse the phase of.

    Note that mono AudioSegments will become stereo.
    """
    if channels == (1, 1):
        inverted = audioop.mul(seg.raw_data, seg.sample_width, -1.0)
        return seg._spawn(data=inverted)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    if seg.channels == 2:  # noqa: PLR2004
        left, right = seg.split_to_mono()
    else:
        msg = "Can't implicitly convert an AudioSegment with " + str(seg.channels) + ' channels to stereo.'
        raise PydubError(msg)

    if channels == (1, 0):
        left = left.pipe(invert_phase)
    else:
        right = right.pipe(invert_phase)

    return seg.from_mono_audiosegments(left, right)


# High and low pass filters based on implementation found on Stack Overflow:
#   http://stackoverflow.com/questions/13882038/implementing-simple-high-and-low-pass-filters-in-c


def low_pass_filter[T: AudioSegment](seg: T, *, cutoff: float) -> T:
    """Cutoff.

    Frequency (in Hz) where higher frequency signal will begin to be reduced
    by 6dB per octave (doubling in frequency) above this point.
    """
    rc = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = dt / (rc + dt)

    original = seg.get_array_of_samples()
    filtered_array = array.array(seg.array_type, original)

    frame_count = int(seg.frame_count())

    last_val: list[float] = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filtered_array[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            last_val[j] = last_val[j] + (alpha * (original[offset] - last_val[j]))
            filtered_array[offset] = int(last_val[j])

    return seg._spawn(data=filtered_array.tobytes())  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def high_pass_filter[T: AudioSegment](seg: T, *, cutoff: float) -> T:
    """Cutoff.

    Frequency (in Hz) where lower frequency signal will begin to
    be reduced by 6dB per octave (doubling in frequency) below this point
    """
    rc = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = rc / (rc + dt)

    minval, maxval = get_min_max_value(seg.sample_width * 8)

    original = seg.get_array_of_samples()
    filtered_array = array.array(seg.array_type, original)

    frame_count = int(seg.frame_count())

    last_val: list[float] = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filtered_array[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            offset_minus_1 = ((i - 1) * seg.channels) + j

            last_val[j] = alpha * (last_val[j] + original[offset] - original[offset_minus_1])
            filtered_array[offset] = int(min(max(last_val[j], minval), maxval))

    return seg._spawn(data=filtered_array.tobytes())  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def pan[T: AudioSegment](seg: T, *, pan_amount: float) -> T:
    """Pan.

    pan_amount should be between -1.0 (100% left) and +1.0 (100% right)

    When pan_amount == 0.0 the left/right balance is not changed.

    Panning does not alter the *perceived* loundness, but since loudness
    is decreasing on one side, the other side needs to get louder to
    compensate. When panned hard left, the left channel will be 3dB louder.
    """
    if not -1.0 <= pan_amount <= 1.0:
        msg = r'pan_amount should be between -1.0 (100% left) and +1.0 (100% right)'
        raise ValueError(msg)

    max_boost_db = power_to_db(2.0)
    boost_db = abs(pan_amount) * max_boost_db

    boost_factor = db_to_power(boost_db)
    reduce_factor = db_to_power(max_boost_db) - boost_factor

    reduce_db = power_to_db(reduce_factor)

    # Cut boost in half (max boost== 3dB) - in reality 2 speakers
    #   do not sum to a full 6 dB.
    boost_db = boost_db / 2.0

    if pan_amount < 0:
        return seg.pipe(apply_gain_stereo, left_gain=boost_db, right_gain=reduce_db)
    return seg.pipe(apply_gain_stereo, left_gain=reduce_db, right_gain=boost_db)


def apply_gain_stereo[T: AudioSegment](seg: T, *, left_gain: float = 0.0, right_gain: float = 0.0) -> T:
    """Apply gain stereo.

    left_gain - amount of gain to apply to the left channel (in dB)
    right_gain - amount of gain to apply to the right channel (in dB)

    note: mono audio segments will be converted to stereo
    """
    if seg.channels == 1:
        left = right = seg
    elif seg.channels == 2:  # noqa: PLR2004
        left, right = seg.split_to_mono()
    else:
        msg = 'Expects 1 or 2 channels.'
        raise PydubError(msg)

    l_mult_factor = db_to_power(left_gain)
    r_mult_factor = db_to_power(right_gain)

    left_data = audioop.mul(left.raw_data, left.sample_width, l_mult_factor)
    left_data = audioop.tostereo(left_data, left.sample_width, 1, 0)

    right_data = audioop.mul(right.raw_data, right.sample_width, r_mult_factor)
    right_data = audioop.tostereo(right_data, right.sample_width, 0, 1)

    output = audioop.add(left_data, right_data, seg.sample_width)

    return seg._spawn(data=output, overrides={'channels': 2, 'frame_width': 2 * seg.sample_width})  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
