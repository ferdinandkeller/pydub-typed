"""Utilities."""

import json
import os
import re
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from math import ceil, log10
from pathlib import Path
from tempfile import TemporaryFile
from typing import IO, Any, Literal
from warnings import warn

import audioop as audioop  # TODO: remove because deprecated  # noqa: TD002, TD003

from .audio_segment import AudioSegment
from .effects import invert_phase
from .exceptions import UnreachableCodeError

FRAME_WIDTHS: dict[int, int] = {
    8: 1,
    16: 2,
    32: 4,
}
ARRAY_TYPES: dict[int, str] = {
    8: 'b',
    16: 'h',
    32: 'i',
}
ARRAY_RANGES: dict[int, tuple[int, int]] = {
    8: (-0x80, 0x7F),
    16: (-0x8000, 0x7FFF),
    32: (-0x80000000, 0x7FFFFFFF),
}


RE_STREAM = re.compile(
    r'(?P<space_start> +)'
    r'Stream #0[:\.]'
    r'(?P<stream_id>[0-9]+)'
    r'(?P<content_0>.+)\n?'
    r'(?! *Stream)'
    r'((?P<space_end> +)(?P<content_1>.+))?'
)
RE_CODECS = re.compile(r'^([D.][E.][AVS.][I.][L.][S.]) (\w*) +(.*)')


def clamp[T](value: T | None, lower: T, upper: T) -> T | None:
    """Clamp the value."""
    if value is None:
        return None
    return max(lower, min(value, upper))  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportArgumentType]


def resolve_time_range(
    start: int | None,
    end: int | None,
    duration: int | None,
    max_duration: int,
) -> tuple[int, int, int]:
    """Resolve start, end & duration from any two of three."""
    if start is None:
        end = end or max_duration
        duration = duration or max_duration
        start = max(0, end - duration)
        return (start, end, end - start)

    if end is None:
        start = start or 0
        duration = duration or max_duration
        end = min(start + duration, max_duration)
        return (start, end, end - start)

    if duration is None:
        start = start or 0
        end = end or max_duration
        return (start, end, end - start)

    msg = 'At least one of start, end, or duration must be None.'
    raise ValueError(msg)


def get_frame_width(bit_depth: int) -> int:
    """Returns the frame width."""
    return FRAME_WIDTHS[bit_depth]


def get_array_type(bit_depth: int, *, signed: bool = True) -> str:
    """Returns the array type."""
    t = ARRAY_TYPES[bit_depth]
    if not signed:
        t = t.upper()
    return t


def get_min_max_value(bit_depth: int) -> tuple[int, int]:
    """Returns the minimum and maximum values."""
    return ARRAY_RANGES[bit_depth]


@contextmanager
def fd_or_path_or_tempfile(
    fd: str | Path | IO[bytes] | None, *, mode: str = 'w+b', tempfile: bool = True
) -> Generator[IO[bytes]]:
    """Get file descriptor or path or tempfile."""
    if fd is None and tempfile:
        with TemporaryFile(mode=mode) as temp_file:
            yield temp_file

    elif isinstance(fd, (str, Path)):
        with Path(fd).open(mode=mode) as file:
            yield file

    elif isinstance(fd, IO):
        yield fd

    else:
        raise UnreachableCodeError


def db_to_power(db: float, *, using_amplitude: bool = True) -> float:
    """Converts the input db to a power."""
    db = float(db)

    if using_amplitude:
        return 10 ** (db / 20)
    # using power
    return 10 ** (db / 10)


def power_to_db(ratio: float, *, using_amplitude: bool = True) -> float:
    """Converts the input power to db."""
    ratio = float(ratio)

    # special case for multiply-by-zero (convert to silence)
    if ratio == 0:
        return -float('inf')

    if using_amplitude:
        return 20 * log10(ratio)
    # using power
    return 10 * log10(ratio)


def make_chunks[T: AudioSegment](audio_segment: T, chunk_length: int) -> list[T]:
    """Breaks an AudioSegment into chunks that are <chunk_length> milliseconds long. The last segment can be shorter."""
    number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
    return [audio_segment[i * chunk_length : (i + 1) * chunk_length] for i in range(int(number_of_chunks))]


def which(program: str) -> Path:
    """Mimics behavior of UNIX which command."""
    # Add .exe program extension for windows support
    if os.name == 'nt' and not program.endswith('.exe'):
        program += '.exe'

    envdir_list = [Path.cwd(), *os.environ['PATH'].split(os.pathsep)]

    for envdir in envdir_list:
        program_path = envdir / Path(program)
        if Path(program_path).is_file() and os.access(program_path, os.X_OK):
            return program_path

    raise UnreachableCodeError


def get_encoder_name() -> Literal['avconv', 'ffmpeg']:
    """Return enconder default application for system, either avconv or ffmpeg."""
    if which('avconv'):
        return 'avconv'
    if which('ffmpeg'):
        return 'ffmpeg'
    # should raise exception
    warn(
        "Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work",
        RuntimeWarning,
        stacklevel=2,
    )
    return 'ffmpeg'


def get_player_name() -> Literal['avplay', 'ffplay']:
    """Return enconder default application for system, either avplay or ffplay."""
    if which('avplay'):
        return 'avplay'
    if which('ffplay'):
        return 'ffplay'
    # should raise exception
    warn(
        "Couldn't find ffplay or avplay - defaulting to ffplay, but may not work",
        RuntimeWarning,
        stacklevel=2,
    )
    return 'ffplay'


def get_prober_name() -> Literal['avprobe', 'ffprobe']:
    """Return probe application, either avprobe or ffprobe."""
    if which('avprobe'):
        return 'avprobe'
    if which('ffprobe'):
        return 'ffprobe'
    # should raise exception
    warn(
        "Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work",
        RuntimeWarning,
        stacklevel=2,
    )
    return 'ffprobe'


def get_extra_info(stderr: str) -> dict[int, list[str]]:
    """
    Avprobe sometimes gives more information on stderr than on the json output.

    The information has to be extracted
    from stderr of the format of:
    '    Stream #0:0: Audio: flac, 88200 Hz, stereo, s32 (24 bit)'
    or (macOS version):
    '    Stream #0:0: Audio: vorbis'
    '      44100 Hz, stereo, fltp, 320 kb/s'
    """
    extra_info: dict[int, list[str]] = {}
    for match in re.finditer(RE_STREAM, stderr):
        if match.group('space_end') is not None and len(match.group('space_start')) <= len(match.group('space_end')):
            content_line = ','.join([match.group('content_0'), match.group('content_1')])
        else:
            content_line = match.group('content_0')
        tokens = [x.strip() for x in re.split('[:,]', content_line) if x]
        extra_info[int(match.group('stream_id'))] = tokens
    return extra_info


def mediainfo(filepath: str | Path | IO[bytes] | None, *, read_ahead_limit: int = -1) -> dict[str, Any] | None:  # noqa: C901
    """Return json dictionary with media info (codec, duration, size, bitrate...) from filepath."""
    prober = get_prober_name()
    command_args = ['-v', 'info', '-show_format', '-show_streams']

    if isinstance(filepath, (str, Path)):
        command_args += [str(filepath)]
        stdin_data = None
    else:
        if prober == 'ffprobe':
            command_args += ['-read_ahead_limit', str(read_ahead_limit), 'cache:pipe:0']
        else:
            command_args += ['-']
        with fd_or_path_or_tempfile(filepath, mode='rb', tempfile=False) as file:
            file.seek(0)
            stdin_data = file.read()

    command = [prober, '-of', 'json', *command_args]
    res = subprocess.run(  # noqa: S603
        command,
        input=stdin_data,
        capture_output=True,
        text=True,
        errors='ignore',
        check=False,
    )

    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError:
        # If ffprobe didn't give any information, just return it
        # (for example, because the file doesn't exist)
        return None
    if not info:
        return None

    extra_info = get_extra_info(res.stderr)

    audio_streams = [x for x in info['streams'] if x['codec_type'] == 'audio']
    if len(audio_streams) == 0:
        return info

    # we just operate on the first audio stream in case there are more
    stream = audio_streams[0]

    def set_property(prop: str, value: str | int) -> None:
        if prop not in stream or stream[prop] == 0:
            stream[prop] = value

    for token in extra_info[stream['index']]:
        m = re.match(r'([su]([0-9]{1,2})p?) \(([0-9]{1,2}) bit\)$', token)
        m2 = re.match(r'([su]([0-9]{1,2})p?)( \(default\))?$', token)
        if m:
            set_property('sample_fmt', m.group(1))
            set_property('bits_per_sample', int(m.group(2)))
            set_property('bits_per_raw_sample', int(m.group(3)))
        elif m2:
            set_property('sample_fmt', m2.group(1))
            set_property('bits_per_sample', int(m2.group(2)))
            set_property('bits_per_raw_sample', int(m2.group(2)))
        elif re.match(r'(flt)p?( \(default\))?$', token):
            set_property('sample_fmt', token)
            set_property('bits_per_sample', 32)
            set_property('bits_per_raw_sample', 32)
        elif re.match(r'(dbl)p?( \(default\))?$', token):
            set_property('sample_fmt', token)
            set_property('bits_per_sample', 64)
            set_property('bits_per_raw_sample', 64)

    return info


@dataclass
class SupportedCodecs:
    """List of supported codecs."""

    encoders: set[str]
    decoders: set[str]


@cache
def get_supported_codecs() -> SupportedCodecs:
    """Returns supported codecs."""
    # retrieve codecs from current encoder
    res = subprocess.run(  # noqa: S603
        [get_encoder_name(), '-codecs'],
        capture_output=True,
        text=True,
        check=False,
    )

    # if failed, return empty list
    if res.returncode != 0:
        return SupportedCodecs(encoders=set(), decoders=set())

    output = res.stdout

    if sys.platform == 'win32':
        output = output.replace('\r', '')

    encoders: set[str] = set()
    decoders: set[str] = set()
    for line in output.split('\n'):
        match = RE_CODECS.match(line.strip())
        if not match:
            continue
        flags, codec, _ = match.groups()
        if flags[0] == 'D':
            decoders.add(codec)
        if flags[1] == 'E':
            encoders.add(codec)
    return SupportedCodecs(encoders=encoders, decoders=decoders)


def get_supported_encoders() -> set[str]:
    """Returns supported encoders."""
    return get_supported_codecs().encoders


def get_supported_decoders() -> set[str]:
    """Returns supported decoders."""
    return get_supported_codecs().decoders


def stereo_to_ms(audio_segment: AudioSegment) -> AudioSegment:
    """Left-Right -> Mid-Side."""
    channels = audio_segment.split_to_mono()
    channels = [channels[0].overlay(channels[1]), channels[0].overlay(channels[1].pipe(invert_phase))]
    return AudioSegment.from_mono_audiosegments(channels[0], channels[1])


def ms_to_stereo(audio_segment: AudioSegment) -> AudioSegment:
    """Mid-Side -> Left-Right."""
    channel = audio_segment.split_to_mono()
    channel = [channel[0].overlay(channel[1]) - 3, channel[0].overlay(channel[1].pipe(invert_phase)) - 3]
    return AudioSegment.from_mono_audiosegments(channel[0], channel[1])
