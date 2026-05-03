"""Audio Segment Module."""

import array
import base64
import struct
import subprocess
import sys
import wave
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Concatenate, Literal, NamedTuple, NotRequired, Self, TypedDict

from .exceptions import (
    CouldntDecodeError,
    CouldntEncodeError,
    InvalidDurationError,
    InvalidID3TagVersionError,
    InvalidParametersError,
    MissingAudioParameterError,
    TooManyMissingFramesError,
)
from .logging_utils import log_conversion, log_subprocess_output
from .utils import (
    audioop,
    clamp,
    db_to_power,
    fd_or_path_or_tempfile,
    get_array_type,
    get_encoder_name,
    mediainfo,
    power_to_db,
    resolve_time_range,
)


def _apply_time_slice[T: AudioSegment](seg: T, *, start_second: float | None, duration: float | None) -> T:
    """Apply time slice."""
    start_ms = int(start_second * 1000) if start_second is not None else 0
    end_ms = int((start_second or 0) * 1000 + duration * 1000) if duration is not None else None
    return seg[start_ms:end_ms]


AUDIO_FILE_EXT_ALIASES: dict[str, str] = {
    'm4a': 'mp4',
    'wave': 'wav',
}


class AudioMetadata(TypedDict):
    """Audio Metadata dict."""

    channels: NotRequired[int | None]
    frame_rate: NotRequired[int | None]
    sample_width: NotRequired[int | None]
    frame_width: NotRequired[int | None]


class WavSubChunk(NamedTuple):
    """Wav sub chunk."""

    id: bytes
    position: int
    size: int


class WavData(NamedTuple):
    """Wav data."""

    audio_format: bytes
    raw_data: bytes
    channels: int
    sample_rate: int
    bits_per_sample: int


def extract_wav_headers(data: bytearray) -> list[WavSubChunk]:
    """Extract wav headers."""
    pos = 12  # The size of the RIFF chunk descriptor
    subchunks: list[WavSubChunk] = []
    while pos + 8 <= len(data) and len(subchunks) < 10:  # noqa: PLR2004
        subchunk_id = bytes(data[pos : pos + 4])
        subchunk_size = struct.unpack_from('<I', data[pos + 4 : pos + 8])[0]
        subchunks.append(WavSubChunk(subchunk_id, pos, subchunk_size))
        if subchunk_id == b'data':
            # 'data' is the last subchunk
            break
        pos += subchunk_size + 8
    return subchunks


def read_wav_audio(data: bytearray, *, headers: list[WavSubChunk] | None = None) -> WavData:
    """Read wav audio."""
    if not headers:
        headers = extract_wav_headers(data)

    fmt = [x for x in headers if x.id == b'fmt ']
    if not fmt or fmt[0].size < 16:  # noqa: PLR2004
        msg = "Couldn't find fmt header in wav data"
        raise CouldntDecodeError(msg)
    fmt = fmt[0]
    pos = fmt.position + 8
    audio_format = struct.unpack_from('<H', data[pos : pos + 2])[0]
    if audio_format not in {1, 0xFFFE}:
        msg = f'Unknown audio format 0x{audio_format:X} in wav data'
        raise CouldntDecodeError(msg)

    channels = struct.unpack_from('<H', data[pos + 2 : pos + 4])[0]
    sample_rate = struct.unpack_from('<I', data[pos + 4 : pos + 8])[0]
    bits_per_sample = struct.unpack_from('<H', data[pos + 14 : pos + 16])[0]

    data_hdr = headers[-1]
    if data_hdr.id != b'data':
        msg = "Couldn't find data header in wav data"
        raise CouldntDecodeError(msg)

    pos = data_hdr.position + 8
    return WavData(
        audio_format,
        bytes(data[pos : pos + data_hdr.size]),
        channels,
        sample_rate,
        bits_per_sample,
    )


def fix_wav_headers(data: bytearray) -> None:
    """Fix wav headers."""
    headers = extract_wav_headers(data)
    if not headers or headers[-1].id != b'data':
        return

    # TODO: Handle huge files in some other way  # noqa: TD002, TD003
    if len(data) > 2**32:
        msg = 'Unable to process >4GB files'
        raise CouldntDecodeError(msg)

    # Set the file size in the RIFF chunk descriptor
    data[4:8] = struct.pack('<I', len(data) - 8)

    # Set the data size in the data subchunk
    pos = headers[-1].position
    data[pos + 4 : pos + 8] = struct.pack('<I', len(data) - pos - 8)


DEFAULT_CODECS: dict[str, str] = {'ogg': 'libvorbis'}


class AudioSegment:
    """
    AudioSegments are *immutable* objects representing segments of audio that can be manipulated using python code.

    AudioSegments are slicable using milliseconds.
    for example:
        a = AudioSegment.from_mp3(mp3file)
        first_second = a[:1000] # get the first second of an mp3
        slice = a[5000:10000] # get a slice from 5 to 10 seconds of an mp3
    """

    converter = get_encoder_name()  # either ffmpeg or avconv

    sample_width: int
    frame_rate: int
    channels: int
    frame_width: int
    _data: bytes

    def __init__(
        self,
        data: bytes,
        *,
        sample_width: int | None = None,
        frame_rate: int | None = None,
        channels: int | None = None,
        metadata: AudioMetadata | None = None,
    ) -> None:
        """Initialize Audio segment."""
        sample_width_tmp = sample_width
        frame_rate_tmp = frame_rate
        channels_tmp = channels

        audio_params = (sample_width_tmp, frame_rate_tmp, channels_tmp)

        # prevent partial specification of arguments
        if any(audio_params) and None in audio_params:
            msg = 'Either all audio parameters or no parameter must be specified.'
            raise MissingAudioParameterError(msg)

        # all arguments are given
        if sample_width_tmp is not None:
            if len(data) % (self.sample_width * self.channels) != 0:
                msg = "data length must be a multiple of '(sample_width * channels)'"
                raise ValueError(msg)

            self.frame_width = self.channels * self.sample_width
            self._data = data

        # keep support for 'metadata' until audio params are used everywhere
        elif metadata is not None:
            # internal use only
            self._data = data
            for attr, val in metadata.items():
                setattr(self, attr, val)
        else:
            # normal construction
            wav_data = read_wav_audio(bytearray(data))
            if not wav_data:
                msg = "Couldn't read wav audio from data"
                raise CouldntDecodeError(msg)

            self.channels = wav_data.channels
            self.sample_width = wav_data.bits_per_sample // 8
            self.frame_rate = wav_data.sample_rate
            self.frame_width = self.channels * self.sample_width
            self._data = wav_data.raw_data
            if self.sample_width == 1:
                # convert from unsigned integers in wav
                self._data = audioop.bias(self._data, 1, -128)

        # Convert 24-bit audio to 32-bit audio.
        # (stdlib audioop and array modules do not support 24-bit data)
        if self.sample_width == 3:  # noqa: PLR2004
            byte_buffer = BytesIO()

            # This conversion maintains the 24 bit values.  The values are
            # not scaled up to the 32 bit range.  Other conversions could be
            # implemented.
            i = iter(self._data)
            padding = {False: b'\x00', True: b'\xff'}
            for b0, b1, b2 in zip(i, i, i, strict=True):
                byte_buffer.write(padding[b2 > b'\x7f'[0]])
                old_bytes = struct.pack('BBB', b0, b1, b2)
                byte_buffer.write(old_bytes)

            self._data = byte_buffer.getvalue()
            self.sample_width = 4
            self.frame_width = self.channels * self.sample_width

    @property
    def raw_data(self) -> bytes:
        """Public access to the raw audio data as a bytestring."""
        return self._data

    def get_array_of_samples(self, *, array_type_override: str | None = None) -> array.array[int]:
        """Return the raw data as an array of samples."""
        return array.array(array_type_override or self.array_type, self.raw_data)

    @property
    def array_type(self) -> str:
        """Return the array type code for this segment's sample width."""
        return get_array_type(self.sample_width * 8)

    def __len__(self) -> int:
        """Return the length of this audio segment in milliseconds."""
        return round(1000 * (self.frame_count() / self.frame_rate))

    def __eq__(self, other: object) -> bool:
        """Compare equality of two audio segments."""
        if not isinstance(other, AudioSegment):
            raise NotImplementedError
        return self.raw_data == other.raw_data

    def __hash__(self) -> int:
        """Compute hash of audio segment."""
        return hash(AudioSegment) ^ hash((self.channels, self.frame_rate, self.sample_width, self._data))

    def __ne__(self, other: object) -> bool:
        """Compare inequality of two audio segments."""
        return not (self == other)

    def __iter__(self) -> Iterator[Self]:
        """Iterate over each millisecond."""
        return (self[i] for i in range(len(self)))

    def iter_chunks(self, chunk_ms: int) -> Generator[Self]:
        """Yield chunks of the given duration in milliseconds."""
        for i in range(0, len(self), chunk_ms):
            yield self[i : i + chunk_ms]

    def __getitem__(self, millisecond: int | slice) -> Self:
        """Get a slice of the audio segment at given millisecond."""
        if isinstance(millisecond, slice):
            if millisecond.step:
                msg = 'Slicing with step is not supported. Use iter_chunks() instead.'
                raise TypeError(msg)

            start = millisecond.start if millisecond.start is not None else 0
            end = millisecond.stop if millisecond.stop is not None else len(self)
            start = min(start, len(self))
            end = min(end, len(self))
        else:
            start = millisecond
            end = millisecond + 1

        start = self._parse_position(start) * self.frame_width
        end = self._parse_position(end) * self.frame_width
        data = self._data[start:end]

        # ensure the output is as long as the requester is expecting
        expected_length = end - start
        missing_frames = (expected_length - len(data)) // self.frame_width
        if missing_frames:
            if missing_frames > self.frame_count(ms=2):
                msg = (
                    'You should never be filling in '
                    '   more than 2 ms with silence here, '
                    f'missing frames: {missing_frames}'
                )
                raise TooManyMissingFramesError(msg)
            silence = audioop.mul(data[: self.frame_width], self.sample_width, 0)
            data += silence * missing_frames

        return self._spawn(data)

    def get_sample_slice(self, *, start_sample: int | None = None, end_sample: int | None = None) -> Self:
        """Get a section of the audio segment by sample index.

        NOTE: Negative indices do *not* address samples backword
        from the end of the audio segment like a python list.
        This is intentional.
        """
        max_val = int(self.frame_count())

        def bounded(val: int | None, default: int) -> int:
            if val is None:
                return default
            if val < 0:
                return 0
            if val > max_val:
                return max_val
            return val

        start_i = bounded(start_sample, 0) * self.frame_width
        end_i = bounded(end_sample, max_val) * self.frame_width

        data = self._data[start_i:end_i]
        return self._spawn(data)

    def __add__(self, arg: 'AudioSegment | float') -> Self:
        """Add to audio segment."""
        if isinstance(arg, AudioSegment):
            return self.append(arg, crossfade=0)
        return self.apply_gain(volume_change=arg)

    def __radd__(self, rarg: float) -> Self:
        """Permit use of sum() builtin with an iterable of AudioSegments."""
        if rarg == 0:
            return self
        msg = 'Gains must be the second addend after the AudioSegment'
        raise TypeError(msg)

    def __sub__(self, arg: 'AudioSegment | float') -> Self:
        """Sub from audio segment."""
        if isinstance(arg, AudioSegment):
            msg = "AudioSegment objects can't be subtracted from each other"
            raise TypeError(msg)
        return self.apply_gain(volume_change=-arg)

    def __mul__(self, arg: 'AudioSegment | int') -> Self:
        """If the argument is an AudioSegment, overlay the multiplied audio segment.

        If it's a number, just use the string multiply operation to repeat the audio.
        The following would return an AudioSegment that contains the audio of audio_seg eight times
        `audio_seg * 8`
        """
        if isinstance(arg, AudioSegment):
            return self.overlay(arg, position=0, loop=True)
        return self._spawn(data=self.raw_data * arg)

    def _spawn(self, data: bytes, *, overrides: AudioMetadata | None = None) -> Self:
        """Creates a new audio segment using the metadata from the current one and the data passed in.

        Should be used whenever an AudioSegment is being returned by an operation that
        would alters the current one, since AudioSegment objects are immutable.
        """
        metadata: AudioMetadata = {
            'sample_width': self.sample_width,
            'frame_rate': self.frame_rate,
            'channels': self.channels,
            'frame_width': self.frame_width,
        }
        if overrides:
            metadata.update(overrides)
        return self.__class__(
            data=data,
            metadata=metadata,
        )

    @classmethod
    def _sync(cls, *segs: 'AudioSegment') -> tuple['AudioSegment', ...]:
        """Synchronize audio segments settings."""
        channels = max(seg.channels for seg in segs)
        frame_rate = max(seg.frame_rate for seg in segs)
        sample_width = max(seg.sample_width for seg in segs)
        return tuple(
            seg.set_channels(channels).set_frame_rate(frame_rate).set_sample_width(sample_width) for seg in segs
        )

    def _parse_position(self, val: float) -> int:
        """Parse position."""
        if val < 0:
            val = len(self) - abs(val)
        val = self.frame_count(ms=len(self)) if val == float('inf') else self.frame_count(ms=val)
        return int(val)

    @classmethod
    def empty(cls) -> 'AudioSegment':
        """Create an empty audio Segment."""
        return cls(
            b'',
            metadata={
                'sample_width': 1,
                'frame_rate': 1,
                'channels': 1,
                'frame_width': 1,
            },
        )

    @classmethod
    def silent(cls, *, duration: int = 1000, frame_rate: int = 11025) -> 'AudioSegment':
        """
        Generate a silent audio segment.

        Duration specified in milliseconds (default duration: 1000ms, default frame_rate: 11025).
        """
        frames = int(frame_rate * (duration / 1000.0))
        data = b'\0\0' * frames
        return cls(
            data,
            metadata={
                'channels': 1,
                'sample_width': 2,
                'frame_rate': frame_rate,
                'frame_width': 2,
            },
        )

    @classmethod
    def from_mono_audiosegments(cls, *mono_segments: 'AudioSegment') -> 'AudioSegment':
        """Combine mono audio segments into a single audio segment."""
        if not mono_segments:
            msg = 'At least one AudioSegment instance is required'
            raise ValueError(msg)

        segs = cls._sync(*mono_segments)

        if segs[0].channels != 1:
            msg = 'AudioSegment.from_mono_audiosegments requires all arguments are mono AudioSegment instances'
            raise ValueError(msg)

        channels = len(segs)
        sample_width = segs[0].sample_width
        frame_rate = segs[0].frame_rate

        frame_count = max(int(seg.frame_count()) for seg in segs)
        data = array.array(segs[0].array_type, b'\0' * (frame_count * sample_width * channels))

        for i, seg in enumerate(segs):
            data[i::channels] = seg.get_array_of_samples()

        return cls(
            data.tobytes(),
            channels=channels,
            sample_width=sample_width,
            frame_rate=frame_rate,
        )

    @classmethod
    def from_file(  # noqa: C901, PLR0911, PLR0912, PLR0915
        cls,
        file: str | Path | IO[bytes],
        file_format: str | None = None,
        codec: str | None = None,
        parameters: list[str] | None = None,
        start_second: float | None = None,
        duration: float | None = None,
        sample_width: int | None = None,
        frame_rate: int | None = None,
        channels: int | None = None,
        read_ahead_limit: int = -1,
    ) -> 'AudioSegment':
        """Create an AudioSegment from an audio file."""
        orig_file = file

        filename = str(file) if isinstance(file, (str, Path)) else None

        if file_format:
            file_format = file_format.lower()
            file_format = AUDIO_FILE_EXT_ALIASES.get(file_format, file_format)

        def is_format(f: str) -> bool:
            f = f.lower()
            if file_format == f:
                return True
            if filename:
                return filename.lower().endswith(f'.{f}')
            return False

        with fd_or_path_or_tempfile(file, mode='rb', tempfile=False) as f:
            if is_format('wav'):
                try:
                    if start_second is None:
                        if duration is None:
                            return cls._from_safe_wav(f)
                        return cls._from_safe_wav(f)[: duration * 1000]
                    if duration is None:
                        return cls._from_safe_wav(f)[start_second * 1000 :]
                    return cls._from_safe_wav(f)[start_second * 1000 : (start_second + duration) * 1000]
                except Exception:  # noqa: BLE001
                    f.seek(0)
            elif is_format('raw') or is_format('pcm'):
                metadata: AudioMetadata = {
                    'sample_width': sample_width,
                    'frame_rate': frame_rate,
                    'channels': channels,
                    'frame_width': channels * sample_width
                    if channels is not None and sample_width is not None
                    else None,
                }
                if start_second is None:
                    if duration is None:
                        return cls(data=f.read(), metadata=metadata)
                    return cls(data=f.read(), metadata=metadata)[: duration * 1000]
                if duration is None:
                    return cls(data=f.read(), metadata=metadata)[start_second * 1000 :]
                return cls(data=f.read(), metadata=metadata)[start_second * 1000 : (start_second + duration) * 1000]

            conversion_command = [
                cls.converter,
                '-y',  # always overwrite existing files
            ]

            # If format is not defined
            # ffmpeg/avconv will detect it automatically
            if file_format:
                conversion_command += ['-f', file_format]

            if codec:
                # force audio decoder
                conversion_command += ['-acodec', codec]

            if filename:
                conversion_command += ['-i', filename]
                stdin_data = None
            else:
                if cls.converter == 'ffmpeg':
                    conversion_command += ['-read_ahead_limit', str(read_ahead_limit), '-i', 'cache:pipe:0']
                else:
                    conversion_command += ['-i', '-']
                stdin_data = f.read()

            info = None if codec else mediainfo(orig_file, read_ahead_limit=read_ahead_limit)

            if info:
                audio_streams = [x for x in info['streams'] if x['codec_type'] == 'audio']
                # This is a workaround for some ffprobe versions that always say
                # that mp3/mp4/aac/webm/ogg files contain fltp samples
                audio_codec = audio_streams[0].get('codec_name')
                if audio_streams[0].get('sample_fmt') == 'fltp' and audio_codec in ['mp3', 'mp4', 'aac', 'webm', 'ogg']:
                    bits_per_sample = 16
                else:
                    bits_per_sample = audio_streams[0]['bits_per_sample']
                acodec = 'pcm_u8' if bits_per_sample == 8 else f'pcm_s{bits_per_sample}le'  # noqa: PLR2004

                conversion_command += ['-acodec', acodec]

            conversion_command += [
                '-vn',  # Drop any video streams if there are any
                '-f',
                'wav',  # output options (filename last)
            ]

            if start_second is not None:
                conversion_command += ['-ss', str(start_second)]

            if duration is not None:
                conversion_command += ['-t', str(duration)]

            conversion_command += ['-']

            if parameters is not None:
                # extend arguments with arbitrary set
                conversion_command.extend(parameters)

            log_conversion(conversion_command)

            result = subprocess.run(  # noqa: S603
                conversion_command,
                input=stdin_data,
                capture_output=True,
                check=False,
            )

            if result.returncode != 0 or not result.stdout:
                msg = (
                    f'Decoding failed. ffmpeg returned error code: {result.returncode}\n\n'
                    f'Output from ffmpeg/avlib:\n\n{result.stderr.decode(errors="ignore")}'
                )
                raise CouldntDecodeError(msg)

            p_out = bytearray(result.stdout)
            fix_wav_headers(p_out)
            return _apply_time_slice(cls(result.stdout), start_second=start_second, duration=duration)

    @classmethod
    def from_mp3(cls, path: str | Path, *, parameters: list[str] | None = None) -> 'AudioSegment':
        """Create an audio segment from a mp3 file."""
        return cls.from_file(path, 'mp3', parameters=parameters)

    @classmethod
    def from_flv(cls, path: str | Path | IO[bytes], *, parameters: list[str] | None = None) -> 'AudioSegment':
        """Create an audio segment from a flv file."""
        return cls.from_file(path, 'flv', parameters=parameters)

    @classmethod
    def from_ogg(cls, path: str | Path | IO[bytes], *, parameters: list[str] | None = None) -> 'AudioSegment':
        """Create an audio segment from an ogg file."""
        return cls.from_file(path, 'ogg', parameters=parameters)

    @classmethod
    def from_wav(cls, path: str | Path | IO[bytes], *, parameters: list[str] | None = None) -> 'AudioSegment':
        """Create an audio segment from a wav file."""
        return cls.from_file(path, 'wav', parameters=parameters)

    @classmethod
    def from_raw(
        cls, path: str | Path | IO[bytes], *, sample_width: int, frame_rate: int, channels: int
    ) -> 'AudioSegment':
        """Create an audio segment from a raw file."""
        return cls.from_file(
            path,
            'raw',
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels,
        )

    @classmethod
    def _from_safe_wav(cls, path: str | Path | IO[bytes] | None) -> Self:
        with fd_or_path_or_tempfile(path, mode='rb', tempfile=False) as file:
            file.seek(0)
            return cls(data=file.read())

    @contextmanager
    def _export_raw(
        self,
        out_f: str | Path | IO[bytes] | None = None,
    ) -> Generator[IO[bytes]]:
        with fd_or_path_or_tempfile(out_f, mode='wb+') as f:
            f.seek(0)
            f.write(self.raw_data)
            f.seek(0)
            yield f

    def _export_wav_logic(self, pcm_for_wav: bytes, f: IO[bytes]) -> None:
        """Handle wav export logic."""
        with wave.open(f, 'wb') as wave_data:
            wave_data.setnchannels(self.channels)
            wave_data.setsampwidth(self.sample_width)
            wave_data.setframerate(self.frame_rate)
            # For some reason packing the wave header struct with
            # a float in python 2 doesn't throw an exception
            wave_data.setnframes(int(self.frame_count()))
            wave_data.writeframesraw(pcm_for_wav)

    @contextmanager
    def _export_generic(  # noqa: C901, PLR0912
        self,
        pcm_for_wav: bytes,
        out_f: str | Path | IO[bytes] | None = None,
        f_format: Literal['mp3', 'wav', 'raw', 'ogg'] | str = 'mp3',  # noqa: PYI051
        codec: str | None = None,
        bitrate: str | None = None,
        parameters: list[str] | None = None,
        tags: dict[str, str] | None = None,
        id3v2_version: str = '4',
        cover: str | None = None,
    ) -> Generator[IO[bytes]]:
        id3v2_allowed_versions = ['3', '4']

        with (
            fd_or_path_or_tempfile(out_f, mode='wb+') as f,
            NamedTemporaryFile(mode='wb', delete=True, delete_on_close=False) as f_tmp,
            NamedTemporaryFile(mode='w+b', delete=True, delete_on_close=False) as output,
        ):
            f.seek(0)
            self._export_wav_logic(pcm_for_wav, f_tmp)
            f_tmp.close()

            # build converter command to export
            conversion_command = [
                self.converter,
                '-y',  # always overwrite existing files
                '-f',
                'wav',
                '-i',
                f_tmp.name,  # input options (filename last)
            ]

            if codec is None:
                codec = DEFAULT_CODECS.get(f_format)

            if cover is not None:
                if cover.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')) and f_format == 'mp3':
                    conversion_command.extend(['-i', cover, '-map', '0', '-map', '1', '-c:v', 'mjpeg'])
                else:
                    msg = (
                        'Currently cover images are only supported by MP3 files. '
                        'The allowed image formats are: .tif, .jpg, .bmp, .jpeg and .png.'
                    )
                    raise AttributeError(msg)

            if codec is not None:
                # force audio encoder
                conversion_command.extend(['-acodec', codec])

            if bitrate is not None:
                conversion_command.extend(['-b:a', bitrate])

            if parameters is not None:
                # extend arguments with arbitrary set
                conversion_command.extend(parameters)

            if tags is not None:
                # Extend converter command with tags
                for key, value in tags.items():
                    conversion_command.extend(['-metadata', f'{key}={value}'])

                if f_format == 'mp3':
                    # set id3v2 tag version
                    if id3v2_version not in id3v2_allowed_versions:
                        msg = f'id3v2_version not allowed, allowed versions: {id3v2_allowed_versions}'
                        raise InvalidID3TagVersionError(msg)
                    conversion_command.extend(['-id3v2_version', id3v2_version])

            if sys.platform == 'darwin' and codec == 'mp3':
                conversion_command.extend(['-write_xing', '0'])

            conversion_command.extend(
                [
                    '-f',
                    f_format,
                    output.name,  # output options (filename last)
                ]
            )

            log_conversion(conversion_command)

            # read stdin / write stdout
            result = subprocess.run(  # noqa: S603
                conversion_command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
            )
            log_subprocess_output(result.stdout)
            log_subprocess_output(result.stderr)

            try:
                if result.returncode != 0:
                    msg = (
                        f'Encoding failed. ffmpeg/avlib returned error code: {result.returncode}\n\n'
                        f'Command:{conversion_command}\n\n'
                        f'Output from ffmpeg/avlib:\n\n'
                        f'{result.stderr}'
                    )
                    raise CouldntEncodeError(msg)

                output.seek(0)
                f.write(output.read())
                f.seek(0)
                yield f
            finally:
                output.close()

    @contextmanager
    def export_handle(
        self,
        out_f: str | Path | IO[bytes] | None = None,
        f_format: Literal['mp3', 'wav', 'raw', 'ogg'] | str = 'mp3',  # noqa: PYI051
        codec: str | None = None,
        bitrate: str | None = None,
        parameters: list[str] | None = None,
        tags: dict[str, str] | None = None,
        id3v2_version: str = '4',
        cover: str | None = None,
    ) -> Generator[IO[bytes]]:
        """Export an AudioSegment to a file with given options.

        out_f (string):
            Path to destination audio file. Also accepts os.PathLike objects on
            python >= 3.6

        format (string)
            Format for destination audio file.
            ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)

        codec (string)
            Codec used to encode the destination file.

        bitrate (string)
            Bitrate used when encoding destination file. (64, 92, 128, 256, 312k...)
            Each codec accepts different bitrate arguments so take a look at the
            ffmpeg documentation for details (bitrate usually shown as -b, -ba or
            -a:b).

        parameters (list of strings)
            Aditional ffmpeg/avconv parameters

        tags (dict)
            Set metadata information to destination files
            usually used as tags. ({title='Song Title', artist='Song Artist'})

        id3v2_version (string)
            Set ID3v2 version for tags. (default: '4')

        cover (file)
            Set cover for audio file from image file. (png or jpg)
        """
        if f_format == 'raw' and (codec is not None or parameters is not None):
            msg = (
                'Can not invoke ffmpeg when export format is "raw"; '
                'specify an ffmpeg raw format like format="s16le" instead '
                'or call export(format="raw") with no codec or parameters'
            )
            raise AttributeError(msg)

        if f_format == 'raw':
            with self._export_raw(out_f) as f:
                yield f
            return

        # wav with no ffmpeg parameters can just be written directly to out_f
        easy_wav = f_format == 'wav' and codec is None and parameters is None

        pcm_for_wav = self._data
        if self.sample_width == 1:
            # convert to unsigned integers for wav
            pcm_for_wav = audioop.bias(self._data, 1, 128)

        if easy_wav:
            with fd_or_path_or_tempfile(out_f, mode='wb+') as f:
                f.seek(0)
                self._export_wav_logic(pcm_for_wav, f)
                f.seek(0)
                yield f
            return

        with self._export_generic(
            pcm_for_wav, out_f, f_format, codec, bitrate, parameters, tags, id3v2_version, cover
        ) as f:
            yield f

    def export(
        self,
        out_f: str | Path | IO[bytes] | None = None,
        f_format: Literal['mp3', 'wav', 'raw', 'ogg'] | str = 'mp3',  # noqa: PYI051
        codec: str | None = None,
        bitrate: str | None = None,
        parameters: list[str] | None = None,
        tags: dict[str, str] | None = None,
        id3v2_version: str = '4',
        cover: str | None = None,
    ) -> IO[bytes]:
        """Export an AudioSegment to a file with given options.

        out_f (string):
            Path to destination audio file. Also accepts os.PathLike objects on
            python >= 3.6

        format (string)
            Format for destination audio file.
            ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)

        codec (string)
            Codec used to encode the destination file.

        bitrate (string)
            Bitrate used when encoding destination file. (64, 92, 128, 256, 312k...)
            Each codec accepts different bitrate arguments so take a look at the
            ffmpeg documentation for details (bitrate usually shown as -b, -ba or
            -a:b).

        parameters (list of strings)
            Aditional ffmpeg/avconv parameters

        tags (dict)
            Set metadata information to destination files
            usually used as tags. ({title='Song Title', artist='Song Artist'})

        id3v2_version (string)
            Set ID3v2 version for tags. (default: '4')

        cover (file)
            Set cover for audio file from image file. (png or jpg)
        """
        with self.export_handle(out_f, f_format, codec, bitrate, parameters, tags, id3v2_version, cover) as f:
            return f

    def get_frame(self, index: int) -> bytes:
        """Get frame."""
        frame_start = index * self.frame_width
        frame_end = frame_start + self.frame_width
        return self.raw_data[frame_start:frame_end]

    def frame_count(self, ms: float | None = None) -> float:
        """
        Returns the number of frames for the given number of milliseconds.

        If not specified, the number of frames in the whole AudioSegment.
        """
        if ms is not None:
            return ms * (self.frame_rate / 1000.0)
        return float(len(self._data) // self.frame_width)

    def set_sample_width(self, sample_width: int) -> Self:
        """Set sample width."""
        if sample_width == self.sample_width:
            return self
        frame_width = self.channels * sample_width
        return self._spawn(
            audioop.lin2lin(self._data, self.sample_width, sample_width),
            overrides={'sample_width': sample_width, 'frame_width': frame_width},
        )

    def set_frame_rate(self, frame_rate: int) -> Self:
        """Set frame rate."""
        if frame_rate == self.frame_rate:
            return self
        if self._data:
            converted, _ = audioop.ratecv(
                self._data, self.sample_width, self.channels, self.frame_rate, frame_rate, None
            )
        else:
            converted = self._data
        return self._spawn(data=converted, overrides={'frame_rate': frame_rate})

    def set_channels(self, channels: int) -> Self:
        """
        Set the number of channels in the audio segment.

        Supports mono-to-multi channel and multi-to-mono channel conversions.
        Specifically handles stereo-to-mono and mono-to-stereo using optimized
        audioop functions, and supports arbitrary mono-to-multi or multi-to-mono
        conversions through downmixing/duplication.

        Args:
            channels (int): The target number of channels.

        Returns:
            Self: A new AudioSegment instance with the specified number of channels.

        Raises:
            ValueError: If the requested conversion is not a mono-to-multi or
            multi-to-mono channel conversion.
        """
        if channels == self.channels:
            return self

        if channels == 2 and self.channels == 1:  # noqa: PLR2004
            frame_width = self.frame_width * 2
            fac = 1
            converted = audioop.tostereo(self._data, self.sample_width, fac, fac)
        elif channels == 1 and self.channels == 2:  # noqa: PLR2004
            frame_width = self.frame_width // 2
            fac = 0.5
            converted = audioop.tomono(self._data, self.sample_width, fac, fac)
        elif channels == 1:
            channels_data = [seg.get_array_of_samples() for seg in self.split_to_mono()]
            frame_count = int(self.frame_count())
            converted = array.array(channels_data[0].typecode, b'\0' * (frame_count * self.sample_width))
            for raw_channel_data in channels_data:
                for i in range(frame_count):
                    converted[i] += raw_channel_data[i] // self.channels
            frame_width = self.frame_width // self.channels
            converted = converted.tobytes()
        elif self.channels == 1:
            dup_channels = [self for _ in range(channels)]
            new_audio = AudioSegment.from_mono_audiosegments(*dup_channels)
            return self._spawn(
                data=new_audio.raw_data,
                overrides={
                    'channels': channels,
                    'frame_width': channels * self.sample_width,
                },
            )
        else:
            msg = 'AudioSegment.set_channels only supports mono-to-multi channel and multi-to-mono channel conversion'
            raise ValueError(msg)

        return self._spawn(
            data=converted,
            overrides={
                'channels': channels,
                'frame_width': frame_width,
            },
        )

    def split_to_mono(self) -> list[Self]:
        """Split the audio's mono channel."""
        if self.channels == 1:
            return [self]

        samples = self.get_array_of_samples()

        mono_channels: list[Self] = []
        for i in range(self.channels):
            samples_for_current_channel = samples[i :: self.channels]

            try:
                mono_data = samples_for_current_channel.tobytes()
            except AttributeError:
                mono_data = samples_for_current_channel.tobytes()

            mono_channels.append(
                self._spawn(
                    mono_data,
                    overrides={
                        'channels': 1,
                        'frame_width': self.sample_width,
                    },
                )
            )

        return mono_channels

    @property
    def rms(self) -> int:
        """Return the root-mean-square of the audio segment."""
        return audioop.rms(self._data, self.sample_width)

    @property
    def dbfs(self) -> float:
        """Return the db to full-scale of the audio segment."""
        rms = self.rms
        if not rms:
            return -float('infinity')
        return power_to_db(self.rms / self.max_possible_amplitude)

    @property
    def max(self) -> int:
        """Return the max value of the segment."""
        return audioop.max(self._data, self.sample_width)

    @property
    def max_possible_amplitude(self) -> float:
        """Return the maximum possible amplitude."""
        bits = self.sample_width * 8
        max_possible_val: int = 2**bits
        # since half is above 0 and half is below the max amplitude is divided
        return max_possible_val / 2

    @property
    def max_dbfs(self) -> float:
        """Return the maximum db to full-scale of the audio segment."""
        return power_to_db(self.max / self.max_possible_amplitude)

    @property
    def duration_seconds(self) -> float:
        """Return duration of audio segment in seconds."""
        return (self.frame_rate and self.frame_count() / self.frame_rate) or 0.0

    def get_dc_offset(self, *, channel: int = 1) -> float:
        """Returns a value between -1.0 and 1.0 representing the DC offset of a channel (1 for left, 2 for right)."""
        if not 1 <= channel <= 2:  # noqa: PLR2004
            msg = 'channel value must be 1 (left) or 2 (right)'
            raise ValueError(msg)

        if self.channels == 1:
            data = self._data
        elif channel == 1:
            data = audioop.tomono(self._data, self.sample_width, 1, 0)
        else:
            data = audioop.tomono(self._data, self.sample_width, 0, 1)

        return float(audioop.avg(data, self.sample_width)) / self.max_possible_amplitude

    def remove_dc_offset(self, offset: int, channel: int | None = None) -> Self:
        """Removes DC offset of given channel. Calculates offset if it's not given.

        Offset values must be in range -1.0 to 1.0. If channel is None, removes
        DC offset from all available channels.
        """
        if channel and not 1 <= channel <= 2:  # noqa: PLR2004
            msg = 'channel value must be None, 1 (left) or 2 (right)'
            raise ValueError(msg)

        if offset and not -1.0 <= offset <= 1.0:
            msg = 'offset value must be in range -1.0 to 1.0'
            raise ValueError(msg)

        if offset:
            offset = round(offset * self.max_possible_amplitude)

        def remove_data_dc(data: bytes, off: int) -> bytes:
            if not off:
                off = audioop.avg(data, self.sample_width)
            return audioop.bias(data, self.sample_width, -off)

        if self.channels == 1:
            return self._spawn(data=remove_data_dc(self.raw_data, offset))

        left_channel = audioop.tomono(self.raw_data, self.sample_width, 1, 0)
        right_channel = audioop.tomono(self.raw_data, self.sample_width, 0, 1)

        if not channel or channel == 1:
            left_channel = remove_data_dc(left_channel, offset)

        if not channel or channel == 2:  # noqa: PLR2004
            right_channel = remove_data_dc(right_channel, offset)

        left_channel = audioop.tostereo(left_channel, self.sample_width, 1, 0)
        right_channel = audioop.tostereo(right_channel, self.sample_width, 0, 1)

        return self._spawn(data=audioop.add(left_channel, right_channel, self.sample_width))

    def apply_gain(self, *, volume_change: float) -> Self:
        """Apply gain."""
        return self._spawn(data=audioop.mul(self._data, self.sample_width, db_to_power(float(volume_change))))

    def overlay(
        self,
        seg: 'AudioSegment',
        *,
        position: int = 0,
        loop: bool = False,
        times: int | None = None,
        gain_during_overlay: int | None = None,
    ) -> Self:
        """Overlay the provided segment on to this segment starting at the specificed position.

        It uses the specfied looping beahvior.

        seg (AudioSegment):
            The audio segment to overlay on to this one.

        position (optional int):
            The position to start overlaying the provided segment in to this
            one.

        loop (optional bool):
            Loop seg as many times as necessary to match this segment's length.
            Overrides loops param.

        times (optional int):
            Loop seg the specified number of times or until it matches this
            segment's length. 1 means once, 2 means twice, ... 0 would make the
            call a no-op
        gain_during_overlay (optional int):
            Changes this segment's volume by the specified amount during the
            duration of time that seg is overlaid on top of it. When negative,
            this has the effect of 'ducking' the audio under the overlay.
        """
        if loop:
            # match loop=True's behavior with new times (count) mechinism.
            times = -1
        elif times is None:
            # no times specified, just once through
            times = 1
        elif times == 0:
            # it's a no-op, make a copy since we never mutate
            return self._spawn(self._data)

        output = BytesIO()

        seg1, seg2 = AudioSegment._sync(self, seg)
        sample_width = seg1.sample_width

        output.write(seg1[:position].raw_data)

        # drop down to the raw data
        seg1 = seg1[position:].raw_data
        seg2 = seg2.raw_data
        pos = 0
        seg1_len = len(seg1)
        seg2_len = len(seg2)
        while times:
            remaining = max(0, seg1_len - pos)
            if seg2_len >= remaining:
                seg2 = seg2[:remaining]
                seg2_len = remaining
                # we've hit the end, we're done looping (if we were) and this
                # is our last go-around
                times = 1

            if gain_during_overlay:
                seg1_overlaid = seg1[pos : pos + seg2_len]
                seg1_adjusted_gain = audioop.mul(
                    seg1_overlaid, self.sample_width, db_to_power(float(gain_during_overlay))
                )
                output.write(audioop.add(seg1_adjusted_gain, seg2, sample_width))
            else:
                output.write(audioop.add(seg1[pos : pos + seg2_len], seg2, sample_width))
            pos += seg2_len

            # dec times to break our while loop (eventually)
            times -= 1

        output.write(seg1[pos:])

        return self._spawn(data=output.getvalue())

    def append(self, seg: 'AudioSegment', *, crossfade: int = 100) -> Self:
        """Append an audio segment to the current one."""
        out: tuple[Self, AudioSegment] = AudioSegment._sync(self, seg)  # pyright: ignore[reportAssignmentType]
        seg1, seg2 = out

        if not crossfade:
            return seg1._spawn(seg1.raw_data + seg2.raw_data)  # noqa: SLF001
        if crossfade > len(self):
            msg = f'Crossfade is longer than the original AudioSegment ({crossfade}ms > {len(self)}ms)'
            raise ValueError(msg)
        if crossfade > len(seg):
            msg = f'Crossfade is longer than the appended AudioSegment ({crossfade}ms > {len(seg)}ms)'
            raise ValueError(msg)

        xf = seg1[-crossfade:].fade(to_gain=-120)
        xf *= seg2[:crossfade].fade(from_gain=-120)

        output = BytesIO()

        output.write(seg1[:-crossfade].raw_data)
        output.write(xf.raw_data)
        output.write(seg2[crossfade:].raw_data)

        output.seek(0)
        obj = seg1._spawn(data=output.getvalue())  # noqa: SLF001
        output.close()
        return obj

    def fade(
        self,
        to_gain: float = 0,
        from_gain: float = 0,
        start: int | None = None,
        end: int | None = None,
        duration: int | None = None,
    ) -> Self:
        """Fade the volume of this audio segment."""
        if None not in [duration, end, start]:
            msg = 'Only two of the three arguments, "start", "end", and "duration" may be specified.'
            raise InvalidParametersError(msg)

        # no fade == the same audio
        if to_gain == 0 and from_gain == 0:
            return self

        start = clamp(start, 0, len(self))
        end = clamp(end, 0, len(self))

        if duration is not None and duration < 0:
            msg = 'duration must be a positive integer.'
            raise InvalidDurationError(msg)

        start, end, duration = resolve_time_range(start, end, duration, len(self))

        from_power = db_to_power(from_gain)

        output: list[bytes] = []

        # original data - up until the crossfade portion, as is
        before_fade = self[:start].raw_data
        if from_gain != 0:
            before_fade = audioop.mul(before_fade, self.sample_width, from_power)
        output.append(before_fade)

        gain_delta = db_to_power(to_gain) - from_power

        # fades longer than 100ms can use coarse fading (one gain step per ms),
        # shorter fades will have audible clicks so they use precise fading
        # (one gain step per sample)
        if duration > 100:  # noqa: PLR2004
            scale_step = gain_delta / duration

            for i in range(duration):
                volume_change = from_power + (scale_step * i)
                chunk = self[start + i]
                chunk = audioop.mul(chunk.raw_data, self.sample_width, volume_change)

                output.append(chunk)
        else:
            start_frame = self.frame_count(ms=start)
            end_frame = self.frame_count(ms=end)
            fade_frames = end_frame - start_frame
            scale_step = gain_delta / fade_frames

            for i in range(int(fade_frames)):
                volume_change = from_power + (scale_step * i)
                sample = self.get_frame(int(start_frame + i))
                sample = audioop.mul(sample, self.sample_width, volume_change)

                output.append(sample)

        # original data after the crossfade portion, at the new volume
        after_fade = self[end:].raw_data
        if to_gain != 0:
            after_fade = audioop.mul(after_fade, self.sample_width, db_to_power(to_gain))
        output.append(after_fade)

        return self._spawn(data=b''.join(output))

    def fade_out(self, duration: int) -> Self:
        """Fade out the volume of this audio segment."""
        return self.fade(to_gain=-120, duration=duration)

    def fade_in(self, duration: int) -> Self:
        """Fade in the volume of this audio segment."""
        return self.fade(from_gain=-120, duration=duration, start=0)

    def reverse(self) -> Self:
        """Reverse this audio segment."""
        return self._spawn(data=audioop.reverse(self._data, self.sample_width))

    def _repr_html_(self) -> str:
        src = """
            <audio controls>
                <source src="data:audio/mpeg;base64,{base64}" type="audio/mpeg"/>
                Your browser does not support the audio element.
            </audio>
        """
        fh = self.export()
        data = base64.b64encode(fh.read()).decode('ascii')
        return src.format(base64=data)

    def pipe[**P](self, fn: Callable[Concatenate[Self, P], Self], *args: P.args, **kwargs: P.kwargs) -> Self:
        """Allow easily piping functions."""
        return fn(self, *args, **kwargs)
