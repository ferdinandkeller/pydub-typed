"""Generators module.

Each generator will return float samples from -1.0 to 1.0, which can be
converted to actual audio with 8, 16, 24, or 32 bit depth using the
SiganlGenerator.to_audio_segment() method (on any of it's subclasses).

See Wikipedia's "waveform" page for info on some of the generators included
here: http://en.wikipedia.org/wiki/Waveform
"""

import array
import itertools
import math
import random
from collections.abc import Generator
from dataclasses import dataclass

from .audio_segment import AudioSegment
from .utils import db_to_power, get_array_type, get_frame_width, get_min_max_value


@dataclass
class SignalGenerator:
    """Signal generator."""

    sample_rate: int = 44100
    bit_depth: int = 16

    def to_audio_segment(self, *, duration: float = 1000.0, volume: float = 0.0) -> AudioSegment:
        """To audio segment.

        Duration in milliseconds
            (default: 1 second)
        Volume in DB relative to maximum amplitude
            (default 0.0 dBFS, which is the maximum value)
        """
        _, maxval = get_min_max_value(self.bit_depth)
        sample_width = get_frame_width(self.bit_depth)
        array_type = get_array_type(self.bit_depth)

        gain = db_to_power(volume)
        sample_count = int(self.sample_rate * (duration / 1000.0))

        sample_data = (int(val * maxval * gain) for val in self.generate())
        sample_data = itertools.islice(sample_data, 0, sample_count)

        data = array.array(array_type, sample_data)

        return AudioSegment(
            data=data.tobytes(),
            metadata={
                'channels': 1,
                'sample_width': sample_width,
                'frame_rate': self.sample_rate,
                'frame_width': sample_width,
            },
        )

    def generate(self) -> Generator[float]:
        """Generate function to be implemented."""
        msg = (
            'SignalGenerator subclasses must implement the generate() method, '
            'and *should not* call the superclass implementation.'
        )
        raise NotImplementedError(msg)


@dataclass
class Sine(SignalGenerator):
    """Signe generator."""

    freq: float = 1

    def generate(self) -> Generator[float]:
        """Generate sine waves."""
        sine_of = (self.freq * 2 * math.pi) / self.sample_rate
        sample_n = 0
        while True:
            yield math.sin(sine_of * sample_n)
            sample_n += 1


@dataclass
class Pulse(SignalGenerator):
    """Pulse generator."""

    freq: float = 1
    duty_cycle: float = 0.5

    def generate(self) -> Generator[float]:
        """Generate pulse."""
        sample_n = 0

        # in samples
        cycle_length = self.sample_rate / float(self.freq)
        pulse_length = cycle_length * self.duty_cycle

        while True:
            if (sample_n % cycle_length) < pulse_length:
                yield 1.0
            else:
                yield -1.0
            sample_n += 1


@dataclass
class Square(Pulse):
    """Square generator."""


@dataclass
class Sawtooth(SignalGenerator):
    """Sawtooth generator."""

    freq: float = 1
    duty_cycle: float = 1.0

    def generate(self) -> Generator[float]:
        """Generate sawtooths."""
        sample_n = 0

        # in samples
        cycle_length = self.sample_rate / float(self.freq)
        midpoint = cycle_length * self.duty_cycle
        ascend_length = midpoint
        descend_length = cycle_length - ascend_length

        while True:
            cycle_position = sample_n % cycle_length
            if cycle_position < midpoint:
                yield (2 * cycle_position / ascend_length) - 1.0
            else:
                yield 1.0 - (2 * (cycle_position - midpoint) / descend_length)
            sample_n += 1


@dataclass
class Triangle(Sawtooth):
    """Triangle generator."""


@dataclass
class WhiteNoise(SignalGenerator):
    """White noise generator."""

    def generate(self) -> Generator[float]:
        """Generate white noise."""
        while True:
            yield (random.random() * 2) - 1.0  # noqa: S311
