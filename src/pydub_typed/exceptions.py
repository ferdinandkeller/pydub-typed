"""Exceptions module."""


class PydubError(Exception):
    """Base class for any Pydub exception."""


class TooManyMissingFramesError(PydubError):
    """Too many missing frames."""


class InvalidDurationError(PydubError):
    """Invalid Duration."""


class InvalidTagError(PydubError):
    """Invalid Tag."""


class InvalidID3TagVersionError(PydubError):
    """Invalid ID D3 Tag Version."""


class CouldntDecodeError(PydubError):
    """Could not decode."""


class CouldntEncodeError(PydubError):
    """Could not encode."""


class InvalidParametersError(PydubError):
    """Invalid Parameters."""


class MissingAudioParameterError(InvalidParametersError):
    """Missing Audio Parameter."""


class UnreachableCodeError(PydubError, RuntimeError):
    """Code that should not be reached has been reached."""
