"""
Input validation utilities for audio files and parameters.
"""

from pathlib import Path
from typing import List, Optional, Union
import soundfile as sf


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class AudioValidator:
    """Validator for audio files and parameters."""

    # Supported audio file extensions
    # Common lossless and lossy formats widely used in music production
    # .mp3 - Most common lossy format
    # .wav - Uncompressed lossless format
    # .flac - Compressed lossless format
    # .ogg - Open-source lossy format
    # .m4a - AAC format, common in Apple ecosystem
    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    
    MAX_SAMPLE_RATE = 96000
    MIN_SAMPLE_RATE = 8000
    MAX_CHANNELS = 8
    MIN_DURATION_SEC = 0.1
    MAX_DURATION_SEC = 7200  # 2 hours

    @classmethod
    def validate_audio_file(
        cls,
        file_path: Union[str, Path],
        max_duration_sec: Optional[float] = None,
        max_channels: Optional[int] = None,
    ) -> Path:
        """
        Validate an audio file exists and is readable.

        Args:
            file_path: Path to audio file
            max_duration_sec: Maximum allowed duration in seconds
            max_channels: Maximum allowed channels

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValidationError(f"Audio file not found: {file_path}")

        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        if file_path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        # Validate file integrity and metadata
        try:
            with sf.SoundFile(str(file_path)) as audio:
                duration = len(audio) / float(audio.samplerate)
                channels = audio.channels
                sample_rate = audio.samplerate

                if duration < cls.MIN_DURATION_SEC:
                    raise ValidationError(
                        f"Audio too short: {duration:.2f}s (min: {cls.MIN_DURATION_SEC}s)"
                    )

                max_dur = max_duration_sec or cls.MAX_DURATION_SEC
                if duration > max_dur:
                    raise ValidationError(
                        f"Audio too long: {duration:.2f}s (max: {max_dur}s)"
                    )

                max_ch = max_channels or cls.MAX_CHANNELS
                if channels > max_ch:
                    raise ValidationError(
                        f"Too many channels: {channels} (max: {max_ch})"
                    )

                if not (cls.MIN_SAMPLE_RATE <= sample_rate <= cls.MAX_SAMPLE_RATE):
                    raise ValidationError(
                        f"Invalid sample rate: {sample_rate} Hz "
                        f"(range: {cls.MIN_SAMPLE_RATE}-{cls.MAX_SAMPLE_RATE})"
                    )

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Cannot read audio file: {e}")

        return file_path

    @classmethod
    def validate_directory(
        cls, dir_path: Union[str, Path], must_exist: bool = True
    ) -> Path:
        """
        Validate a directory path.

        Args:
            dir_path: Directory path
            must_exist: Whether directory must already exist

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        dir_path = Path(dir_path)

        if must_exist:
            if not dir_path.exists():
                raise ValidationError(f"Directory not found: {dir_path}")
            if not dir_path.is_dir():
                raise ValidationError(f"Path is not a directory: {dir_path}")

        return dir_path

    @classmethod
    def validate_audio_files_in_directory(
        cls, dir_path: Union[str, Path]
    ) -> List[Path]:
        """
        Find and validate all audio files in a directory.

        Args:
            dir_path: Directory path

        Returns:
            List of validated audio file paths

        Raises:
            ValidationError: If directory invalid or no files found
        """
        dir_path = cls.validate_directory(dir_path, must_exist=True)

        audio_files = []
        for ext in cls.SUPPORTED_EXTENSIONS:
            audio_files.extend(dir_path.glob(f"*{ext}"))

        if not audio_files:
            raise ValidationError(
                f"No audio files found in {dir_path} "
                f"(supported: {', '.join(cls.SUPPORTED_EXTENSIONS)})"
            )

        return sorted(audio_files)


class ParameterValidator:
    """Validator for numeric parameters."""

    @staticmethod
    def validate_positive_int(value: int, name: str, max_value: Optional[int] = None) -> int:
        """Validate positive integer parameter."""
        if not isinstance(value, int):
            raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}, got {value}")
        return value

    @staticmethod
    def validate_positive_float(
        value: float, name: str, max_value: Optional[float] = None
    ) -> float:
        """Validate positive float parameter."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric, got {type(value).__name__}")
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}, got {value}")
        return float(value)

    @staticmethod
    def validate_probability(value: float, name: str) -> float:
        """Validate probability value [0, 1]."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric, got {type(value).__name__}")
        if not (0 <= value <= 1):
            raise ValidationError(f"{name} must be in [0, 1], got {value}")
        return float(value)
