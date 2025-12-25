"""
Vocal Separator - Separates vocals from music files using Demucs.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from .config import get_config
    from .logging_config import get_logger
    from .validators import AudioValidator, ValidationError
except ImportError:
    from config import get_config
    from logging_config import get_logger
    from validators import AudioValidator, ValidationError

logger = get_logger(__name__)


class VocalSeparatorError(Exception):
    """Exception raised for vocal separation errors."""

    pass


class VocalSeparator:
    """Separates vocals from music using Demucs."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize vocal separator.

        Args:
            model_name: Demucs model name (htdemucs, htdemucs_ft, mdx_extra, etc.)
        """
        cfg = get_config()
        self.model_name = model_name if model_name is not None else cfg.demucs_model
        logger.info(f"VocalSeparator initialized with model: {self.model_name}")

    def separate(self, audio_path: str, output_dir: str) -> Dict[str, Path]:
        """
        Separate audio file into vocal and instrumental stems.

        Args:
            audio_path: Input audio file path
            output_dir: Output directory for separated files

        Returns:
            Dictionary with paths to vocals, instrumental, and original

        Raises:
            ValidationError: If audio file is invalid
            VocalSeparatorError: If separation fails
        """
        try:
            # Validate input file
            audio_path = AudioValidator.validate_audio_file(audio_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Separating vocals from: {audio_path.name}")

            # Demucs command - use sys.executable to ensure correct Python
            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems=vocals",  # Faster: only vocal/instrumental
                "-n",
                self.model_name,
                "-o",
                str(output_dir),
                str(audio_path),
            ]

            # Run Demucs with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,  # 10 minute timeout
            )

            # Demucs output structure: output_dir/model_name/song_name/vocals.wav
            song_name = audio_path.stem
            separated_dir = output_dir / self.model_name / song_name

            vocals_path = separated_dir / "vocals.wav"
            instrumental_path = separated_dir / "no_vocals.wav"

            # Verify output files exist
            if not vocals_path.exists() or not instrumental_path.exists():
                raise VocalSeparatorError(
                    f"Demucs separation incomplete. Expected files not found in {separated_dir}"
                )

            logger.info(f"Successfully separated: {audio_path.name}")

            return {
                "vocals": vocals_path,
                "instrumental": instrumental_path,
                "original": audio_path,
            }

        except subprocess.TimeoutExpired as e:
            error_msg = f"Demucs timeout after 10 minutes: {audio_path.name}"
            logger.error(error_msg)
            raise VocalSeparatorError(error_msg) from e

        except subprocess.CalledProcessError as e:
            error_msg = f"Demucs separation failed: {e.stderr}"
            logger.error(error_msg)
            raise VocalSeparatorError(error_msg) from e

        except ValidationError:
            raise

        except Exception as e:
            error_msg = f"Unexpected error during separation: {e}"
            logger.error(error_msg)
            raise VocalSeparatorError(error_msg) from e

    def get_instrumental_only(self, audio_path: str, output_dir: str) -> Path:
        """
        Extract only the instrumental track.

        Args:
            audio_path: Input audio file
            output_dir: Output directory

        Returns:
            Path to instrumental file

        Raises:
            VocalSeparatorError: If separation fails
        """
        result = self.separate(audio_path, output_dir)
        return result["instrumental"]


class SimpleStemExtractor:
    """
    Alternative approach: simple vocal reduction without Demucs.
    Use as fallback if Demucs is not available.
    """

    @staticmethod
    def extract_instrumental_basic(audio_path: str, output_path: str) -> Path:
        """
        Basic vocal reduction using spectral subtraction.

        Note: This method is less effective than Demucs, only use as fallback.

        Args:
            audio_path: Input audio file path
            output_path: Output file path

        Returns:
            Path to output file

        Raises:
            VocalSeparatorError: If extraction fails
        """
        try:
            import librosa
            import numpy as np
            import soundfile as sf

            logger.info(f"Basic vocal reduction for: {audio_path}")

            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, mono=False)

            if len(y.shape) == 1:
                # Mono to stereo
                y = np.stack([y, y])

            # Simple vocal reduction: center channel subtraction
            # Vocals are typically in the center channel
            if y.shape[0] == 2:
                # Side signal (mostly instrumental)
                instrumental = y[0] - y[1]
            else:
                instrumental = y[0]

            # Normalize
            instrumental = librosa.util.normalize(instrumental)

            # Save
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, instrumental, sr)

            logger.info(f"Basic extraction saved to: {output_path}")
            return output_path

        except Exception as e:
            error_msg = f"Basic vocal extraction failed: {e}"
            logger.error(error_msg)
            raise VocalSeparatorError(error_msg) from e


if __name__ == "__main__":
    import sys

    # Test
    separator = VocalSeparator()

    # Example usage
    test_file = "backend/data/raw/test_song.mp3"
    if Path(test_file).exists():
        try:
            result = separator.separate(test_file, "backend/temp/separated")
            logger.info(f"Instrumental saved to: {result['instrumental']}")
        except (ValidationError, VocalSeparatorError) as e:
            logger.error(f"Separation failed: {e}")
            sys.exit(1)
    else:
        logger.warning(f"Test file not found: {test_file}")
