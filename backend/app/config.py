"""
Configuration management - centralized settings for the application.
"""

import os
from pathlib import Path


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class Config:
    """Application configuration with environment variable support."""

    def __init__(self):
        """Initialize configuration."""
        self._base_dir = Path(__file__).resolve().parents[2]
        self._data_dir = None
        self._models_dir = None
        self._temp_dir = None
        self._uploads_dir = None

    @property
    def base_dir(self) -> Path:
        """Get base directory of the application."""
        return self._base_dir

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        if self._data_dir is None:
            self._data_dir = Path(
                os.getenv("MUSIC_DETECTOR_DATA_DIR", str(self.base_dir / "backend" / "data"))
            )
        return self._data_dir

    @property
    def raw_dir(self) -> Path:
        """Get raw data directory path."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Get processed data directory path."""
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        if self._models_dir is None:
            self._models_dir = Path(
                os.getenv("MUSIC_DETECTOR_MODELS_DIR", str(self.data_dir / "models"))
            )
        return self._models_dir

    @property
    def analysis_dir(self) -> Path:
        """Get analysis directory path."""
        return self.data_dir / "analysis"

    @property
    def temp_dir(self) -> Path:
        """Get temporary files directory path."""
        if self._temp_dir is None:
            self._temp_dir = Path(
                os.getenv("MUSIC_DETECTOR_TEMP_DIR", str(self.base_dir / "backend" / "temp"))
            )
        return self._temp_dir

    @property
    def uploads_dir(self) -> Path:
        """Get uploads directory path."""
        if self._uploads_dir is None:
            self._uploads_dir = Path(
                os.getenv(
                    "MUSIC_DETECTOR_UPLOADS_DIR", str(self.base_dir / "backend" / "uploads")
                )
            )
        return self._uploads_dir

    @property
    def ai_generated_dir(self) -> Path:
        """Get AI generated music directory path."""
        return self.raw_dir / "ai_generated"

    @property
    def human_made_dir(self) -> Path:
        """Get human made music directory path."""
        return self.raw_dir / "human_made"

    @property
    def sample_rate(self) -> int:
        """Get audio sample rate."""
        try:
            return int(os.getenv("MUSIC_DETECTOR_SAMPLE_RATE", "22050"))
        except ValueError as e:
            raise ConfigError(f"Invalid MUSIC_DETECTOR_SAMPLE_RATE value: {e}")

    @property
    def demucs_model(self) -> str:
        """Get Demucs model name."""
        return os.getenv("MUSIC_DETECTOR_DEMUCS_MODEL", "htdemucs")

    @property
    def latest_model_path(self) -> Path:
        """Get latest model file path."""
        return self.models_dir / "latest_model.pkl"

    @property
    def latest_scaler_path(self) -> Path:
        """Get latest scaler file path."""
        return self.models_dir / "latest_scaler.pkl"

    @property
    def latest_metadata_path(self) -> Path:
        """Get latest metadata file path."""
        return self.models_dir / "latest_metadata.json"

    @property
    def features_file(self) -> Path:
        """Get features CSV file path."""
        return self.processed_dir / "features.csv"

    @property
    def metadata_file(self) -> Path:
        """Get metadata JSON file path."""
        return self.processed_dir / "metadata.json"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.raw_dir,
            self.ai_generated_dir,
            self.human_made_dir,
            self.processed_dir,
            self.models_dir,
            self.analysis_dir,
            self.temp_dir,
            self.uploads_dir,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """
    Get the application configuration singleton.

    Returns:
        Config instance
    """
    if not hasattr(get_config, "_instance"):
        get_config._instance = Config()
    return get_config._instance
