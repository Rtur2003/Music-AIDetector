"""
Configuration management - centralized settings for the application.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration with environment variable support."""

    def __init__(self):
        self._base_dir = Path(__file__).resolve().parents[2]
        self._data_dir = None
        self._models_dir = None
        self._temp_dir = None
        self._uploads_dir = None

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def data_dir(self) -> Path:
        if self._data_dir is None:
            self._data_dir = Path(
                os.getenv("MUSIC_DETECTOR_DATA_DIR", str(self.base_dir / "backend" / "data"))
            )
        return self._data_dir

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        if self._models_dir is None:
            self._models_dir = Path(
                os.getenv("MUSIC_DETECTOR_MODELS_DIR", str(self.data_dir / "models"))
            )
        return self._models_dir

    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"

    @property
    def temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = Path(
                os.getenv("MUSIC_DETECTOR_TEMP_DIR", str(self.base_dir / "backend" / "temp"))
            )
        return self._temp_dir

    @property
    def uploads_dir(self) -> Path:
        if self._uploads_dir is None:
            self._uploads_dir = Path(
                os.getenv("MUSIC_DETECTOR_UPLOADS_DIR", str(self.base_dir / "backend" / "uploads"))
            )
        return self._uploads_dir

    @property
    def ai_generated_dir(self) -> Path:
        return self.raw_dir / "ai_generated"

    @property
    def human_made_dir(self) -> Path:
        return self.raw_dir / "human_made"

    @property
    def sample_rate(self) -> int:
        return int(os.getenv("MUSIC_DETECTOR_SAMPLE_RATE", "22050"))

    @property
    def demucs_model(self) -> str:
        return os.getenv("MUSIC_DETECTOR_DEMUCS_MODEL", "htdemucs")

    @property
    def latest_model_path(self) -> Path:
        return self.models_dir / "latest_model.pkl"

    @property
    def latest_scaler_path(self) -> Path:
        return self.models_dir / "latest_scaler.pkl"

    @property
    def latest_metadata_path(self) -> Path:
        return self.models_dir / "latest_metadata.json"

    @property
    def features_file(self) -> Path:
        return self.processed_dir / "features.csv"

    @property
    def metadata_file(self) -> Path:
        return self.processed_dir / "metadata.json"

    def ensure_directories(self):
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
    """Get the application configuration singleton."""
    if not hasattr(get_config, "_instance"):
        get_config._instance = Config()
    return get_config._instance
