"""
Feature Extractor - extracts numeric descriptors for AI vs Human music detection.
"""

import warnings
from typing import Dict

import librosa
import numpy as np
from scipy import stats

try:
    from .config import get_config
    from .logging_config import get_logger
    from .validators import AudioValidator, ValidationError
except ImportError:
    from config import get_config
    from logging_config import get_logger
    from validators import AudioValidator, ValidationError

warnings.filterwarnings("ignore")

# Increment when feature definitions change
FEATURE_EXTRACTOR_VERSION = "1.0.0"

logger = get_logger(__name__)


class FeatureExtractionError(Exception):
    """Exception raised for feature extraction errors."""

    pass


class MusicFeatureExtractor:
    """Extracts audio features for music classification."""

    def __init__(self, sr: int = None):
        """
        Initialize feature extractor.

        Args:
            sr: Sample rate (None = use config default)
        """
        cfg = get_config()
        self.sr = sr if sr is not None else cfg.sample_rate
        self.eps = 1e-10
        logger.info(f"FeatureExtractor initialized with sr={self.sr}")

    def extract_all_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract all features from an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary of feature names to values

        Raises:
            ValidationError: If audio file is invalid
            FeatureExtractionError: If extraction fails
        """
        try:
            # Validate audio file
            audio_path = AudioValidator.validate_audio_file(audio_path)
            logger.debug(f"Extracting features from: {audio_path}")

            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sr)

            features = {}

            # Extract feature groups
            features.update(self._extract_tempo_features(y, sr))
            features.update(self._extract_pitch_features(y, sr))
            features.update(self._extract_spectral_features(y, sr))
            features.update(self._extract_timing_features(y, sr))
            features.update(self._extract_dynamic_features(y, sr))
            features.update(self._extract_harmonic_percussive_features(y, sr))

            logger.debug(f"Extracted {len(features)} features")
            return features

        except ValidationError:
            raise
        except Exception as e:
            error_msg = f"Feature extraction failed for {audio_path}: {e}"
            logger.error(error_msg)
            raise FeatureExtractionError(error_msg) from e

    def _extract_tempo_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract tempo and rhythm features.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of tempo features
        """
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)

            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                tempo_stability = float(np.std(beat_intervals))
                tempo_variance = float(np.var(beat_intervals))
                tempo_cv = float(stats.variation(beat_intervals)) if len(beat_intervals) > 0 else 0.0
            else:
                tempo_stability = 0.0
                tempo_variance = 0.0
                tempo_cv = 0.0

            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_variation = float(np.std(onset_env))

            return {
                "tempo": float(tempo),
                "tempo_stability": tempo_stability,
                "tempo_variance": tempo_variance,
                "tempo_cv": tempo_cv,
                "onset_variation": onset_variation,
                "num_beats": int(len(beats)),
            }
        except Exception as e:
            logger.warning(f"Tempo feature extraction warning: {e}")
            return {
                "tempo": 0.0,
                "tempo_stability": 0.0,
                "tempo_variance": 0.0,
                "tempo_cv": 0.0,
                "onset_variation": 0.0,
                "num_beats": 0,
            }

    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract pitch and harmony features.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of pitch features
        """
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) > 0:
                pitch_values = np.array(pitch_values)

                pitch_std = float(np.std(pitch_values))
                pitch_variance = float(np.var(pitch_values))

                pitch_cents = 1200 * np.log2(pitch_values / 440.0)
                semitone_deviation = float(np.mean(np.abs(pitch_cents % 100 - 50)))

                pitch_diff = np.diff(pitch_values)
                vibrato_strength = float(np.std(pitch_diff))
            else:
                pitch_std = 0.0
                pitch_variance = 0.0
                semitone_deviation = 0.0
                vibrato_strength = 0.0

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_std = float(np.std(chroma))
            chroma_mean = float(np.mean(chroma))

            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_var = float(np.var(tonnetz))

            return {
                "pitch_std": pitch_std,
                "pitch_variance": pitch_variance,
                "semitone_deviation": semitone_deviation,
                "vibrato_strength": vibrato_strength,
                "chroma_std": chroma_std,
                "chroma_mean": chroma_mean,
                "tonnetz_variance": tonnetz_var,
            }
        except Exception as e:
            logger.warning(f"Pitch feature extraction warning: {e}")
            return {
                "pitch_std": 0.0,
                "pitch_variance": 0.0,
                "semitone_deviation": 0.0,
                "vibrato_strength": 0.0,
                "chroma_std": 0.0,
                "chroma_mean": 0.0,
                "tonnetz_variance": 0.0,
            }

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral features (timbre, brightness, noisiness).

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of spectral features
        """
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            sc_mean = float(np.mean(spectral_centroids))
            sc_std = float(np.std(spectral_centroids))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            sr_mean = float(np.mean(spectral_rolloff))
            sr_std = float(np.std(spectral_rolloff))

            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = float(np.mean(spectral_contrast))
            contrast_std = float(np.std(spectral_contrast))

            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            flatness_mean = float(np.mean(spectral_flatness))
            flatness_std = float(np.std(spectral_flatness))

            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_std = float(np.std(zcr))

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            features = {
                "spectral_centroid_mean": sc_mean,
                "spectral_centroid_std": sc_std,
                "spectral_rolloff_mean": sr_mean,
                "spectral_rolloff_std": sr_std,
                "spectral_contrast_mean": contrast_mean,
                "spectral_contrast_std": contrast_std,
                "spectral_flatness_mean": flatness_mean,
                "spectral_flatness_std": flatness_std,
                "zcr_mean": zcr_mean,
                "zcr_std": zcr_std,
            }

            for i in range(13):
                features[f"mfcc_{i}_mean"] = float(mfcc_mean[i])
                features[f"mfcc_{i}_std"] = float(mfcc_std[i])

            return features

        except Exception as e:
            logger.warning(f"Spectral feature extraction warning: {e}")
            # Return defaults for 36 spectral features
            features = {
                "spectral_centroid_mean": 0.0,
                "spectral_centroid_std": 0.0,
                "spectral_rolloff_mean": 0.0,
                "spectral_rolloff_std": 0.0,
                "spectral_contrast_mean": 0.0,
                "spectral_contrast_std": 0.0,
                "spectral_flatness_mean": 0.0,
                "spectral_flatness_std": 0.0,
                "zcr_mean": 0.0,
                "zcr_std": 0.0,
            }
            for i in range(13):
                features[f"mfcc_{i}_mean"] = 0.0
                features[f"mfcc_{i}_std"] = 0.0
            return features

    def _extract_timing_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract micro-timing features (groove/variation).

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of timing features
        """
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            if len(onset_times) > 1:
                ioi = np.diff(onset_times)
                ioi_variance = float(np.var(ioi))
                ioi_cv = float(stats.variation(ioi)) if len(ioi) > 0 else 0.0
                ioi_entropy = float(stats.entropy(np.histogram(ioi, bins=20)[0] + self.eps))
            else:
                ioi_variance = 0.0
                ioi_cv = 0.0
                ioi_entropy = 0.0

            return {
                "ioi_variance": ioi_variance,
                "ioi_cv": ioi_cv,
                "ioi_entropy": ioi_entropy,
                "num_onsets": int(len(onset_frames)),
            }
        except Exception as e:
            logger.warning(f"Timing feature extraction warning: {e}")
            return {
                "ioi_variance": 0.0,
                "ioi_cv": 0.0,
                "ioi_entropy": 0.0,
                "num_onsets": 0,
            }

    def _extract_dynamic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract dynamic range and loudness features.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of dynamic features
        """
        try:
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            rms_range = float(np.max(rms) - np.min(rms))

            peak = float(np.max(np.abs(y)))
            mean_abs = float(np.mean(np.abs(y)) + self.eps)
            peak_safe = max(peak, self.eps)

            dynamic_range = float(20 * np.log10(peak_safe / mean_abs))
            peak_to_avg = float(peak_safe / mean_abs)

            return {
                "rms_mean": rms_mean,
                "rms_std": rms_std,
                "rms_range": rms_range,
                "dynamic_range": dynamic_range,
                "peak_to_avg_ratio": peak_to_avg,
            }
        except Exception as e:
            logger.warning(f"Dynamic feature extraction warning: {e}")
            return {
                "rms_mean": 0.0,
                "rms_std": 0.0,
                "rms_range": 0.0,
                "dynamic_range": 0.0,
                "peak_to_avg_ratio": 1.0,
            }

    def _extract_harmonic_percussive_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract harmonic-percussive separation features.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary of harmonic-percussive features
        """
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            harmonic_energy = float(np.sum(y_harmonic**2))
            percussive_energy = float(np.sum(y_percussive**2))

            hp_ratio = float(harmonic_energy / (percussive_energy + self.eps))
            harmonic_std = float(np.std(y_harmonic))

            return {
                "hp_ratio": hp_ratio,
                "harmonic_energy": harmonic_energy,
                "percussive_energy": percussive_energy,
                "harmonic_std": harmonic_std,
            }
        except Exception as e:
            logger.warning(f"Harmonic-percussive feature extraction warning: {e}")
            return {
                "hp_ratio": 1.0,
                "harmonic_energy": 0.0,
                "percussive_energy": 0.0,
                "harmonic_std": 0.0,
            }


if __name__ == "__main__":
    from pathlib import Path

    extractor = MusicFeatureExtractor()

    test_file = "backend/data/raw/test_song.mp3"
    if Path(test_file).exists():
        try:
            features = extractor.extract_all_features(test_file)
            logger.info(f"Extracted {len(features)} features:")
            for key, value in list(features.items())[:10]:
                logger.info(f"  {key}: {value:.4f}")
        except (ValidationError, FeatureExtractionError) as e:
            logger.error(f"Feature extraction failed: {e}")
    else:
        logger.warning(f"Test file not found: {test_file}")

