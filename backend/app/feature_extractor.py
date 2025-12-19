"""
Feature Extractor - extracts numeric descriptors for AI vs Human music detection.
"""

import os
import warnings
import librosa
import numpy as np
from scipy import stats

try:
    from .config import get_config
except ImportError:
    from config import get_config

warnings.filterwarnings("ignore")

# Increment when feature definitions change
FEATURE_EXTRACTOR_VERSION = "1.0.0"


class MusicFeatureExtractor:
    def __init__(self, sr=None):
        cfg = get_config()
        self.sr = sr if sr is not None else cfg.sample_rate
        self.eps = 1e-10

    def extract_all_features(self, audio_path):
        """
        Extract all features and return a flat feature dict.
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)

        features = {}

        # 1. Tempo & rhythm
        features.update(self._extract_tempo_features(y, sr))

        # 2. Pitch & harmony
        features.update(self._extract_pitch_features(y, sr))

        # 3. Spectral
        features.update(self._extract_spectral_features(y, sr))

        # 4. Timing / micro-variations
        features.update(self._extract_timing_features(y, sr))

        # 5. Dynamics
        features.update(self._extract_dynamic_features(y, sr))

        # 6. Harmonic-percussive
        features.update(self._extract_harmonic_percussive_features(y, sr))

        return features

    def _extract_tempo_features(self, y, sr):
        """
        Tempo stability and onset variation.
        """
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            tempo_stability = np.std(beat_intervals)
            tempo_variance = np.var(beat_intervals)
            tempo_cv = stats.variation(beat_intervals) if len(beat_intervals) > 0 else 0.0
        else:
            tempo_stability = 0.0
            tempo_variance = 0.0
            tempo_cv = 0.0

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_variation = np.std(onset_env)

        return {
            "tempo": float(tempo),
            "tempo_stability": float(tempo_stability),
            "tempo_variance": float(tempo_variance),
            "tempo_cv": float(tempo_cv),
            "onset_variation": float(onset_variation),
            "num_beats": int(len(beats)),
        }

    def _extract_pitch_features(self, y, sr):
        """
        Pitch stability and harmony descriptors.
        """
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            pitch_values = np.array(pitch_values)

            pitch_std = np.std(pitch_values)
            pitch_variance = np.var(pitch_values)

            pitch_cents = 1200 * np.log2(pitch_values / 440.0)
            semitone_deviation = np.mean(np.abs(pitch_cents % 100 - 50))

            pitch_diff = np.diff(pitch_values)
            vibrato_strength = np.std(pitch_diff)
        else:
            pitch_std = 0.0
            pitch_variance = 0.0
            semitone_deviation = 0.0
            vibrato_strength = 0.0

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_std = np.std(chroma)
        chroma_mean = np.mean(chroma)

        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_var = np.var(tonnetz)

        return {
            "pitch_std": float(pitch_std),
            "pitch_variance": float(pitch_variance),
            "semitone_deviation": float(semitone_deviation),
            "vibrato_strength": float(vibrato_strength),
            "chroma_std": float(chroma_std),
            "chroma_mean": float(chroma_mean),
            "tonnetz_variance": float(tonnetz_var),
        }

    def _extract_spectral_features(self, y, sr):
        """
        Spectral descriptors (timbre, brightness, noisiness).
        """
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_mean = np.mean(spectral_centroids)
        sc_std = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast)
        contrast_std = np.std(spectral_contrast)

        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_mean = np.mean(spectral_flatness)
        flatness_std = np.std(spectral_flatness)

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        features = {
            "spectral_centroid_mean": float(sc_mean),
            "spectral_centroid_std": float(sc_std),
            "spectral_rolloff_mean": float(sr_mean),
            "spectral_rolloff_std": float(sr_std),
            "spectral_contrast_mean": float(contrast_mean),
            "spectral_contrast_std": float(contrast_std),
            "spectral_flatness_mean": float(flatness_mean),
            "spectral_flatness_std": float(flatness_std),
            "zcr_mean": float(zcr_mean),
            "zcr_std": float(zcr_std),
        }

        for i in range(13):
            features[f"mfcc_{i}_mean"] = float(mfcc_mean[i])
            features[f"mfcc_{i}_std"] = float(mfcc_std[i])

        return features

    def _extract_timing_features(self, y, sr):
        """
        Micro-timing analysis (groove/variation).
        """
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        if len(onset_times) > 1:
            ioi = np.diff(onset_times)
            ioi_variance = np.var(ioi)
            ioi_cv = stats.variation(ioi) if len(ioi) > 0 else 0.0
            ioi_entropy = stats.entropy(np.histogram(ioi, bins=20)[0] + self.eps)
        else:
            ioi_variance = 0.0
            ioi_cv = 0.0
            ioi_entropy = 0.0

        return {
            "ioi_variance": float(ioi_variance),
            "ioi_cv": float(ioi_cv),
            "ioi_entropy": float(ioi_entropy),
            "num_onsets": int(len(onset_frames)),
        }

    def _extract_dynamic_features(self, y, sr):
        """
        Dynamic range and loudness.
        """
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_range = np.max(rms) - np.min(rms)

        peak = float(np.max(np.abs(y)))
        mean_abs = float(np.mean(np.abs(y)) + self.eps)
        peak_safe = max(peak, self.eps)

        dynamic_range = 20 * np.log10(peak_safe / mean_abs)
        peak_to_avg = peak_safe / mean_abs

        return {
            "rms_mean": float(rms_mean),
            "rms_std": float(rms_std),
            "rms_range": float(rms_range),
            "dynamic_range": float(dynamic_range),
            "peak_to_avg_ratio": float(peak_to_avg),
        }

    def _extract_harmonic_percussive_features(self, y, sr):
        """
        Harmonic-percussive separation analysis.
        """
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))

        hp_ratio = harmonic_energy / (percussive_energy + self.eps)
        harmonic_std = float(np.std(y_harmonic))

        return {
            "hp_ratio": float(hp_ratio),
            "harmonic_energy": harmonic_energy,
            "percussive_energy": percussive_energy,
            "harmonic_std": harmonic_std,
        }


if __name__ == "__main__":
    extractor = MusicFeatureExtractor()

    test_file = "backend/data/raw/test_song.mp3"
    if os.path.exists(test_file):
        features = extractor.extract_all_features(test_file)
        print(f"Extracted {len(features)} features:")
        for key, value in list(features.items())[:10]:
            print(f"  {key}: {value:.4f}")
