"""
Feature Extractor - AI vs Human müziği ayırt etmek için özellikler çıkarır

AI müziğin tipik özellikleri:
1. Aşırı düzgün tempo ve rhythm patterns
2. Çok mükemmel pitch alignment (insan hatası yok)
3. Harmony kurallarına aşırı uyum veya tamamen uyumsuzluk
4. Spectral features'da pattern tekrarı
5. Dinamik range çok dar veya çok geniş
6. Mikro-timing variations eksikliği (insan "groove" yok)
"""

import librosa
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class MusicFeatureExtractor:
    def __init__(self, sr=22050):
        self.sr = sr

    def extract_all_features(self, audio_path):
        """
        Tüm özellikleri çıkarır ve feature vector döndürür
        """
        # Ses dosyasını yükle
        y, sr = librosa.load(audio_path, sr=self.sr)

        features = {}

        # 1. TEMPO VE RHYTHM ANALİZİ
        tempo_features = self._extract_tempo_features(y, sr)
        features.update(tempo_features)

        # 2. PITCH VE HARMONY ANALİZİ
        pitch_features = self._extract_pitch_features(y, sr)
        features.update(pitch_features)

        # 3. SPECTRAL FEATURES
        spectral_features = self._extract_spectral_features(y, sr)
        features.update(spectral_features)

        # 4. TIMING VE MICRO-VARIATIONS
        timing_features = self._extract_timing_features(y, sr)
        features.update(timing_features)

        # 5. DYNAMIC RANGE VE LOUDNESS
        dynamic_features = self._extract_dynamic_features(y, sr)
        features.update(dynamic_features)

        # 6. HARMONIC-PERCUSSIVE SEPARATION ANALİZİ
        hp_features = self._extract_harmonic_percussive_features(y, sr)
        features.update(hp_features)

        return features

    def _extract_tempo_features(self, y, sr):
        """
        Tempo stabilitesi - AI çok düzgün, insan約間 varied
        """
        # Tempo ve beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Beat intervals (insan müziğinde daha varied olmalı)
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            tempo_stability = np.std(beat_intervals)  # AI'da çok düşük
            tempo_variance = np.var(beat_intervals)
            tempo_cv = stats.variation(beat_intervals) if len(beat_intervals) > 0 else 0
        else:
            tempo_stability = 0
            tempo_variance = 0
            tempo_cv = 0

        # Onset strength variation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_variation = np.std(onset_env)

        return {
            'tempo': tempo,
            'tempo_stability': tempo_stability,
            'tempo_variance': tempo_variance,
            'tempo_cv': tempo_cv,
            'onset_variation': onset_variation,
            'num_beats': len(beats)
        }

    def _extract_pitch_features(self, y, sr):
        """
        Pitch perfection analysis - AI çok perfect, insan約間 imperfect
        """
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Her frame için dominant pitch
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            pitch_values = np.array(pitch_values)

            # Pitch stability (AI çok stable)
            pitch_std = np.std(pitch_values)
            pitch_variance = np.var(pitch_values)

            # Pitch quantization (AI'da çok quantized olabilir)
            # Semitone'lara ne kadar yakın
            pitch_cents = 1200 * np.log2(pitch_values / 440)  # A4 referans
            semitone_deviation = np.mean(np.abs(pitch_cents % 100 - 50))

            # Vibrato detection (insan müziğinde olmalı)
            pitch_diff = np.diff(pitch_values)
            vibrato_strength = np.std(pitch_diff)

        else:
            pitch_std = 0
            pitch_variance = 0
            semitone_deviation = 0
            vibrato_strength = 0

        # Chroma features - harmony analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_std = np.std(chroma)
        chroma_mean = np.mean(chroma)

        # Tonnetz - tonal centroid features (harmony space)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_var = np.var(tonnetz)

        return {
            'pitch_std': pitch_std,
            'pitch_variance': pitch_variance,
            'semitone_deviation': semitone_deviation,
            'vibrato_strength': vibrato_strength,
            'chroma_std': chroma_std,
            'chroma_mean': chroma_mean,
            'tonnetz_variance': tonnetz_var
        }

    def _extract_spectral_features(self, y, sr):
        """
        Spectral analysis - AI patterns vs human organic sounds
        """
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_mean = np.mean(spectral_centroids)
        sc_std = np.std(spectral_centroids)

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast)
        contrast_std = np.std(spectral_contrast)

        # Spectral flatness (noise vs tonal)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_mean = np.mean(spectral_flatness)
        flatness_std = np.std(spectral_flatness)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # MFCC (timbre)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        features = {
            'spectral_centroid_mean': sc_mean,
            'spectral_centroid_std': sc_std,
            'spectral_rolloff_mean': sr_mean,
            'spectral_rolloff_std': sr_std,
            'spectral_contrast_mean': contrast_mean,
            'spectral_contrast_std': contrast_std,
            'spectral_flatness_mean': flatness_mean,
            'spectral_flatness_std': flatness_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
        }

        # MFCC features
        for i in range(13):
            features[f'mfcc_{i}_mean'] = mfcc_mean[i]
            features[f'mfcc_{i}_std'] = mfcc_std[i]

        return features

    def _extract_timing_features(self, y, sr):
        """
        Micro-timing analysis - insan müziğinde groove var
        """
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        if len(onset_times) > 1:
            # Inter-onset intervals (IOI)
            ioi = np.diff(onset_times)

            # IOI variance (insan daha varied, AI çok consistent)
            ioi_variance = np.var(ioi)
            ioi_cv = stats.variation(ioi) if len(ioi) > 0 else 0

            # Entropy of IOI (randomness measure)
            ioi_entropy = stats.entropy(np.histogram(ioi, bins=20)[0] + 1e-10)
        else:
            ioi_variance = 0
            ioi_cv = 0
            ioi_entropy = 0

        return {
            'ioi_variance': ioi_variance,
            'ioi_cv': ioi_cv,
            'ioi_entropy': ioi_entropy,
            'num_onsets': len(onset_frames)
        }

    def _extract_dynamic_features(self, y, sr):
        """
        Dynamic range ve loudness - AI çok compressed veya çok wide
        """
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_range = np.max(rms) - np.min(rms)

        # Dynamic range (dB)
        dynamic_range = 20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10))

        # Peak-to-average ratio
        peak_to_avg = np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10)

        return {
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'rms_range': rms_range,
            'dynamic_range': dynamic_range,
            'peak_to_avg_ratio': peak_to_avg
        }

    def _extract_harmonic_percussive_features(self, y, sr):
        """
        Harmonic-Percussive separation analysis
        """
        # Harmonic ve percussive component'leri ayır
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Harmonic/Percussive ratio
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        hp_ratio = harmonic_energy / (percussive_energy + 1e-10)

        # Harmonic consistency
        harmonic_std = np.std(y_harmonic)

        return {
            'hp_ratio': hp_ratio,
            'harmonic_energy': harmonic_energy,
            'percussive_energy': percussive_energy,
            'harmonic_std': harmonic_std
        }


if __name__ == "__main__":
    # Test
    extractor = MusicFeatureExtractor()

    test_file = "backend/data/raw/test_song.mp3"
    if os.path.exists(test_file):
        features = extractor.extract_all_features(test_file)
        print(f"Extracted {len(features)} features:")
        for key, value in list(features.items())[:10]:
            print(f"  {key}: {value:.4f}")
