import tempfile
import unittest
import numpy as np
import soundfile as sf

# Allow running tests from repo root
try:
    from backend.app.feature_extractor import MusicFeatureExtractor
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "backend" / "app"))
    from feature_extractor import MusicFeatureExtractor


class FeatureExtractorSmokeTest(unittest.TestCase):
    def test_extract_all_features_on_sine_wave(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.2 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            extractor = MusicFeatureExtractor(sr=sr)
            features = extractor.extract_all_features(tmp.name)

        # Basic expectations
        self.assertGreater(len(features), 10)
        self.assertTrue(all(np.isfinite(list(features.values()))))
        self.assertIn("tempo", features)
        self.assertIn("spectral_centroid_mean", features)


if __name__ == "__main__":
    unittest.main()
