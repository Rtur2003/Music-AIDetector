"""
Predictor - load trained model and classify new music files.
"""

import json
import joblib
import numpy as np
import shutil
from pathlib import Path
from uuid import uuid4

try:
    # Preferred when used as a package
    from .vocal_separator import VocalSeparator
    from .feature_extractor import MusicFeatureExtractor, FEATURE_EXTRACTOR_VERSION
except Exception:  # pragma: no cover - fallback for direct script usage
    from vocal_separator import VocalSeparator
    from feature_extractor import MusicFeatureExtractor, FEATURE_EXTRACTOR_VERSION


class MusicAIPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        """
        Args:
            model_path: Trained model path (None -> latest_model.pkl)
            scaler_path: Scaler path (None -> latest_scaler.pkl)
        """
        models_dir = Path("backend/data/models")

        if model_path is None:
            model_path = models_dir / "latest_model.pkl"
        if scaler_path is None:
            scaler_path = models_dir / "latest_scaler.pkl"

        # Load model and scaler
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model not found! Please train the model first.\n"
                f"Expected: {model_path}"
            )

        self.separator = VocalSeparator()
        self.extractor = MusicFeatureExtractor()

        # Read training metadata (feature ordering)
        self.feature_names = None
        self.metadata_version = None
        metadata_path = models_dir / "latest_metadata.json"
        if not metadata_path.exists() and model_path.name.startswith("model_"):
            ts = model_path.stem.replace("model_", "")
            candidate = models_dir / f"metadata_{ts}.json"
            if candidate.exists():
                metadata_path = candidate

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names")
                    self.metadata_version = metadata.get("feature_extractor_version")
            except Exception as err:
                print(f"Warning: Could not read metadata ({metadata_path}): {err}")

    def predict(self, audio_path, separate_vocals=True, return_features=False, temp_dir=None):
        """
        Predict whether a music file is AI or Human.

        Args:
            audio_path: Path to music file
            separate_vocals: Whether to run vocal separation first
            return_features: Also return extracted features
            temp_dir: Optional temp directory for stems (auto-created if omitted)
        """
        print(f"\nAnalyzing: {audio_path}")

        cleanup_dir = None

        try:
            # 1. Vocal separation
            if separate_vocals:
                print("  [1/3] Separating vocals...")
                session_dir = Path(temp_dir) if temp_dir else Path("backend/temp") / f"session_{uuid4().hex}"
                if temp_dir is None:
                    cleanup_dir = session_dir
                result = self.separator.separate(audio_path, session_dir)
                audio_to_analyze = result["instrumental"]
            else:
                audio_to_analyze = audio_path

            # 2. Feature extraction
            print("  [2/3] Extracting features...")
            features = self.extractor.extract_all_features(str(audio_to_analyze))

            # Order features according to training metadata if present
            if self.feature_names:
                missing = [name for name in self.feature_names if name not in features]
                if missing:
                    raise ValueError(f"Missing expected features in inference: {missing}")
                feature_values = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            else:
                feature_values = np.array(list(features.values())).reshape(1, -1)

            # 3. Prediction
            print("  [3/3] Making prediction...")
            feature_values_scaled = self.scaler.transform(feature_values)

            prediction = self.model.predict(feature_values_scaled)[0]
            probabilities = self.model.predict_proba(feature_values_scaled)[0]

            result = {
                "prediction": "AI" if prediction == 1 else "Human",
                "confidence": max(probabilities),
                "ai_probability": probabilities[1],
                "human_probability": probabilities[0],
            }

            if return_features:
                result["features"] = features

            return result

        finally:
            if cleanup_dir and cleanup_dir.exists():
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def predict_batch(self, audio_paths, separate_vocals=True):
        """
        Predict multiple files.
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, separate_vocals)
                result["file"] = str(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    "file": str(audio_path),
                    "error": str(e)
                })

        return results

    def explain_prediction(self, audio_path, separate_vocals=True):
        """
        Explain prediction and print important features.
        """
        result = self.predict(audio_path, separate_vocals, return_features=True)

        print("\n" + "=" * 60)
        print("PREDICTION EXPLANATION")
        print("=" * 60)
        print(f"File: {audio_path}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"  - AI probability: {result['ai_probability']:.2%}")
        print(f"  - Human probability: {result['human_probability']:.2%}")

        # Feature importance (if supported by model)
        if hasattr(self.model, "feature_importances_"):
            print("\nTop 10 Contributing Features:")
            importances = self.model.feature_importances_
            feature_names = list(result["features"].keys())

            top_indices = np.argsort(importances)[-10:][::-1]
            for idx in top_indices:
                feat_name = feature_names[idx]
                feat_value = result["features"][feat_name]
                importance = importances[idx]
                print(f"  {feat_name:30s}: {feat_value:12.4f} (importance: {importance:.4f})")

        print("=" * 60)

        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predictor.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Predictor oluştur
    predictor = MusicAIPredictor()

    # Tahmin yap ve açıkla
    result = predictor.explain_prediction(audio_file)
