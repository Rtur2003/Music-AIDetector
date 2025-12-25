"""
Predictor - load trained model and classify new music files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

try:
    from .config import get_config
    from .feature_extractor import FEATURE_EXTRACTOR_VERSION, FeatureExtractionError, MusicFeatureExtractor
    from .logging_config import get_logger
    from .utils import ResourceManager
    from .validators import ValidationError
    from .vocal_separator import VocalSeparator, VocalSeparatorError
except Exception:
    from config import get_config
    from feature_extractor import FEATURE_EXTRACTOR_VERSION, FeatureExtractionError, MusicFeatureExtractor
    from logging_config import get_logger
    from utils import ResourceManager
    from validators import ValidationError
    from vocal_separator import VocalSeparator, VocalSeparatorError

logger = get_logger(__name__)


class PredictionError(Exception):
    """Exception raised for prediction errors."""

    pass


class MusicAIPredictor:
    """Predicts whether music is AI-generated or human-made."""

    def __init__(
        self, model_path: Optional[str] = None, scaler_path: Optional[str] = None
    ):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Trained model path (None -> latest_model.pkl)
            scaler_path: Scaler path (None -> latest_scaler.pkl)

        Raises:
            PredictionError: If model cannot be loaded
        """
        cfg = get_config()
        models_dir = cfg.models_dir

        if model_path is None:
            model_path = cfg.latest_model_path
        if scaler_path is None:
            scaler_path = cfg.latest_scaler_path

        # Load model and scaler
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Model loaded from: {model_path}")
        except FileNotFoundError as e:
            raise PredictionError(
                f"Model not found! Please train the model first.\n"
                f"Expected: {model_path}"
            ) from e
        except Exception as e:
            raise PredictionError(f"Failed to load model: {e}") from e

        self.separator = VocalSeparator()
        self.extractor = MusicFeatureExtractor()

        # Read training metadata (feature ordering)
        self.feature_names = None
        self.metadata_version = None
        metadata_path = cfg.latest_metadata_path
        if not metadata_path.exists() and str(model_path).startswith("model_"):
            ts = Path(model_path).stem.replace("model_", "")
            candidate = models_dir / f"metadata_{ts}.json"
            if candidate.exists():
                metadata_path = candidate

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names")
                    self.metadata_version = metadata.get("feature_extractor_version")
                logger.info(f"Metadata loaded from: {metadata_path}")
            except Exception as err:
                logger.warning(f"Could not read metadata ({metadata_path}): {err}")

    def predict(
        self,
        audio_path: str,
        separate_vocals: bool = True,
        return_features: bool = False,
        temp_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict whether a music file is AI or Human.

        Args:
            audio_path: Path to music file
            separate_vocals: Whether to run vocal separation first
            return_features: Also return extracted features
            temp_dir: Optional temp directory for stems (auto-created if omitted)

        Returns:
            Dictionary with prediction results

        Raises:
            PredictionError: If prediction fails
            ValidationError: If audio file is invalid
        """
        logger.info(f"Analyzing: {audio_path}")

        cfg = get_config()

        try:
            if self.metadata_version and self.metadata_version != FEATURE_EXTRACTOR_VERSION:
                raise PredictionError(
                    f"Feature extractor version mismatch: "
                    f"model expects {self.metadata_version}, runtime {FEATURE_EXTRACTOR_VERSION}"
                )

            # Use context manager for temp directory
            if separate_vocals:
                logger.debug("Separating vocals...")
                with ResourceManager.temporary_directory(
                    base_dir=temp_dir, prefix="prediction_"
                ) as session_dir:
                    result = self.separator.separate(audio_path, str(session_dir))
                    audio_to_analyze = result["instrumental"]
                    
                    # Extract features while temp dir exists
                    logger.debug("Extracting features...")
                    features = self.extractor.extract_all_features(str(audio_to_analyze))
            else:
                audio_to_analyze = audio_path
                logger.debug("Extracting features...")
                features = self.extractor.extract_all_features(str(audio_to_analyze))

            # Order features according to training metadata if present
            if self.feature_names:
                missing = [name for name in self.feature_names if name not in features]
                if missing:
                    raise PredictionError(f"Missing expected features in inference: {missing}")
                feature_values = np.array(
                    [features[name] for name in self.feature_names]
                ).reshape(1, -1)
            else:
                feature_values = np.array(list(features.values())).reshape(1, -1)

            # Prediction
            logger.debug("Making prediction...")
            feature_values_scaled = self.scaler.transform(feature_values)

            prediction = self.model.predict(feature_values_scaled)[0]
            probabilities = self.model.predict_proba(feature_values_scaled)[0]

            result = {
                "prediction": "AI" if prediction == 1 else "Human",
                "confidence": float(max(probabilities)),
                "ai_probability": float(probabilities[1]),
                "human_probability": float(probabilities[0]),
            }

            if return_features:
                result["features"] = features

            logger.info(
                f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})"
            )

            return result

        except (ValidationError, VocalSeparatorError, FeatureExtractionError):
            raise
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}") from e

    def predict_batch(
        self, audio_paths: List[str], separate_vocals: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict multiple files.

        Args:
            audio_paths: List of audio file paths
            separate_vocals: Whether to perform vocal separation

        Returns:
            List of prediction results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, separate_vocals)
                result["file"] = str(audio_path)
                results.append(result)
            except (PredictionError, ValidationError) as e:
                logger.error(f"Error processing {audio_path}: {e}")
                results.append({"file": str(audio_path), "error": str(e)})
            except Exception as e:
                logger.error(f"Unexpected error processing {audio_path}: {e}")
                results.append({"file": str(audio_path), "error": str(e)})

        return results

    def explain_prediction(self, audio_path: str, separate_vocals: bool = True) -> Dict[str, Any]:
        """
        Explain prediction and show important features.

        Args:
            audio_path: Path to audio file
            separate_vocals: Whether to perform vocal separation

        Returns:
            Prediction result with feature explanations

        Raises:
            PredictionError: If prediction fails
        """
        result = self.predict(audio_path, separate_vocals, return_features=True)

        logger.info("=" * 60)
        logger.info("PREDICTION EXPLANATION")
        logger.info("=" * 60)
        logger.info(f"File: {audio_path}")
        if self.metadata_version:
            logger.info(f"Feature extractor version (model): {self.metadata_version}")
            logger.info(f"Feature extractor version (runtime): {FEATURE_EXTRACTOR_VERSION}")
        logger.info(f"Prediction: {result['prediction']}")
        logger.info(f"Confidence: {result['confidence']:.2%}")
        logger.info(f"  - AI probability: {result['ai_probability']:.2%}")
        logger.info(f"  - Human probability: {result['human_probability']:.2%}")

        # Feature importance (if supported by model)
        if hasattr(self.model, "feature_importances_"):
            logger.info("\nTop 10 Contributing Features:")
            importances = self.model.feature_importances_
            feature_names = list(result["features"].keys())

            top_indices = np.argsort(importances)[-10:][::-1]
            for idx in top_indices:
                feat_name = feature_names[idx]
                feat_value = result["features"][feat_name]
                importance = importances[idx]
                logger.info(
                    f"  {feat_name:30s}: {feat_value:12.4f} (importance: {importance:.4f})"
                )

        logger.info("=" * 60)

        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python predictor.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    try:
        # Create predictor
        predictor = MusicAIPredictor()

        # Make prediction and explain
        result = predictor.explain_prediction(audio_file)

    except PredictionError as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
