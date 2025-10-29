"""
Predictor - Eğitilmiş model ile yeni müzikleri tahmin eder
"""

import joblib
import numpy as np
from pathlib import Path
from vocal_separator import VocalSeparator
from feature_extractor import MusicFeatureExtractor


class MusicAIPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        """
        Args:
            model_path: Trained model path (None ise latest kullanılır)
            scaler_path: Scaler path (None ise latest kullanılır)
        """
        models_dir = Path("backend/data/models")

        if model_path is None:
            model_path = models_dir / "latest_model.pkl"
        if scaler_path is None:
            scaler_path = models_dir / "latest_scaler.pkl"

        # Model ve scaler yükle
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

    def predict(self, audio_path, separate_vocals=True, return_features=False):
        """
        Bir müzik dosyasının AI mi Human mi olduğunu tahmin et

        Args:
            audio_path: Müzik dosyası yolu
            separate_vocals: Vocal'leri ayır mı?
            return_features: Feature'ları da döndür mü?

        Returns:
            dict: {
                'prediction': 'AI' veya 'Human',
                'confidence': 0-1 arası güven skoru,
                'ai_probability': AI olma olasılığı,
                'human_probability': Human olma olasılığı,
                'features': (opsiyonel) çıkarılan feature'lar
            }
        """
        print(f"\nAnalyzing: {audio_path}")

        # 1. Vocal separation
        if separate_vocals:
            print("  [1/3] Separating vocals...")
            result = self.separator.separate(audio_path, "backend/temp")
            audio_to_analyze = result['instrumental']
        else:
            audio_to_analyze = audio_path

        # 2. Feature extraction
        print("  [2/3] Extracting features...")
        features = self.extractor.extract_all_features(str(audio_to_analyze))

        # Convert to array
        feature_values = np.array(list(features.values())).reshape(1, -1)

        # 3. Prediction
        print("  [3/3] Making prediction...")
        feature_values_scaled = self.scaler.transform(feature_values)

        prediction = self.model.predict(feature_values_scaled)[0]
        probabilities = self.model.predict_proba(feature_values_scaled)[0]

        result = {
            'prediction': 'AI' if prediction == 1 else 'Human',
            'confidence': max(probabilities),
            'ai_probability': probabilities[1],
            'human_probability': probabilities[0],
        }

        if return_features:
            result['features'] = features

        return result

    def predict_batch(self, audio_paths, separate_vocals=True):
        """
        Birden fazla dosya için tahmin
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, separate_vocals)
                result['file'] = str(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'file': str(audio_path),
                    'error': str(e)
                })

        return results

    def explain_prediction(self, audio_path, separate_vocals=True):
        """
        Tahmini açıkla - hangi feature'lar önemli
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

        # Feature importance (eğer model destekliyorsa)
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop 10 Contributing Features:")
            importances = self.model.feature_importances_
            feature_names = list(result['features'].keys())

            top_indices = np.argsort(importances)[-10:][::-1]
            for idx in top_indices:
                feat_name = feature_names[idx]
                feat_value = result['features'][feat_name]
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
