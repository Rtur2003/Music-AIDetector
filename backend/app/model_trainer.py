"""
Model Trainer - AI vs Human müzik sınıflandırma modeli eğitir

Birden fazla algoritma dener:
1. Random Forest
2. XGBoost
3. SVM
4. Neural Network
5. Ensemble (voting)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


class MusicAIDetectorTrainer:
    def __init__(self, data_dir="backend/data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None

    def load_data(self):
        """
        İşlenmiş veri setini yükle
        """
        features_file = self.processed_dir / "features.csv"

        if not features_file.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_file}\n"
                "Run dataset_processor.py first!"
            )

        df = pd.read_csv(features_file)

        # Features ve labels ayır
        X = df.drop('label', axis=1)
        y = df['label']

        print(f"Dataset loaded: {len(df)} samples, {len(X.columns)} features")
        print(f"  AI samples: {sum(y)}")
        print(f"  Human samples: {len(y) - sum(y)}")

        return X, y

    def train_all_models(self, X, y, test_size=0.2):
        """
        Tüm modelleri eğit ve karşılaştır
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        results = {}

        # 1. Random Forest
        print("\n[1/5] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        results['Random Forest'] = self._evaluate_model(
            rf_model, X_test_scaled, y_test, "Random Forest"
        )

        # 2. XGBoost
        print("\n[2/5] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        results['XGBoost'] = self._evaluate_model(
            xgb_model, X_test_scaled, y_test, "XGBoost"
        )

        # 3. SVM
        print("\n[3/5] Training SVM...")
        svm_model = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train_scaled, y_train)
        results['SVM'] = self._evaluate_model(
            svm_model, X_test_scaled, y_test, "SVM"
        )

        # 4. Neural Network
        print("\n[4/5] Training Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        nn_model.fit(X_train_scaled, y_train)
        results['Neural Network'] = self._evaluate_model(
            nn_model, X_test_scaled, y_test, "Neural Network"
        )

        # 5. Ensemble (Voting)
        print("\n[5/5] Training Ensemble (Voting)...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('svm', svm_model),
                ('nn', nn_model)
            ],
            voting='soft'
        )
        ensemble_model.fit(X_train_scaled, y_train)
        results['Ensemble'] = self._evaluate_model(
            ensemble_model, X_test_scaled, y_test, "Ensemble"
        )

        # En iyi modeli seç
        best_accuracy = 0
        for model_name, metrics in results.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                self.best_model_name = model_name

        if self.best_model_name == 'Random Forest':
            self.best_model = rf_model
        elif self.best_model_name == 'XGBoost':
            self.best_model = xgb_model
        elif self.best_model_name == 'SVM':
            self.best_model = svm_model
        elif self.best_model_name == 'Neural Network':
            self.best_model = nn_model
        else:
            self.best_model = ensemble_model

        # Feature importance (eğer varsa)
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

        # Summary
        self._print_summary(results)

        return results, X_test_scaled, y_test

    def _evaluate_model(self, model, X_test, y_test, model_name):
        """
        Model performansını değerlendir
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"{model_name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def _print_summary(self, results):
        """
        Tüm modellerin özeti
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        summary_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [r['accuracy'] for r in results.values()],
            'Precision': [r['precision'] for r in results.values()],
            'Recall': [r['recall'] for r in results.values()],
            'F1': [r['f1'] for r in results.values()],
            'AUC': [r['auc'] for r in results.values()]
        })

        print(summary_df.to_string(index=False))
        print(f"\nBest Model: {self.best_model_name}")

    def save_model(self):
        """
        En iyi modeli kaydet
        """
        if self.best_model is None:
            raise ValueError("No model trained yet!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Model kaydet
        model_file = self.models_dir / f"model_{timestamp}.pkl"
        scaler_file = self.models_dir / f"scaler_{timestamp}.pkl"

        joblib.dump(self.best_model, model_file)
        joblib.dump(self.scaler, scaler_file)

        # Latest symlink
        latest_model = self.models_dir / "latest_model.pkl"
        latest_scaler = self.models_dir / "latest_scaler.pkl"

        # Windows'da copy kullan (symlink yerine)
        import shutil
        shutil.copy(model_file, latest_model)
        shutil.copy(scaler_file, latest_scaler)

        # Metadata kaydet
        metadata = {
            'model_name': self.best_model_name,
            'timestamp': timestamp,
            'model_file': str(model_file),
            'scaler_file': str(scaler_file)
        }

        if self.feature_importance is not None:
            metadata['top_features'] = self.feature_importance.head(10).to_dict('records')

        metadata_file = self.models_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel saved:")
        print(f"  Model: {model_file}")
        print(f"  Scaler: {scaler_file}")
        print(f"  Metadata: {metadata_file}")

        return model_file, scaler_file

    def plot_feature_importance(self, top_n=20):
        """
        Feature importance plot
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model")
            return

        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {self.best_model_name}')
        plt.tight_layout()

        plot_file = self.models_dir / "feature_importance.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nFeature importance plot saved: {plot_file}")

        plt.close()


def main():
    """
    Ana eğitim pipeline
    """
    trainer = MusicAIDetectorTrainer()

    # Veri yükle
    X, y = trainer.load_data()

    # Tüm modelleri eğit
    results, X_test, y_test = trainer.train_all_models(X, y)

    # Modeli kaydet
    trainer.save_model()

    # Feature importance plot
    if trainer.feature_importance is not None:
        trainer.plot_feature_importance()
        print("\nTop 10 Most Important Features:")
        print(trainer.feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
