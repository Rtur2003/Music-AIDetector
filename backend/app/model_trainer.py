"""
Model Trainer - trains AI vs Human music classifier with multiple algorithms.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import xgboost as xgb

try:
    from .feature_extractor import FEATURE_EXTRACTOR_VERSION
except Exception:  # pragma: no cover - fallback for direct script usage
    from feature_extractor import FEATURE_EXTRACTOR_VERSION


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
        self.feature_names = None

    def load_data(self):
        """
        Load processed dataset.
        """
        features_file = self.processed_dir / "features.csv"

        if not features_file.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_file}\n"
                "Run dataset_processor.py first!"
            )

        df = pd.read_csv(features_file)

        # Features and labels
        X = df.drop("label", axis=1)
        y = df["label"]
        self.feature_names = list(X.columns)

        print(f"Dataset loaded: {len(df)} samples, {len(X.columns)} features")
        print(f"  AI samples: {sum(y)}")
        print(f"  Human samples: {len(y) - sum(y)}")

        # Basic class balance check
        if y.nunique() < 2:
            raise ValueError("Dataset must contain at least two classes (AI and Human).")
        if min(y.value_counts()) < 2:
            raise ValueError("Each class must have at least 2 samples for a split.")

        return X, y

    def train_all_models(self, X, y, test_size=0.2, cv_folds=5):
        """
        Train and compare multiple models. Optionally run stratified CV.
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
        cv_results = {}
        cv_enabled = y.value_counts().min() >= cv_folds
        if not cv_enabled:
            print(f"Skipping cross-validation (need at least {cv_folds} samples per class).")

        # 1. Random Forest
        print("\n[1/5] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train_scaled, y_train)
        results["Random Forest"] = self._evaluate_model(
            rf_model, X_test_scaled, y_test, "Random Forest"
        )
        if cv_enabled:
            cv_results["Random Forest"] = self._cross_validate_model(rf_model, X, y, cv_folds)

        # 2. XGBoost
        print("\n[2/5] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train_scaled, y_train)
        results["XGBoost"] = self._evaluate_model(
            xgb_model, X_test_scaled, y_test, "XGBoost"
        )
        if cv_enabled:
            cv_results["XGBoost"] = self._cross_validate_model(xgb_model, X, y, cv_folds)

        # 3. SVM
        print("\n[3/5] Training SVM...")
        svm_model = SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42,
        )
        svm_model.fit(X_train_scaled, y_train)
        results["SVM"] = self._evaluate_model(
            svm_model, X_test_scaled, y_test, "SVM"
        )
        if cv_enabled:
            cv_results["SVM"] = self._cross_validate_model(svm_model, X, y, cv_folds)

        # 4. Neural Network
        print("\n[4/5] Training Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
        )
        nn_model.fit(X_train_scaled, y_train)
        results["Neural Network"] = self._evaluate_model(
            nn_model, X_test_scaled, y_test, "Neural Network"
        )
        if cv_enabled:
            cv_results["Neural Network"] = self._cross_validate_model(nn_model, X, y, cv_folds)

        # 5. Ensemble (Voting)
        print("\n[5/5] Training Ensemble (Voting)...")
        ensemble_model = VotingClassifier(
            estimators=[
                ("rf", rf_model),
                ("xgb", xgb_model),
                ("svm", svm_model),
                ("nn", nn_model),
            ],
            voting="soft",
        )
        ensemble_model.fit(X_train_scaled, y_train)
        results["Ensemble"] = self._evaluate_model(
            ensemble_model, X_test_scaled, y_test, "Ensemble"
        )
        # CV for ensemble is costly; keep optional
        if cv_enabled:
            cv_results["Ensemble"] = self._cross_validate_model(ensemble_model, X, y, cv_folds)

        # Choose best model
        best_accuracy = -1
        for model_name, metrics in results.items():
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                self.best_model_name = model_name

        if self.best_model_name == "Random Forest":
            self.best_model = rf_model
        elif self.best_model_name == "XGBoost":
            self.best_model = xgb_model
        elif self.best_model_name == "SVM":
            self.best_model = svm_model
        elif self.best_model_name == "Neural Network":
            self.best_model = nn_model
        else:
            self.best_model = ensemble_model

        # Feature importance (if available)
        if hasattr(self.best_model, "feature_importances_"):
            self.feature_importance = pd.DataFrame({
                "feature": X.columns,
                "importance": self.best_model.feature_importances_,
            }).sort_values("importance", ascending=False)

        # Summary
        self._print_summary(results, cv_results if cv_enabled else None)

        return results, X_test_scaled, y_test

    def _evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model performance.
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = None

        print(f"{model_name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}" if auc is not None else "  AUC:       N/A")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

    def _cross_validate_model(self, model, X, y, cv_folds):
        """
        Stratified k-fold cross-validation (accuracy).
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        print(f"  CV ({cv_folds} folds) accuracy: {mean_score:.4f} ± {std_score:.4f}")
        return {"mean": mean_score, "std": std_score, "folds": cv_folds}

    def _print_summary(self, results, cv_results=None):
        """
        Summary of all models.
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        summary_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [r["accuracy"] for r in results.values()],
            "Precision": [r["precision"] for r in results.values()],
            "Recall": [r["recall"] for r in results.values()],
            "F1": [r["f1"] for r in results.values()],
            "AUC": [r["auc"] for r in results.values()],
        })

        print(summary_df.to_string(index=False))
        print(f"\nBest Model: {self.best_model_name}")

        if cv_results:
            print("\nCross-Validation (accuracy)")
            cv_df = pd.DataFrame({
                "Model": list(cv_results.keys()),
                "CV_Mean": [r["mean"] for r in cv_results.values()],
                "CV_Std": [r["std"] for r in cv_results.values()],
                "Folds": [r["folds"] for r in cv_results.values()],
            })
            print(cv_df.to_string(index=False))

    def save_model(self):
        """
        Save the best model and scaler + metadata.
        """
        if self.best_model is None:
            raise ValueError("No model trained yet!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_file = self.models_dir / f"model_{timestamp}.pkl"
        scaler_file = self.models_dir / f"scaler_{timestamp}.pkl"

        joblib.dump(self.best_model, model_file)
        joblib.dump(self.scaler, scaler_file)

        # Latest copies (Windows friendly)
        import shutil
        latest_model = self.models_dir / "latest_model.pkl"
        latest_scaler = self.models_dir / "latest_scaler.pkl"
        shutil.copy(model_file, latest_model)
        shutil.copy(scaler_file, latest_scaler)

        # Metadata
        metadata = {
            "model_name": self.best_model_name,
            "timestamp": timestamp,
            "model_file": str(model_file),
            "scaler_file": str(scaler_file),
            "feature_names": self.feature_names,
            "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        }

        if self.feature_importance is not None:
            metadata["top_features"] = self.feature_importance.head(10).to_dict("records")

        metadata_file = self.models_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Keep a "latest" metadata for inference alignment
        latest_metadata = self.models_dir / "latest_metadata.json"
        shutil.copy(metadata_file, latest_metadata)

        print(f"\nModel saved:")
        print(f"  Model: {model_file}")
        print(f"  Scaler: {scaler_file}")
        print(f"  Metadata: {metadata_file}")

        return model_file, scaler_file

    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance.
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model")
            return

        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)

        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importance - {self.best_model_name}")
        plt.tight_layout()

        plot_file = self.models_dir / "feature_importance.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"\nFeature importance plot saved: {plot_file}")

        plt.close()


def main():
    """
    Main training pipeline.
    """
    trainer = MusicAIDetectorTrainer()

    # Load data
    X, y = trainer.load_data()

    # Train all models
    results, X_test, y_test = trainer.train_all_models(X, y)

    # Save model
    trainer.save_model()

    # Feature importance plot
    if trainer.feature_importance is not None:
        trainer.plot_feature_importance()
        print("\nTop 10 Most Important Features:")
        print(trainer.feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
