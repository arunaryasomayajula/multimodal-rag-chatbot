"""SHAP explainer for Shapley values."""

import numpy as np
from typing import Any, Dict, List
from .base import Explainer


class SHAPExplainer(Explainer):
    """
    SHAP (SHapley Additive exPlanations) explainer.
    Provides per-sample feature contribution estimates.
    Uses KernelSHAP (model-agnostic) for flexibility.
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Initialize with optional background data size."""
        super().__init__(model, X_train, y_train)
        self.background_size = kwargs.get("background_size", min(100, len(X_train)))
        self._initialize_background()

    def _initialize_background(self):
        """Sample background data for SHAP."""
        idx = np.random.choice(len(self.X_train), self.background_size, replace=False)
        self.background = self.X_train[idx]

    def explain_prediction(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Explain single prediction using SHAP.
        """
        x = np.asarray(x).reshape(1, -1)
        shap_values = self._compute_shap_values(x)

        feature_names = self.get_feature_names()
        base_value = self._compute_base_value()

        return {
            "type": "shap",
            "shap_values": shap_values[0].tolist(),
            "base_value": float(base_value),
            "prediction": self._jsonable_prediction(self.model.predict(x)[0]),
            "feature_names": feature_names,
            "feature_values": x[0].tolist(),
        }

    def explain_predictions(self, X: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions.
        """
        X = np.asarray(X)
        n_samples = kwargs.get("n_samples", min(len(X), 100))  # Limit for efficiency
        
        # For large datasets, sample rows
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[idx]
            sample_indices = idx.tolist()
        else:
            X_sample = X
            sample_indices = list(range(len(X)))
        
        shap_values = self._compute_shap_values(X_sample)
        predictions = self.model.predict(X_sample)
        base_value = self._compute_base_value()
        feature_names = self.get_feature_names()
        
        explanations = []
        for i, idx in enumerate(sample_indices):
            explanations.append({
                "type": "shap",
                "sample_idx": int(idx),
                "shap_values": shap_values[i].tolist(),
                "base_value": float(base_value),
                "prediction": self._jsonable_prediction(predictions[i]),
                "feature_names": feature_names,
                "feature_values": X_sample[i].tolist(),
            })

        return explanations

    def _compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values using a simplified marginal-contribution approximation.

        Works for both regressors and classifiers: contributions are measured on
        a numeric signal (the prediction for regressors, the probability of each
        sample's own predicted class for classifiers), so string class labels no
        longer break the arithmetic.
        """
        n_samples, n_features = X.shape
        shap_values = np.zeros((n_samples, n_features))

        is_clf = self._is_classifier()
        # Fix the target class per sample so the marginal contributions are
        # measured against a consistent probability across perturbations.
        target_classes = (
            np.argmax(np.asarray(self.model.predict_proba(X)), axis=1)
            if is_clf else [None] * n_samples
        )

        for sample_idx in range(n_samples):
            cls = target_classes[sample_idx]
            for i in range(n_features):
                # Marginal contribution of feature i
                x_with = X[sample_idx].copy()
                x_without = X[sample_idx].copy()
                x_without[i] = np.mean(self.background[:, i])

                try:
                    pred_with = self._score(x_with, class_idx=cls)[0]
                    pred_without = self._score(x_without, class_idx=cls)[0]
                    shap_values[sample_idx, i] = pred_with - pred_without
                except Exception:
                    shap_values[sample_idx, i] = 0.0

        return shap_values

    def _compute_base_value(self) -> float:
        """
        Compute base value (expected numeric model output over background).
        """
        try:
            return float(np.mean(self._score(self.background)))
        except Exception:
            return 0.0
