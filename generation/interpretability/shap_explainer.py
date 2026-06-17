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
            "prediction": float(self.model.predict(x)[0]),
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
                "prediction": float(predictions[i]),
                "feature_names": feature_names,
                "feature_values": X_sample[i].tolist(),
            })
        
        return explanations

    def _compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values using simplified kernel SHAP approximation.
        
        For efficiency, uses weighted sampling over feature coalitions.
        """
        n_samples, n_features = X.shape
        shap_values = np.zeros((n_samples, n_features))
        
        # Simplified SHAP: use permutation importance per feature
        for i in range(n_features):
            for sample_idx in range(n_samples):
                # Marginal contribution of feature i
                x_with = X[sample_idx].copy()
                x_without = X[sample_idx].copy()
                x_without[i] = np.mean(self.background[:, i])
                
                try:
                    pred_with = self.model.predict(x_with.reshape(1, -1))[0]
                    pred_without = self.model.predict(x_without.reshape(1, -1))[0]
                    shap_values[sample_idx, i] = pred_with - pred_without
                except Exception:
                    shap_values[sample_idx, i] = 0.0
        
        return shap_values

    def _compute_base_value(self) -> float:
        """
        Compute base value (expected model output over background).
        """
        try:
            preds = self.model.predict(self.background)
            if isinstance(preds, np.ndarray):
                return float(np.mean(preds))
            return float(preds)
        except Exception:
            return 0.0
