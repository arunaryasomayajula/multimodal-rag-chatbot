"""LIME explainer for local model-agnostic explanations."""

import numpy as np
from typing import Any, Dict, List
from .base import Explainer


class LIMEExplainer(Explainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations).
    Fits local linear model to explain predictions.
    """

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Initialize LIME explainer."""
        super().__init__(model, X_train, y_train)
        self.kernel_width = kwargs.get("kernel_width", 0.25)
        self.n_samples = kwargs.get("n_samples", 1000)
        self._compute_stats()

    def _compute_stats(self):
        """Compute mean and std for normalization."""
        self.mean = np.mean(self.X_train, axis=0)
        self.std = np.std(self.X_train, axis=0) + 1e-10

    def explain_prediction(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Explain single prediction using LIME.
        """
        x = np.asarray(x).reshape(1, -1)
        coefficients = self._fit_local_model(x[0])
        
        feature_names = self.get_feature_names()
        
        return {
            "type": "lime",
            "coefficients": coefficients.tolist(),
            "feature_names": feature_names,
            "feature_values": x[0].tolist(),
            "prediction": float(self.model.predict(x)[0]),
            "intercept": 0.0,  # Included in coefficients
        }

    def explain_predictions(self, X: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions using LIME.
        """
        X = np.asarray(X)
        n_samples_to_explain = kwargs.get("n_samples", min(len(X), 50))
        
        # Sample rows to explain
        if len(X) > n_samples_to_explain:
            idx = np.random.choice(len(X), n_samples_to_explain, replace=False)
            X_sample = X[idx]
            sample_indices = idx.tolist()
        else:
            X_sample = X
            sample_indices = list(range(len(X)))
        
        predictions = self.model.predict(X_sample)
        feature_names = self.get_feature_names()
        
        explanations = []
        for i, idx in enumerate(sample_indices):
            coefficients = self._fit_local_model(X_sample[i])
            explanations.append({
                "type": "lime",
                "sample_idx": int(idx),
                "coefficients": coefficients.tolist(),
                "feature_names": feature_names,
                "feature_values": X_sample[i].tolist(),
                "prediction": float(predictions[i]),
            })
        
        return explanations

    def _fit_local_model(self, x: np.ndarray) -> np.ndarray:
        """
        Fit local linear model around instance x.
        
        1. Generate perturbed samples around x
        2. Compute distances (weights)
        3. Get predictions from model
        4. Fit weighted linear regression
        """
        # Generate perturbed samples
        X_perturbed = self._generate_perturbed_samples(x)
        
        # Get model predictions
        y_perturbed = self.model.predict(X_perturbed)
        
        # Compute kernel distances
        distances = self._compute_kernel_distance(x, X_perturbed)
        weights = np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))
        
        # Fit weighted linear regression
        X_perturbed_norm = (X_perturbed - self.mean) / self.std
        
        try:
            # Weighted least squares
            W = np.diag(weights)
            XtW = X_perturbed_norm.T @ W
            coefficients = np.linalg.solve(XtW @ X_perturbed_norm + 1e-6 * np.eye(X_perturbed_norm.shape[1]),
                                           XtW @ y_perturbed)
        except Exception:
            # Fallback to unweighted
            coefficients = np.linalg.lstsq(X_perturbed_norm, y_perturbed, rcond=None)[0]
        
        return coefficients

    def _generate_perturbed_samples(self, x: np.ndarray, n_samples: int = None) -> np.ndarray:
        """
        Generate perturbed samples by randomly toggling features.
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        n_features = len(x)
        X_perturbed = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            # Randomly perturb each feature
            perturbation = np.random.normal(0, 0.1, n_features)
            X_perturbed[i] = x + perturbation * self.std
        
        return X_perturbed

    def _compute_kernel_distance(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance in normalized space.
        """
        x_norm = (x - self.mean) / self.std
        X_norm = (X - self.mean) / self.std
        
        distances = np.linalg.norm(X_norm - x_norm, axis=1)
        return distances
