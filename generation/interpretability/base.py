"""Base class for explainers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np


class Explainer(ABC):
    """Abstract base for model explainers."""

    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialize explainer.
        
        Args:
            model: Fitted model with predict method
            X_train: Training features
            y_train: Training labels/values
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

    @abstractmethod
    def explain_prediction(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Explain individual prediction.
        
        Args:
            x: Single sample to explain (1D array)
            **kwargs: Method-specific parameters
            
        Returns:
            Dict with explanation data
        """
        pass

    @abstractmethod
    def explain_predictions(self, X: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions.
        
        Args:
            X: Multiple samples (2D array)
            
        Returns:
            List of explanation dicts
        """
        pass

    def get_feature_names(self):
        """Get feature names if available (real column names if provided)."""
        names = getattr(self, "feature_names", None)
        if names is not None and len(names) == self.X_train.shape[1]:
            return list(names)
        return [f"feature_{i}" for i in range(self.X_train.shape[1])]

    # ── Numeric prediction signal ────────────────────────────────────────
    # SHAP/LIME need a numeric output to differentiate. For regressors that is
    # the prediction itself; for classifiers (string/categorical labels) we use
    # the probability of a chosen class, which is always numeric.

    def _is_classifier(self) -> bool:
        return hasattr(self.model, "predict_proba")

    def _instance_target_class(self, x: np.ndarray) -> int:
        """Index (into predict_proba columns) of the predicted class for x."""
        proba = self.model.predict_proba(np.asarray(x).reshape(1, -1))[0]
        return int(np.argmax(proba))

    def _score(self, X: np.ndarray, class_idx=None) -> np.ndarray:
        """
        Numeric model output for rows of X.

        - Regressor: the raw prediction (float).
        - Classifier: probability of ``class_idx`` (defaults to each row's own
          predicted class) — a continuous, differentiable signal SHAP/LIME can use.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self._is_classifier():
            proba = np.asarray(self.model.predict_proba(X))
            if class_idx is None:
                return proba[np.arange(len(proba)), np.argmax(proba, axis=1)]
            return proba[:, class_idx]
        return np.asarray(self.model.predict(X), dtype=float)

    @staticmethod
    def _jsonable_prediction(value) -> Any:
        """Return a JSON-serializable prediction (float for numbers, str for labels)."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
