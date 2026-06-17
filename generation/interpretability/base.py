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
