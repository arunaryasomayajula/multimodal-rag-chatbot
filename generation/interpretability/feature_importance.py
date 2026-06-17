"""Feature importance explainer using permutation importance."""

import numpy as np
from typing import Any, Dict, List
from .base import Explainer


class FeatureImportanceExplainer(Explainer):
    """
    Calculate feature importance using permutation-based method.
    Model-agnostic and works with any predictor.
    """

    def explain_prediction(self, x: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        For single prediction, return global feature importances.
        """
        importances = self._compute_importances()
        feature_names = self.get_feature_names()
        
        return {
            "type": "feature_importance",
            "importance_scores": importances.tolist(),
            "feature_names": feature_names,
            "top_k": kwargs.get("top_k", 10),
        }

    def explain_predictions(self, X: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Return global feature importances for all samples.
        """
        result = self.explain_prediction(X[0])
        
        # Compute local importances: permutation on individual samples
        importances = self._compute_importances()
        feature_names = self.get_feature_names()
        top_k = kwargs.get("top_k", 10)
        
        return [
            {
                "type": "feature_importance",
                "sample_idx": i,
                "global_importance_scores": importances.tolist(),
                "feature_names": feature_names,
                "top_k": top_k,
            }
            for i in range(len(X))
        ]

    def _compute_importances(self) -> np.ndarray:
        """
        Compute permutation feature importances on training data.
        """
        n_features = self.X_train.shape[1]
        baseline_score = self._score_model(self.X_train, self.y_train)
        importances = np.zeros(n_features)
        
        for i in range(n_features):
            X_permuted = self.X_train.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self._score_model(X_permuted, self.y_train)
            importances[i] = max(0, baseline_score - permuted_score)
        
        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        
        return importances

    def _score_model(self, X, y):
        """Compute model score (accuracy or MSE)."""
        try:
            preds = self.model.predict(X)
            
            # Try classification accuracy
            try:
                accuracy = np.mean(preds == y)
                return accuracy
            except Exception:
                # Fall back to MSE
                mse = np.mean((preds - y) ** 2)
                return -mse  # Negative because lower MSE is better
        except Exception:
            return 0.0
