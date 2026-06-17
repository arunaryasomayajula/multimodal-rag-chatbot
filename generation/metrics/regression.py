"""Regression metrics calculator."""

import numpy as np
from typing import Any, Dict
from .base import MetricsCalculator


class RegressionMetrics(MetricsCalculator):
    """Calculate regression performance metrics."""

    def calculate(self, y_true, y_pred, **kwargs) -> Dict[str, Any]:
        """
        Calculate regression metrics.
        
        Metrics:
        - MAE: Mean Absolute Error
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - R²: Coefficient of determination
        - MAPE: Mean Absolute Percentage Error
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        if len(y_true) == 0:
            return {}
        
        # MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # MSE
        mse = float(np.mean((y_true - y_pred) ** 2))
        
        # RMSE
        rmse = float(np.sqrt(mse))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        # MAPE - handle division by zero
        mask = y_true != 0
        if np.any(mask):
            mape = float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        else:
            mape = 0.0
        
        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 4),
        }

    def get_visualization_data(self, y_true, y_pred, **kwargs) -> Dict[str, Any]:
        """
        Return data for residual plot and predicted vs actual.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = (y_true - y_pred).tolist()
        
        return {
            "plot_type": "regression_analysis",
            "actual": y_true.tolist(),
            "predicted": y_pred.tolist(),
            "residuals": residuals,
            "x_indices": list(range(len(y_true))),
        }
