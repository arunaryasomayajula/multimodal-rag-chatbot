"""Time series metrics calculator."""

import numpy as np
from typing import Any, Dict
from .base import MetricsCalculator


class TimeSeriesMetrics(MetricsCalculator):
    """Calculate time series forecasting performance metrics."""

    def calculate(self, y_true, y_pred, seasonal_period=None, **kwargs) -> Dict[str, Any]:
        """
        Calculate time series metrics.
        
        Metrics:
        - MAE, RMSE, MAPE (same as regression)
        - SMAPE: Symmetric MAPE
        - MASE: Mean Absolute Scaled Error (normalized by in-sample MAE)
        - Directional Accuracy: % of correct direction changes
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        if len(y_true) == 0:
            return {}
        
        # Basic regression metrics
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # MAPE
        mask = y_true != 0
        if np.any(mask):
            mape = float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        else:
            mape = 0.0
        
        # SMAPE: Symmetric MAPE
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = float(100 * np.mean(2 * np.abs(y_true - y_pred) / (denominator + 1e-10)))
        
        # Directional Accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = float(np.mean(true_direction == pred_direction)) * 100
        else:
            direction_accuracy = 0.0
        
        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "smape": round(smape, 4),
            "direction_accuracy": round(direction_accuracy, 2),
        }
        
        # MASE (requires seasonal period hint)
        if seasonal_period is not None and len(y_true) > seasonal_period:
            try:
                mase = self._compute_mase(y_true, y_pred, seasonal_period)
                metrics["mase"] = round(mase, 4)
            except Exception:
                pass
        
        return metrics

    def get_visualization_data(self, y_true, y_pred, **kwargs) -> Dict[str, Any]:
        """
        Return time series data for visualization.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        return {
            "plot_type": "time_series_forecast",
            "actual": y_true.tolist(),
            "predicted": y_pred.tolist(),
            "time_index": list(range(len(y_true))),
        }

    @staticmethod
    def _compute_mase(y_true, y_pred, seasonal_period):
        """Compute Mean Absolute Scaled Error."""
        # Compute in-sample MAE using naive seasonal forecast
        if len(y_true) > seasonal_period:
            naive_errors = np.abs(
                y_true[seasonal_period:] - y_true[:-seasonal_period]
            )
            denominator = np.mean(naive_errors)
        else:
            denominator = np.mean(np.abs(np.diff(y_true)))
        
        mae = np.mean(np.abs(y_true - y_pred))
        return mae / (denominator + 1e-10)
