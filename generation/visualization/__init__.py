"""Visualization utilities for metrics and explanations."""

import json
from typing import Any, Dict, List
import numpy as np


class PlotGenerator:
    """Generate plot specifications for frontend rendering."""

    @staticmethod
    def generate_confusion_matrix_plot(confusion_matrix: List[List[int]], classes: List[str]) -> Dict[str, Any]:
        """Generate heatmap plot spec for confusion matrix."""
        return {
            "type": "heatmap",
            "title": "Confusion Matrix",
            "data": confusion_matrix,
            "x_labels": classes,
            "y_labels": classes,
            "colorscale": "Viridis",
        }

    @staticmethod
    def generate_residual_plot(residuals: List[float], predictions: List[float]) -> Dict[str, Any]:
        """Generate scatter plot of residuals vs predicted values."""
        return {
            "type": "scatter",
            "title": "Residual Plot",
            "x": predictions,
            "y": residuals,
            "x_label": "Predicted Value",
            "y_label": "Residuals",
            "hline": 0,
        }

    @staticmethod
    def generate_actual_vs_predicted_plot(actual: List[float], predicted: List[float]) -> Dict[str, Any]:
        """Generate scatter plot of actual vs predicted."""
        # Compute R² for display
        actual_arr = np.asarray(actual)
        pred_arr = np.asarray(predicted)
        ss_res = np.sum((actual_arr - pred_arr) ** 2)
        ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "type": "scatter",
            "title": f"Actual vs Predicted (R² = {r2:.4f})",
            "x": actual,
            "y": predicted,
            "x_label": "Actual Value",
            "y_label": "Predicted Value",
        }

    @staticmethod
    def generate_feature_importance_plot(
        importance_scores: List[float],
        feature_names: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Generate bar chart for feature importance."""
        # Get top k
        scores = np.asarray(importance_scores)
        names = np.asarray(feature_names)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        
        return {
            "type": "bar",
            "title": f"Top {top_k} Feature Importances",
            "x": names[top_idx].tolist(),
            "y": scores[top_idx].tolist(),
            "x_label": "Feature",
            "y_label": "Importance Score",
        }

    @staticmethod
    def generate_shap_plot(
        shap_values: List[float],
        feature_values: List[float],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate SHAP force plot data."""
        return {
            "type": "shap_force",
            "title": "SHAP Force Plot",
            "shap_values": shap_values,
            "feature_values": feature_values,
            "feature_names": feature_names,
        }

    @staticmethod
    def generate_time_series_plot(
        actual: List[float],
        predicted: List[float],
        time_index: List[int]
    ) -> Dict[str, Any]:
        """Generate time series comparison plot."""
        return {
            "type": "line",
            "title": "Time Series Forecast",
            "series": [
                {
                    "name": "Actual",
                    "x": time_index,
                    "y": actual,
                    "mode": "lines",
                    "line": {"color": "blue"},
                },
                {
                    "name": "Predicted",
                    "x": time_index,
                    "y": predicted,
                    "mode": "lines",
                    "line": {"color": "red", "dash": "dash"},
                },
            ],
            "x_label": "Time",
            "y_label": "Value",
        }

    @staticmethod
    def generate_clustering_plot(
        coordinates: List[List[float]],
        labels: List[int],
        n_clusters: int
    ) -> Dict[str, Any]:
        """Generate scatter plot for clusters."""
        coords = np.asarray(coordinates)
        
        return {
            "type": "scatter",
            "title": f"Clusters ({n_clusters} clusters)",
            "x": coords[:, 0].tolist() if coords.shape[1] > 0 else [],
            "y": coords[:, 1].tolist() if coords.shape[1] > 1 else [],
            "labels": labels,
            "colorscale": "Viridis",
        }

    @staticmethod
    def generate_roc_curve_plot(fpr: List[float], tpr: List[float], auc: float) -> Dict[str, Any]:
        """Generate ROC curve plot."""
        return {
            "type": "line",
            "title": f"ROC Curve (AUC = {auc:.4f})",
            "x": fpr,
            "y": tpr,
            "x_label": "False Positive Rate",
            "y_label": "True Positive Rate",
        }


class MetricsFormatter:
    """Format metrics for display and API responses."""

    @staticmethod
    def format_metrics_for_display(metrics: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Format metrics dict for human-readable display.
        """
        formatted = {}
        
        # Organize by category
        if task_type == "regression":
            formatted["Error Metrics"] = {
                k: v for k, v in metrics.items() if k in ["mae", "mse", "rmse", "mape"]
            }
            formatted["Goodness of Fit"] = {
                k: v for k, v in metrics.items() if k in ["r2"]
            }
        elif task_type == "classification":
            formatted["Overall"] = {
                k: v for k, v in metrics.items() if k in ["accuracy", "n_classes"]
            }
            formatted["Macro-averaged"] = {
                k: v for k, v in metrics.items() if "macro" in k
            }
            formatted["Weighted-averaged"] = {
                k: v for k, v in metrics.items() if "weighted" in k and "macro" not in k
            }
            formatted["Additional"] = {
                k: v for k, v in metrics.items() if "roc_auc" in k
            }
        elif task_type == "clustering":
            formatted["Quality Metrics"] = {
                k: v for k, v in metrics.items() 
                if k in ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_index"]
            }
            formatted["Structure"] = {
                k: v for k, v in metrics.items() if k in ["n_clusters", "cluster_sizes"]
            }
        elif task_type == "time_series":
            formatted["Accuracy"] = {
                k: v for k, v in metrics.items() if k in ["mae", "rmse", "mape"]
            }
            formatted["Direction"] = {
                k: v for k, v in metrics.items() if k in ["direction_accuracy", "mase"]
            }
        
        return formatted

    @staticmethod
    def format_explanation_for_display(explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Format explanation for display."""
        formatted = {
            "type": explanation.get("type"),
            "prediction": explanation.get("prediction"),
        }
        
        if explanation.get("type") == "feature_importance":
            formatted["top_features"] = explanation.get("feature_names", [])[:10]
            formatted["scores"] = explanation.get("importance_scores", [])[:10]
        
        elif explanation.get("type") == "shap":
            shap_vals = explanation.get("shap_values", [])
            feature_names = explanation.get("feature_names", [])
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
            formatted["top_features"] = [feature_names[i] for i in top_indices]
            formatted["shap_values"] = [shap_vals[i] for i in top_indices]
        
        elif explanation.get("type") == "lime":
            coefs = explanation.get("coefficients", [])
            feature_names = explanation.get("feature_names", [])
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(coefs))[-10:][::-1]
            formatted["top_features"] = [feature_names[i] for i in top_indices]
            formatted["coefficients"] = [coefs[i] for i in top_indices]
        
        return formatted
