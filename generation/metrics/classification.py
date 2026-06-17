"""Classification metrics calculator."""

import numpy as np
from typing import Any, Dict, List
from .base import MetricsCalculator


class ClassificationMetrics(MetricsCalculator):
    """Calculate classification performance metrics."""

    def calculate(self, y_true, y_pred, y_proba=None, **kwargs) -> Dict[str, Any]:
        """
        Calculate classification metrics.
        
        Metrics:
        - Accuracy
        - Precision, Recall, F1 (macro, weighted)
        - Confusion matrix
        - ROC-AUC (if y_proba provided and binary)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if len(y_true) == 0:
            return {}
        
        # Accuracy
        accuracy = float(np.mean(y_true == y_pred))
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        # Build confusion matrix
        confusion_matrix = self._build_confusion_matrix(y_true, y_pred, classes)
        
        # Per-class metrics
        tp = np.diag(confusion_matrix)
        fp = confusion_matrix.sum(axis=0) - tp
        fn = confusion_matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Macro and weighted averages
        macro_precision = float(np.mean(precision))
        macro_recall = float(np.mean(recall))
        macro_f1 = float(np.mean(f1))
        
        # Support = true instances per class, taken from the confusion-matrix rows
        # so it is position-aligned with tp/precision/recall and works for any
        # label dtype (strings, floats, non-contiguous ints).
        support = confusion_matrix.sum(axis=1).astype(float)
        if support.sum() > 0:
            weighted_precision = float(np.average(precision, weights=support))
            weighted_recall = float(np.average(recall, weights=support))
            weighted_f1 = float(np.average(f1, weights=support))
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        metrics = {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_precision, 4),
            "weighted_recall": round(weighted_recall, 4),
            "weighted_f1": round(weighted_f1, 4),
            "n_classes": int(n_classes),
        }
        
        # ROC-AUC for binary classification
        if n_classes == 2 and y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                y_proba = np.asarray(y_proba)
                if y_proba.ndim > 1:
                    y_proba = y_proba[:, 1]
                roc_auc = float(roc_auc_score(y_true, y_proba))
                metrics["roc_auc"] = round(roc_auc, 4)
            except Exception:
                pass
        
        return metrics

    def get_visualization_data(self, y_true, y_pred, y_proba=None, **kwargs) -> Dict[str, Any]:
        """
        Return confusion matrix and per-class metrics for visualization.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        confusion_matrix = self._build_confusion_matrix(y_true, y_pred, classes)
        
        return {
            "plot_type": "classification_analysis",
            "confusion_matrix": confusion_matrix.tolist(),
            "classes": classes.tolist(),
            "actual": y_true.tolist(),
            "predicted": y_pred.tolist(),
        }

    @staticmethod
    def _build_confusion_matrix(y_true, y_pred, classes):
        """Build confusion matrix."""
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for true, pred in zip(y_true, y_pred):
            i = class_to_idx.get(true, -1)
            j = class_to_idx.get(pred, -1)
            if i >= 0 and j >= 0:
                matrix[i, j] += 1
        
        return matrix
