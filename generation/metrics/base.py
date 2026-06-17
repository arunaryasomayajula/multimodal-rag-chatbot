"""Base class for metric calculators."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class MetricsCalculator(ABC):
    """Abstract base for metrics calculators."""

    @abstractmethod
    def calculate(self, y_true, y_pred, **kwargs) -> Dict[str, Any]:
        """
        Calculate metrics for predictions.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            **kwargs: Task-specific parameters
            
        Returns:
            Dictionary of metric names to values
        """
        pass

    @abstractmethod
    def get_visualization_data(self, y_true, y_pred, **kwargs) -> Dict[str, Any]:
        """
        Get data suitable for visualization.
        
        Returns:
            Dict with 'plot_type' and data for rendering
        """
        pass
