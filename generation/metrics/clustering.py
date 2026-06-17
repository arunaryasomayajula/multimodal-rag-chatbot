"""Clustering metrics calculator."""

import numpy as np
from typing import Any, Dict
from .base import MetricsCalculator


class ClusteringMetrics(MetricsCalculator):
    """Calculate clustering performance metrics."""

    def calculate(self, X, labels, **kwargs) -> Dict[str, Any]:
        """
        Calculate clustering metrics.
        
        Metrics:
        - Silhouette Score (requires ground truth for supervised)
        - Davies-Bouldin Index
        - Calinski-Harabasz Index
        """
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        
        if len(X) == 0 or len(np.unique(labels)) < 2:
            return {}
        
        metrics = {}
        
        # Silhouette Score
        try:
            from sklearn.metrics import silhouette_score
            sil_score = float(silhouette_score(X, labels))
            metrics["silhouette_score"] = round(sil_score, 4)
        except Exception:
            pass
        
        # Davies-Bouldin Index (lower is better)
        try:
            db_index = self._davies_bouldin_index(X, labels)
            metrics["davies_bouldin_index"] = round(db_index, 4)
        except Exception:
            pass
        
        # Calinski-Harabasz Index (higher is better)
        try:
            from sklearn.metrics import calinski_harabasz_score
            ch_index = float(calinski_harabasz_score(X, labels))
            metrics["calinski_harabasz_index"] = round(ch_index, 4)
        except Exception:
            pass
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics["n_clusters"] = int(len(unique))
        metrics["cluster_sizes"] = counts.tolist()
        
        return metrics

    def get_visualization_data(self, X, labels, **kwargs) -> Dict[str, Any]:
        """
        Return cluster assignments and (if 2D/3D) coordinates for visualization.
        """
        X = np.asarray(X)
        labels = np.asarray(labels)
        
        # Try PCA for 2D visualization if high-dimensional
        if X.shape[1] > 2:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_vis = pca.fit_transform(X)
            except Exception:
                X_vis = X[:, :2] if X.shape[1] >= 2 else X
        else:
            X_vis = X
        
        return {
            "plot_type": "clustering_analysis",
            "coordinates": X_vis.tolist(),
            "labels": labels.tolist(),
            "n_clusters": int(len(np.unique(labels))),
        }

    @staticmethod
    def _davies_bouldin_index(X, labels):
        """Compute Davies-Bouldin Index."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters == 0 or n_clusters == 1:
            return 0.0
        
        # Compute centroids
        centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
        
        # Compute average distances to centroid
        avg_distances = np.zeros(n_clusters)
        for i, label in enumerate(unique_labels):
            avg_distances[i] = np.mean(np.linalg.norm(X[labels == label] - centroids[i], axis=1))
        
        # Compute Davies-Bouldin Index
        db_index = 0.0
        for i in range(n_clusters):
            max_ratio = 0.0
            for j in range(n_clusters):
                if i != j:
                    centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                    ratio = (avg_distances[i] + avg_distances[j]) / (centroid_distance + 1e-10)
                    max_ratio = max(max_ratio, ratio)
            db_index += max_ratio
        
        return db_index / n_clusters
