"""
TABICLv2 wrapper for zero-shot tabular prediction with comprehensive metrics and interpretability.
Uses tabicl.TabICLClassifier / TabICLRegressor (sklearn-compatible).
Includes metrics calculators, interpretability (SHAP/LIME/Feature Importance), and visualization support.
"""
import re
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from db.session import SessionLocal
from db.models import TabularDataset
from generation.metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    TimeSeriesMetrics,
    ClusteringMetrics,
)
from generation.interpretability import (
    FeatureImportanceExplainer,
    SHAPExplainer,
    LIMEExplainer,
)
from generation.visualization import PlotGenerator, MetricsFormatter


def _load_dataset_df(dataset_id: str, user_id: str) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, filename). Raises ValueError if not found."""
    with SessionLocal() as db:
        row = db.query(TabularDataset).filter(
            TabularDataset.id == dataset_id,
            TabularDataset.user_id == user_id,
        ).first()
        if row is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        filename = row.filename

    from config import settings
    import pathlib
    path = pathlib.Path(settings.upload_dir) / filename
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")
    return df, filename


def _detect_task_type(series: pd.Series) -> str:
    """Auto-detect task type: classification or regression."""
    if series.dtype == object or series.nunique() <= 20:
        return "classification"
    return "regression"


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns numerically for model input."""
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = df_enc[col].astype("category").cat.codes
    return df_enc


def run_prediction(
    dataset_id: str,
    user_id: str,
    target_column: str,
    context_rows: int = 50,
    task_type: str = "auto",
    include_metrics: bool = True,
    include_interpretability: bool = False,
    interpretability_methods: Optional[List[str]] = None,
    n_samples_explain: int = 100,
    drop_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run tabular prediction with comprehensive metrics and optional interpretability.

    Args:
        dataset_id: ID of uploaded dataset
        user_id: User ID for access control
        target_column: Column to predict
        context_rows: Number of rows for in-context learning
        task_type: "auto", "classification", "regression", "time_series", or "clustering"
        include_metrics: Whether to compute metrics
        include_interpretability: Whether to compute interpretability
        interpretability_methods: List of ["feature_importance", "shap", "lime"] (all if not specified)
        n_samples_explain: Number of samples to explain (for efficiency)
        drop_columns: Columns to exclude from the feature set (e.g. ID columns).
            The target column is never dropped even if listed.

    Returns:
        Dict with predictions, metrics, explanations, and visualization data
    """
    df, filename = _load_dataset_df(dataset_id, user_id)

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not in dataset")

    # Clean data
    df = df.dropna(subset=[target_column])

    # Drop user-excluded columns (e.g. ID columns) from the feature set.
    dropped_columns = []
    if drop_columns:
        dropped_columns = [c for c in drop_columns if c in df.columns and c != target_column]
        if dropped_columns:
            df = df.drop(columns=dropped_columns)

    X = df.drop(columns=[target_column])
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after dropping columns; keep at least one.")
    y = df[target_column]

    # Encode categoricals
    X_enc = _encode_categoricals(X)
    feature_names = list(X_enc.columns)

    # Detect task type if auto
    if task_type == "auto":
        task_type = _detect_task_type(y)

    # Split: first context_rows as in-context set, rest as test
    n = len(X_enc)
    ctx_n = min(context_rows, max(1, n // 2))
    X_ctx, y_ctx = X_enc.iloc[:ctx_n].values, y.iloc[:ctx_n].values
    X_test, y_test = X_enc.iloc[ctx_n:].values, y.iloc[ctx_n:].values

    # Initialize result
    result = {
        "task_type": task_type,
        "target_column": target_column,
        "filename": filename,
        "n_test_rows": len(X_test),
        "n_context_rows": len(X_ctx),
        "dropped_columns": dropped_columns,
        "feature_columns": feature_names,
    }

    # Train model based on task type
    model, preds, confidence, proba = _train_and_predict(
        X_ctx, y_ctx, X_test, y_test, task_type
    )

    result["predictions"] = preds
    result["confidence"] = confidence

    # Compute metrics if requested. Guard so a metrics failure degrades
    # gracefully instead of failing the whole prediction request.
    if include_metrics and len(y_test) > 0:
        try:
            metrics_calc = _get_metrics_calculator(task_type)
            # Pass the full class-probability matrix (not the max-prob confidence)
            # so ROC-AUC uses the positive-class probability for binary tasks.
            metrics = metrics_calc.calculate(y_test, preds, y_proba=proba)

            result["metrics"] = metrics

            # Add visualization data
            viz_data = metrics_calc.get_visualization_data(y_test, preds)
            if viz_data:
                result["metrics_visualization"] = {
                    "plot_type": viz_data.get("plot_type"),
                    "data": viz_data,
                }
        except Exception as e:
            result["metrics_error"] = str(e)
    
    # Compute interpretability if requested
    if include_interpretability and model is not None:
        if interpretability_methods is None:
            interpretability_methods = ["feature_importance", "shap", "lime"]
        
        explanations = _compute_interpretability(
            model, X_ctx, y_ctx, X_test, preds,
            interpretability_methods=interpretability_methods,
            n_samples=n_samples_explain,
            feature_names=feature_names,
        )
        
        result["interpretability"] = explanations

    # Summary
    result["summary"] = _generate_summary(result, filename)

    return result


def _train_and_predict(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> tuple[Any, List, List, Optional[np.ndarray]]:
    """
    Train TabICLv2 model and generate predictions.

    Returns:
        (model, predictions_list, confidence_list, proba_matrix)
        proba_matrix is the full class-probability matrix for classification
        (used for ROC-AUC), or None for other tasks.
    """
    model = None
    preds = []
    confidence = []
    proba = None

    if task_type == "classification":
        from tabicl import TabICLClassifier
        model = TabICLClassifier()
        model.fit(X_ctx, y_ctx)
        preds = model.predict(X_test).tolist()

        try:
            proba = model.predict_proba(X_test)
            confidence = np.max(proba, axis=1).tolist()
        except Exception:
            proba = None
            confidence = []

    elif task_type == "regression":
        from tabicl import TabICLRegressor
        model = TabICLRegressor()
        model.fit(X_ctx, y_ctx.astype(float))
        preds = model.predict(X_test).tolist()
        confidence = []

    elif task_type == "time_series":
        # For time series, treat as regression with temporal awareness
        from tabicl import TabICLRegressor
        model = TabICLRegressor()
        model.fit(X_ctx, y_ctx.astype(float))
        preds = model.predict(X_test).tolist()
        confidence = []

    elif task_type == "clustering":
        # Use simple k-means for clustering
        from sklearn.cluster import KMeans
        n_clusters = max(2, min(10, len(np.unique(y_ctx))))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X_ctx)
        preds = model.predict(X_test).tolist()
        confidence = []

    return model, preds, confidence, proba


def _get_metrics_calculator(task_type: str):
    """Get appropriate metrics calculator for task type."""
    if task_type == "classification":
        return ClassificationMetrics()
    elif task_type == "regression":
        return RegressionMetrics()
    elif task_type == "time_series":
        return TimeSeriesMetrics()
    elif task_type == "clustering":
        return ClusteringMetrics()
    else:
        return RegressionMetrics()  # Default


def _compute_interpretability(
    model: Any,
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_test: np.ndarray,
    preds: List,
    interpretability_methods: List[str],
    n_samples: int = 100,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute interpretability explanations.

    Returns:
        Dict with explanations by method
    """
    explanations = {}

    # SHAP/LIME issue a model call per feature-perturbation. For in-context
    # models such as TabICL every call is a full forward pass over the context,
    # so the per-row methods are bounded to keep latency reasonable. The UI
    # aggregates over the explained rows, so a representative sample is enough.
    SHAP_MAX_ROWS = 10
    LIME_MAX_ROWS = 10
    LIME_PERTURBATIONS = 100
    n_test = len(X_test)

    try:
        if "feature_importance" in interpretability_methods:
            explainer = FeatureImportanceExplainer(model, X_ctx, y_ctx)
            explainer.feature_names = feature_names
            expl = explainer.explain_predictions(X_test[:n_samples])
            explanations["feature_importance"] = expl

    except Exception as e:
        explanations["feature_importance_error"] = str(e)

    try:
        if "shap" in interpretability_methods:
            shap_rows = min(n_samples, SHAP_MAX_ROWS, n_test)
            explainer = SHAPExplainer(model, X_ctx, y_ctx, background_size=min(50, len(X_ctx)))
            explainer.feature_names = feature_names
            expl = explainer.explain_predictions(X_test[:shap_rows], n_samples=shap_rows)
            explanations["shap"] = expl

    except Exception as e:
        explanations["shap_error"] = str(e)

    try:
        if "lime" in interpretability_methods:
            lime_rows = min(n_samples, LIME_MAX_ROWS, n_test)
            explainer = LIMEExplainer(model, X_ctx, y_ctx, n_samples=LIME_PERTURBATIONS)
            explainer.feature_names = feature_names
            expl = explainer.explain_predictions(X_test[:lime_rows], n_samples=lime_rows)
            explanations["lime"] = expl

    except Exception as e:
        explanations["lime_error"] = str(e)

    return explanations


def _generate_summary(result: Dict[str, Any], filename: str) -> str:
    """Generate human-readable summary."""
    task_type = result.get("task_type", "unknown")
    target = result.get("target_column", "?")
    n_preds = result.get("n_test_rows", 0)
    
    summary = f"TABICLv2 {task_type} on '{filename}': "
    summary += f"predicted {n_preds} rows for '{target}'. "
    
    metrics = result.get("metrics", {})
    if metrics:
        summary += "\n\n**Metrics** — " + _format_compact_metrics(task_type, metrics)
    elif result.get("metrics_error"):
        summary += f"\n\n_(metrics unavailable: {result['metrics_error']})_"

    # Mention interpretability if it was computed
    interp = result.get("interpretability", {})
    methods = [m for m in ("feature_importance", "shap", "lime") if interp.get(m)]
    if methods:
        summary += f"\n\nExplanations computed: {', '.join(methods)}."

    return summary


def _format_compact_metrics(task_type: str, metrics: Dict[str, Any]) -> str:
    """Build a one-line, human-readable metrics string for the chat reply."""
    def g(key):
        v = metrics.get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) else None

    parts = []
    if task_type == "classification":
        order = [("Accuracy", "accuracy"), ("F1 (macro)", "macro_f1"),
                 ("Precision (macro)", "macro_precision"), ("Recall (macro)", "macro_recall"),
                 ("ROC-AUC", "roc_auc")]
    elif task_type == "regression":
        order = [("R²", "r2"), ("MAE", "mae"), ("RMSE", "rmse"), ("MAPE %", "mape")]
    elif task_type == "time_series":
        order = [("RMSE", "rmse"), ("MAE", "mae"), ("Direction acc. %", "direction_accuracy"), ("MASE", "mase")]
    elif task_type == "clustering":
        order = [("Silhouette", "silhouette_score"), ("Davies-Bouldin", "davies_bouldin_index"),
                 ("Clusters", "n_clusters")]
    else:
        order = [(k, k) for k in metrics]

    for label, key in order:
        val = g(key)
        if val is not None:
            parts.append(f"{label}: {val}")
    return " · ".join(parts) if parts else "computed."


def predict_from_session(dataset_id: str, query: str, user_id: str) -> dict:
    """Called by LangGraph tabular_predict_node. Extracts target column from the query."""
    # Try to extract a column name from the query e.g. "predict diagnosis"
    match = re.search(
        r"(?:predict|classify|forecast|regress(?:ion on)?)\s+['\"]?(\w+)['\"]?",
        query,
        re.IGNORECASE,
    )
    with SessionLocal() as db:
        row = db.query(TabularDataset).filter(
            TabularDataset.id == dataset_id,
            TabularDataset.user_id == user_id,
        ).first()
        columns = row.column_names if row else []

    target = None
    if match:
        candidate = match.group(1)
        if candidate in columns:
            target = candidate

    if not target and columns:
        target = columns[-1]  # default: last column

    if not target:
        return {"summary": "Could not determine target column. Specify 'predict <column_name>'."}

    return run_prediction(dataset_id, user_id, target)
