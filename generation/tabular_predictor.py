"""
TABICLv2 wrapper for zero-shot tabular prediction.
Uses tabicl.TabICLClassifier / TabICLRegressor (sklearn-compatible).
"""
import re
import numpy as np
import pandas as pd
from db.session import SessionLocal
from db.models import TabularDataset


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
    if series.dtype == object or series.nunique() <= 20:
        return "classify"
    return "regress"


def run_prediction(
    dataset_id: str,
    user_id: str,
    target_column: str,
    context_rows: int = 50,
    task_type: str = "auto",
) -> dict:
    df, filename = _load_dataset_df(dataset_id, user_id)

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not in dataset")

    df = df.dropna(subset=[target_column])
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categoricals numerically for TABICLv2
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include="object").columns:
        X_enc[col] = X_enc[col].astype("category").cat.codes

    if task_type == "auto":
        task_type = _detect_task_type(y)

    # Split: first context_rows as in-context set, rest as test
    n = len(X_enc)
    ctx_n = min(context_rows, max(1, n // 2))
    X_ctx, y_ctx = X_enc.iloc[:ctx_n], y.iloc[:ctx_n]
    X_test, y_test = X_enc.iloc[ctx_n:], y.iloc[ctx_n:]

    if task_type == "classify":
        from tabicl import TabICLClassifier
        model = TabICLClassifier()
        model.fit(X_ctx.values, y_ctx.values)
        preds = model.predict(X_test.values).tolist()
        try:
            proba = model.predict_proba(X_test.values)
            confidence = np.max(proba, axis=1).tolist()
        except Exception:
            confidence = []
        metrics = {}
        if len(y_test) > 0:
            correct = sum(p == a for p, a in zip(preds, y_test.tolist()))
            metrics["accuracy"] = round(correct / len(preds), 4)
    else:
        from tabicl import TabICLRegressor
        model = TabICLRegressor()
        model.fit(X_ctx.values, y_ctx.values.astype(float))
        preds = model.predict(X_test.values).tolist()
        confidence = []
        metrics = {}
        if len(y_test) > 0:
            from sklearn.metrics import mean_absolute_error
            metrics["mae"] = round(mean_absolute_error(y_test.values.astype(float), preds), 4)

    summary = (
        f"TABICLv2 {'classification' if task_type == 'classify' else 'regression'} "
        f"on '{filename}': predicted {len(preds)} rows for column '{target_column}'. "
        + (f"Accuracy: {metrics.get('accuracy', 'N/A')}" if task_type == "classify"
           else f"MAE: {metrics.get('mae', 'N/A')}")
    )

    return {
        "predictions": preds,
        "confidence": confidence,
        "metrics": metrics,
        "summary": summary,
        "task_type": task_type,
        "target_column": target_column,
        "n_test_rows": len(preds),
    }


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
