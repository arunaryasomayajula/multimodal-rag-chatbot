# Tabular Prediction System Refactoring - Complete Guide

## Overview

This document describes the comprehensive refactoring of the TabICLv2-based prediction system. The new system provides:

- **Multiple Task Types**: Classification, Regression, Time Series, Clustering
- **Comprehensive Metrics**: Task-specific performance metrics with visualization support
- **Interpretability**: SHAP values, LIME explanations, and feature importance
- **Graphics Ready**: Metric calculators provide data for frontend visualization
- **Optional Storage**: Predictions can be saved to session storage for later retrieval
- **Batch Operations**: Multiple predictions in a single request

---

## Architecture

### Directory Structure

```
generation/
├── tabular_predictor.py       # Main prediction orchestrator (refactored)
├── metrics/                    # Task-specific metric calculators
│   ├── __init__.py
│   ├── base.py                # Abstract base class
│   ├── regression.py          # Regression metrics (MAE, RMSE, R², MAPE)
│   ├── classification.py      # Classification metrics (Accuracy, Precision, F1, etc.)
│   ├── clustering.py          # Clustering metrics (Silhouette, Davies-Bouldin, etc.)
│   └── time_series.py         # Time series metrics (SMAPE, MASE, Direction Accuracy)
├── interpretability/           # Explainability tools
│   ├── __init__.py
│   ├── base.py                # Abstract Explainer base
│   ├── feature_importance.py  # Permutation-based feature importance
│   ├── shap_explainer.py      # SHAP/Shapley value explanations
│   └── lime_explainer.py      # LIME local explanations
└── visualization/              # Graphics and formatting
    └── __init__.py            # PlotGenerator and MetricsFormatter classes

api/routes/
└── predict.py                 # Updated API endpoints with new parameters
```

---

## Core Components

### 1. Metrics Framework

Each task type has a dedicated metrics calculator implementing the `MetricsCalculator` interface:

```python
class MetricsCalculator(ABC):
    def calculate(self, y_true, y_pred, **kwargs) -> Dict[str, Any]:
        """Compute metrics"""
    
    def get_visualization_data(self, y_true, y_pred, **kwargs) -> Dict[str, Any]:
        """Return data for frontend plotting"""
```

#### Regression Metrics

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

Visualization: Residual plots, Actual vs. Predicted scatter

#### Classification Metrics

- **Accuracy**: Overall correctness
- **Precision, Recall, F1**: Per-class and macro/weighted averages
- **Confusion Matrix**: Full classification breakdown
- **ROC-AUC**: Binary classification curve (when available)

Visualization: Confusion matrix heatmap, ROC curve

#### Time Series Metrics

- **MAE, RMSE, MAPE**: Standard regression metrics
- **SMAPE**: Symmetric MAPE for forecast comparison
- **Direction Accuracy**: % of correct trend predictions
- **MASE**: Mean Absolute Scaled Error (seasonal normalization)

Visualization: Time series line plot (actual vs. predicted)

#### Clustering Metrics

- **Silhouette Score**: Quality of cluster separation (-1 to 1)
- **Davies-Bouldin Index**: Average similarity ratio (lower is better)
- **Calinski-Harabasz Index**: Cluster dispersion (higher is better)
- **Cluster Sizes**: Distribution across clusters

Visualization: 2D scatter plot with cluster colors

### 2. Interpretability Module

Three explainer types provide different perspectives on model decisions:

#### Feature Importance (Permutation-based)

- **Model-agnostic**: Works with any model
- **Global explanation**: Which features matter most overall
- **Method**: Permute each feature, measure prediction change
- **Output**: Importance scores for all features (normalized 0-1)

```python
explainer = FeatureImportanceExplainer(model, X_train, y_train)
importance = explainer.explain_prediction(x_sample)
# Returns: {shap_values, feature_names, importance_scores}
```

#### SHAP (SHapley Additive exPlanations)

- **Theoretically grounded**: Based on Shapley values from game theory
- **Per-sample explanation**: How much each feature contributes to prediction
- **Method**: Simplified kernel SHAP approximation
- **Output**: Additive feature contributions for each sample

```python
explainer = SHAPExplainer(model, X_train, y_train)
explanation = explainer.explain_prediction(x_sample)
# Returns: {shap_values, base_value, prediction, feature_values}
```

#### LIME (Local Interpretable Model-agnostic Explanations)

- **Local linear approximation**: Fits linear model in sample neighborhood
- **Per-sample explanation**: Local feature importance
- **Method**: Perturb around sample, fit weighted linear regression
- **Output**: Linear coefficients (feature contributions)

```python
explainer = LIMEExplainer(model, X_train, y_train)
explanation = explainer.explain_prediction(x_sample)
# Returns: {coefficients, feature_values, prediction}
```

### 3. Visualization & Formatting

`PlotGenerator` creates plot specifications for frontend rendering:

- `generate_confusion_matrix_plot()`: Heatmap for classification
- `generate_residual_plot()`: Scatter for regression diagnostics
- `generate_actual_vs_predicted_plot()`: Regression comparison
- `generate_feature_importance_plot()`: Bar chart of feature importance
- `generate_shap_plot()`: SHAP force plot data
- `generate_time_series_plot()`: Line plot for forecasts
- `generate_clustering_plot()`: 2D scatter for clusters
- `generate_roc_curve_plot()`: ROC curve for binary classification

`MetricsFormatter` organizes metrics for display:

```python
formatted = MetricsFormatter.format_metrics_for_display(metrics, task_type="regression")
# Returns: {
#   "Error Metrics": {"mae": 0.5, "rmse": 0.6, ...},
#   "Goodness of Fit": {"r2": 0.85, ...},
# }
```

---

## API Endpoints

### Main Prediction Endpoint

```http
POST /api/predict
Content-Type: application/json

{
  "dataset_id": "uuid",
  "target_column": "price",
  "context_rows": 50,
  "task_type": "auto",                    // auto | classification | regression | time_series | clustering
  "include_metrics": true,
  "include_interpretability": false,       // Optional - slower if true
  "interpretability_methods": ["feature_importance", "shap", "lime"],
  "n_samples_explain": 100,
  "save_result": false,                   // Save to session storage
  "notes": "Testing with 50 context rows"
}
```

**Response:**

```json
{
  "task_type": "regression",
  "target_column": "price",
  "filename": "housing.csv",
  "n_test_rows": 150,
  "n_context_rows": 50,
  "predictions": [150.2, 200.5, ...],
  "confidence": [0.85, 0.92, ...],
  "metrics": {
    "mae": 15.3,
    "rmse": 20.1,
    "r2": 0.87,
    "mape": 8.5
  },
  "metrics_visualization": {
    "plot_type": "regression_analysis",
    "data": {
      "actual": [...],
      "predicted": [...],
      "residuals": [...]
    }
  },
  "interpretability": {
    "feature_importance": [
      {"sample_idx": 0, "importance_scores": [...], "feature_names": [...]}
    ],
    "shap": [
      {"sample_idx": 0, "shap_values": [...], "prediction": 150.2}
    ],
    "lime": [
      {"sample_idx": 0, "coefficients": [...]}
    ]
  },
  "summary": "TABICLv2 regression on 'housing.csv': predicted 150 rows for 'price'. R²: 0.87, MAE: 15.3",
  "result_id": "uuid"  // Only if save_result=true
}
```

### Batch Prediction

```http
POST /api/predict/batch
Content-Type: application/json

{
  "requests": [
    {"dataset_id": "...", "target_column": "price", ...},
    {"dataset_id": "...", "target_column": "bedrooms", ...}
  ]
}
```

### Session Storage Endpoints

**List saved predictions:**

```http
GET /api/predict/results?dataset_id=optional&limit=50
```

**Retrieve specific prediction:**

```http
GET /api/predict/results/{result_id}
```

**Update prediction metadata:**

```http
PUT /api/predict/results/{result_id}?notes=new_note&is_pinned=true
```

**Delete prediction:**

```http
DELETE /api/predict/results/{result_id}
```

---

## Usage Examples

### Example 1: Simple Regression Prediction

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={
        "dataset_id": "abc123",
        "target_column": "price",
        "task_type": "regression",
        "include_metrics": True,
        "include_interpretability": False
    },
    headers={"Authorization": f"Bearer {token}"}
)

result = response.json()
print(f"Predictions: {result['predictions'][:5]}")
print(f"Metrics: {result['metrics']}")
# Output:
# Predictions: [150.2, 200.5, 175.3, 190.1, 210.7]
# Metrics: {'mae': 15.3, 'rmse': 20.1, 'r2': 0.87, 'mape': 8.5}
```

### Example 2: Classification with Interpretability

```python
response = requests.post(
    "http://localhost:8000/api/predict",
    json={
        "dataset_id": "abc123",
        "target_column": "churn",
        "task_type": "classification",
        "include_metrics": True,
        "include_interpretability": True,
        "interpretability_methods": ["feature_importance", "shap"],
        "n_samples_explain": 50,
        "save_result": True,
        "notes": "Churn prediction - threshold analysis"
    },
    headers={"Authorization": f"Bearer {token}"}
)

result = response.json()
print(f"Accuracy: {result['metrics']['accuracy']}")
print(f"Saved as: {result['result_id']}")

# Retrieve later
saved = requests.get(
    f"http://localhost:8000/api/predict/results/{result['result_id']}",
    headers={"Authorization": f"Bearer {token}"}
).json()

print(f"Feature importance: {saved['feature_importance']}")
```

### Example 3: Time Series with Batch Processing

```python
response = requests.post(
    "http://localhost:8000/api/predict/batch",
    json={
        "requests": [
            {
                "dataset_id": "abc123",
                "target_column": "daily_sales",
                "task_type": "time_series",
                "context_rows": 100,
                "save_result": True
            },
            {
                "dataset_id": "abc123",
                "target_column": "daily_sales",
                "task_type": "time_series",
                "context_rows": 200,
                "save_result": True
            }
        ]
    },
    headers={"Authorization": f"Bearer {token}"}
)

results = response.json()["results"]
for r in results:
    print(f"Context: {r['n_context_rows']}, Direction Accuracy: {r['metrics'].get('direction_accuracy', 'N/A')}%")
```

---

## Database Schema

### PredictionResult Table

```sql
CREATE TABLE prediction_results (
  id VARCHAR PRIMARY KEY,
  user_id VARCHAR NOT NULL,
  dataset_id VARCHAR NOT NULL,
  target_column VARCHAR NOT NULL,
  task_type VARCHAR NOT NULL,           -- classification | regression | time_series | clustering
  context_rows INTEGER NOT NULL,
  predictions JSON NOT NULL,             -- List of predictions
  confidence JSON,                       -- List of confidence scores
  metrics JSON,                          -- {accuracy: 0.92, ...}
  feature_importance JSON,               -- [{sample_idx, importance_scores, feature_names}]
  shap_values JSON,                      -- [{sample_idx, shap_values, prediction}]
  lime_explanations JSON,                -- [{sample_idx, coefficients, feature_values}]
  created_at TIMESTAMP DEFAULT NOW(),
  notes TEXT,                            -- User notes
  is_pinned INTEGER DEFAULT 0,           -- Boolean flag for pinning
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (dataset_id) REFERENCES tabular_datasets(id)
);
```

---

## Configuration & Performance Tuning

### Key Parameters

| Parameter | Default | Impact | Notes |
|-----------|---------|--------|-------|
| `context_rows` | 50 | In-context learning set size | Larger = more context, slower |
| `include_metrics` | True | Compute performance metrics | Always recommended |
| `include_interpretability` | False | Enable explanations | Adds 20-50% overhead |
| `n_samples_explain` | 100 | Samples to explain | Higher = more thorough, slower |

### Performance Expectations

- **Simple prediction** (no interpretability): 0.5-2 seconds
- **With metrics**: +0.2-0.5 seconds
- **With feature_importance**: +1-3 seconds
- **With SHAP**: +2-5 seconds (depends on background size)
- **With LIME**: +1-2 seconds (depends on n_samples)

### Optimization Tips

1. Set `include_interpretability=False` for real-time predictions
2. Limit `n_samples_explain` to 50-100 for large datasets
3. Use `task_type="auto"` to benefit from optimized paths
4. Cache interpretability results in session storage

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Dataset not found" | Invalid dataset_id or no access | Verify dataset_id, check permissions |
| "Column not in dataset" | Typo in target_column | Use /predict/datasets endpoint to list columns |
| "Prediction failed" | Model training error | Check data quality, try different context_rows |
| "SHAP computation error" | Background data issues | Reduce n_samples_explain or use feature_importance instead |

---

## Frontend Integration

### Recommended Components

1. **Metrics Display**
   - Task-type-specific metric cards
   - Color-coded performance indicators

2. **Visualizations**
   - Use plot data from `metrics_visualization`
   - Render with Plotly, Matplotlib, or D3.js
   - Show residual plots for regression
   - Show confusion matrices for classification

3. **Interpretability UI**
   - Tabs for SHAP / LIME / Feature Importance
   - Show top-10 features
   - Interactive row-level explanations
   - Export explanations as CSV/JSON

4. **Results Management**
   - Save/pin important predictions
   - Add notes to predictions
   - Compare prediction results
   - Filter by task_type or dataset

---

## Migration Guide

### For Existing Code

Old API:

```python
result = run_prediction(
    dataset_id="abc",
    user_id="user1",
    target_column="price",
    task_type="auto"
)
```

New API (backward compatible):

```python
result = run_prediction(
    dataset_id="abc",
    user_id="user1",
    target_column="price",
    task_type="auto",
    include_metrics=True,        # NEW: explicit metrics
    include_interpretability=False  # NEW: optional interpretability
)
```

### To Use New Features

```python
result = run_prediction(
    ...,
    include_metrics=True,
    include_interpretability=True,
    interpretability_methods=["feature_importance", "shap", "lime"],
    n_samples_explain=50
)
```

---

## Future Enhancements

- [ ] SHAP TreeExplainer for tree-based models
- [ ] Integrated LIME package for better local explanations
- [ ] Partial dependence plots
- [ ] ICE (Individual Conditional Expectation) plots
- [ ] Anomaly detection on predictions
- [ ] A/B testing framework for model comparison
- [ ] Real-time metric streaming
- [ ] GPU-accelerated metric computation

---

## Support & Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Debug Points

```python
# Check metrics availability
result = run_prediction(..., include_metrics=True)
print(f"Metrics: {result.get('metrics', {}).keys()}")

# Verify interpretability data
print(f"Interpretability: {result.get('interpretability', {}).keys()}")

# Check visualization format
print(f"Plot type: {result.get('metrics_visualization', {}).get('plot_type')}")
```

---

## References

- SHAP documentation: https://shap.readthedocs.io/
- LIME documentation: https://lime-ml.readthedocs.io/
- TabICLv2: https://github.com/nband/tabicl
- Scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
