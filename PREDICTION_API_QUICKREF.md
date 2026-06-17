# Quick Reference - API Endpoints & Usage

## Endpoints Summary

### Prediction Endpoints

```
POST   /api/predict              - Run single prediction
POST   /api/predict/batch        - Batch process predictions
GET    /api/predict/datasets     - List available datasets
GET    /api/predict/datasets/{id} - Get dataset metadata
```

### Result Management Endpoints

```
GET    /api/predict/results                    - List saved results
GET    /api/predict/results/{result_id}        - Retrieve result
PUT    /api/predict/results/{result_id}        - Update metadata
DELETE /api/predict/results/{result_id}        - Delete result
```

---

## Quick Start

### 1. Basic Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123",
    "target_column": "price",
    "task_type": "auto"
  }'
```

### 2. Prediction with Metrics

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123",
    "target_column": "price",
    "task_type": "regression",
    "include_metrics": true
  }'
```

### 3. Prediction with Interpretability

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123",
    "target_column": "churn",
    "task_type": "classification",
    "include_metrics": true,
    "include_interpretability": true,
    "interpretability_methods": ["feature_importance", "shap"],
    "n_samples_explain": 50,
    "save_result": true,
    "notes": "Testing new threshold"
  }'
```

### 4. Retrieve Saved Result

```bash
curl -X GET http://localhost:8000/api/predict/results/result_uuid \
  -H "Authorization: Bearer TOKEN"
```

---

## Parameter Guide

| Parameter | Type | Default | Valid Values | Notes |
|-----------|------|---------|--------------|-------|
| `dataset_id` | string | — | UUID | Required |
| `target_column` | string | — | column name | Required |
| `context_rows` | integer | 50 | 1-500 | In-context learning set size |
| `task_type` | string | "auto" | auto, classification, regression, time_series, clustering | Auto-detect or specify |
| `include_metrics` | boolean | true | true, false | Compute performance metrics |
| `include_interpretability` | boolean | false | true, false | Compute explanations (slower) |
| `interpretability_methods` | array | all | ["feature_importance", "shap", "lime"] | Which methods to use |
| `n_samples_explain` | integer | 100 | 1-1000 | Samples to explain (efficiency) |
| `save_result` | boolean | false | true, false | Save to session storage |
| `notes` | string | null | any text | User notes for result |

---

## Task Types & Recommended Metrics

### Auto-Detection
- Numeric with >20 unique values → **Regression**
- Numeric/string with ≤20 unique values → **Classification**

### Classification
Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
```json
"metrics": {
  "accuracy": 0.92,
  "macro_precision": 0.90,
  "macro_recall": 0.88,
  "macro_f1": 0.89,
  "weighted_precision": 0.92,
  "weighted_recall": 0.92,
  "weighted_f1": 0.92,
  "roc_auc": 0.96,
  "n_classes": 2
}
```

### Regression
Metrics: MAE, MSE, RMSE, R², MAPE
```json
"metrics": {
  "mae": 15.3,
  "mse": 404.09,
  "rmse": 20.1,
  "r2": 0.87,
  "mape": 8.5
}
```

### Time Series
Metrics: MAE, RMSE, MAPE, SMAPE, Direction Accuracy, MASE
```json
"metrics": {
  "mae": 5.2,
  "rmse": 7.1,
  "mape": 3.2,
  "smape": 3.5,
  "direction_accuracy": 78.5,
  "mase": 1.2
}
```

### Clustering
Metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz, Cluster Sizes
```json
"metrics": {
  "silhouette_score": 0.65,
  "davies_bouldin_index": 0.42,
  "calinski_harabasz_index": 125.3,
  "n_clusters": 5,
  "cluster_sizes": [102, 95, 88, 110, 105]
}
```

---

## Interpretability Methods

### Feature Importance
- **Best for**: Quick understanding of feature impact
- **Time**: 1-3 seconds
- **Output**: Importance scores (0-1 normalized)
- **Access**: `result["interpretability"]["feature_importance"]`

```python
{
  "sample_idx": 0,
  "importance_scores": [0.35, 0.28, 0.15, 0.10, 0.12],
  "feature_names": ["area", "bedrooms", "location", "age", "condition"],
  "top_k": 10
}
```

### SHAP (Shapley)
- **Best for**: Theoretically-grounded per-sample explanations
- **Time**: 2-5 seconds
- **Output**: Additive feature contributions
- **Access**: `result["interpretability"]["shap"]`

```python
{
  "sample_idx": 0,
  "shap_values": [50.2, 25.3, -10.1, 5.0, 8.7],
  "base_value": 100.0,
  "prediction": 150.2,
  "feature_names": ["area", "bedrooms", "location", "age", "condition"],
  "feature_values": [2000, 3, "downtown", 10, "good"]
}
```

### LIME (Local)
- **Best for**: Understanding local decision boundaries
- **Time**: 1-2 seconds
- **Output**: Local linear regression coefficients
- **Access**: `result["interpretability"]["lime"]`

```python
{
  "sample_idx": 0,
  "coefficients": [0.025, 0.008, -0.002, 0.001, 0.003],
  "feature_names": ["area", "bedrooms", "location", "age", "condition"],
  "feature_values": [2000, 3, "downtown", 10, "good"],
  "prediction": 150.2
}
```

---

## Visualization Data Format

All `metrics_visualization` responses include plot specifications:

```json
"metrics_visualization": {
  "plot_type": "regression_analysis|classification_analysis|time_series_forecast|clustering_analysis",
  "data": {
    // Plot-specific data - use with your visualization framework
  }
}
```

### Rendering Options
- **Plotly.js**: Recommended for web UIs
- **Matplotlib**: Python backend rendering
- **D3.js**: Custom interactive visualizations
- **Altair/Vega-Lite**: Declarative specifications

---

## Common Workflows

### Scenario 1: Evaluate Model on New Data
```python
# Get metrics without interpretability
result = requests.post("/api/predict", json={
    "dataset_id": "data123",
    "target_column": "target",
    "include_metrics": True,
    "include_interpretability": False  # Skip slow computation
})

print(f"Accuracy: {result['metrics']['accuracy']}")
print(f"R²: {result['metrics']['r2']}")
```

### Scenario 2: Explain Specific Predictions
```python
# Get full explanations
result = requests.post("/api/predict", json={
    "dataset_id": "data123",
    "target_column": "target",
    "include_interpretability": True,
    "interpretability_methods": ["shap"],  # Focus on SHAP
    "n_samples_explain": 20,
    "save_result": True  # Save for reference
})

result_id = result["result_id"]

# Access explanations
for shap_ex in result["interpretability"]["shap"]:
    print(f"Row {shap_ex['sample_idx']}: {shap_ex['prediction']}")
    print(f"Top contributing features: {shap_ex['shap_values'][:3]}")
```

### Scenario 3: Compare Multiple Context Sizes
```python
# Batch process with different parameters
batch_result = requests.post("/api/predict/batch", json={
    "requests": [
        {"dataset_id": "...", "target_column": "...", "context_rows": 25, "save_result": True},
        {"dataset_id": "...", "target_column": "...", "context_rows": 50, "save_result": True},
        {"dataset_id": "...", "target_column": "...", "context_rows": 100, "save_result": True},
    ]
})

results = batch_result["results"]
for r in results:
    print(f"Context {r['n_context_rows']}: R² = {r['metrics'].get('r2')}")
```

### Scenario 4: Audit & Archive Predictions
```python
# List all saved predictions
saved = requests.get("/api/predict/results?limit=100").json()

# Update with notes
for r in saved:
    requests.put(f"/api/predict/results/{r['id']}", 
                json={"notes": f"Reviewed on 2024-01-15", "is_pinned": True})

# Export important ones
important = [r for r in saved if r['is_pinned']]
```

---

## Error Codes & Troubleshooting

| Error | Code | Solution |
|-------|------|----------|
| "Dataset not found" | 400 | Check dataset_id, verify access permissions |
| "Column not in dataset" | 400 | List columns with `/api/predict/datasets/{id}` |
| "Prediction failed" | 500 | Check data quality, try different context_rows |
| "Authentication failed" | 401 | Verify JWT token is valid |
| "SHAP computation error" | 500 | Reduce n_samples_explain, use feature_importance instead |

---

## Performance Tips

1. **For real-time predictions**: Skip interpretability
   ```python
   "include_interpretability": False
   ```

2. **For batch analysis**: Use batch endpoint
   ```python
   POST /api/predict/batch  # 10x faster per prediction
   ```

3. **For large datasets**: Limit explanations
   ```python
   "n_samples_explain": 50  # Instead of 100+
   ```

4. **For later analysis**: Save results
   ```python
   "save_result": True
   ```

---

## Response Time Expectations

| Configuration | Time | Notes |
|---------------|------|-------|
| Minimal | 0.5-2s | No metrics, no interpretability |
| With metrics | +0.2-0.5s | — |
| + Feature Importance | +1-3s | — |
| + SHAP | +2-5s | Depends on background size |
| + LIME | +1-2s | Depends on n_samples |
| All features | 5-10s | Full analysis mode |
| Batch (10x) | ~6s total | ~0.6s per prediction |
