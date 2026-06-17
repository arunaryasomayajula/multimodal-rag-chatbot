# Implementation Examples - Prediction System Refactoring

## Integration Guide with Code Examples

This document provides practical code examples for integrating the new prediction system into your application.

---

## 1. Basic Prediction (Python Backend)

### Example 1: Simple Regression

```python
from generation.tabular_predictor import run_prediction

# Basic prediction
result = run_prediction(
    dataset_id="housing-data-uuid",
    user_id="user123",
    target_column="price",
    task_type="auto"
)

print(f"Predictions: {result['predictions'][:5]}")
print(f"Task Type: {result['task_type']}")
print(f"Summary: {result['summary']}")
```

**Output:**
```
Predictions: [150.2, 200.5, 175.3, 190.1, 210.7]
Task Type: regression
Summary: TABICLv2 regression on 'housing.csv': predicted 150 rows for 'price'. R²: 0.87, MAE: 15.3
```

### Example 2: Classification with Metrics

```python
from generation.tabular_predictor import run_prediction

# Classification with metrics
result = run_prediction(
    dataset_id="iris-data-uuid",
    user_id="user123",
    target_column="species",
    task_type="classification",
    include_metrics=True,
    context_rows=75
)

print(f"Predictions: {result['predictions'][:5]}")
print(f"Confidence: {result['confidence'][:5]}")
print(f"Metrics: {result['metrics']}")
```

**Output:**
```
Predictions: [0, 2, 1, 0, 2]
Confidence: [0.95, 0.87, 0.92, 0.91, 0.88]
Metrics: {
    'accuracy': 0.96,
    'macro_precision': 0.96,
    'macro_recall': 0.96,
    'macro_f1': 0.96,
    'weighted_f1': 0.96,
    'n_classes': 3
}
```

### Example 3: Full Analysis with Interpretability

```python
from generation.tabular_predictor import run_prediction

# Full analysis
result = run_prediction(
    dataset_id="credit-risk-uuid",
    user_id="user123",
    target_column="default",
    task_type="classification",
    include_metrics=True,
    include_interpretability=True,
    interpretability_methods=["feature_importance", "shap", "lime"],
    n_samples_explain=50
)

# Access all components
print(f"Predictions: {len(result['predictions'])} samples")
print(f"Metrics: {result['metrics'].keys()}")
print(f"Explanations: {result['interpretability'].keys()}")

# Access specific explanation
if result['interpretability'].get('shap'):
    first_explanation = result['interpretability']['shap'][0]
    print(f"SHAP values for sample 0: {first_explanation['shap_values']}")
    print(f"Prediction: {first_explanation['prediction']}")
    print(f"Top contributing features: {first_explanation['feature_names'][:3]}")
```

---

## 2. Using Metrics Directly

### Example 4: Custom Metric Calculation

```python
from generation.metrics import RegressionMetrics, ClassificationMetrics
import numpy as np

# For regression
reg_metrics = RegressionMetrics()
y_true = [10, 20, 30, 40, 50]
y_pred = [9.5, 21.0, 29.5, 40.5, 50.2]

metrics = reg_metrics.calculate(y_true, y_pred)
print(f"MAE: {metrics['mae']}")
print(f"R²: {metrics['r2']}")

# Get visualization data
viz_data = reg_metrics.get_visualization_data(y_true, y_pred)
print(f"Plot type: {viz_data['plot_type']}")
print(f"Actual: {viz_data['actual']}")
print(f"Residuals: {viz_data['residuals']}")

# For classification
clf_metrics = ClassificationMetrics()
y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1]
y_proba = [[1, 0], [0.1, 0.9], [0.6, 0.4], [0.95, 0.05], [0.2, 0.8], [0.1, 0.9]]

metrics = clf_metrics.calculate(y_true, y_pred, y_proba=y_proba)
print(f"Accuracy: {metrics['accuracy']}")
print(f"F1: {metrics['macro_f1']}")
print(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
```

---

## 3. Using Explainers Directly

### Example 5: Feature Importance

```python
from generation.interpretability import FeatureImportanceExplainer
import numpy as np

# Simple mock model
class SimpleModel:
    def predict(self, X):
        return X.sum(axis=1)

X_train = np.random.rand(100, 5)
y_train = X_train.sum(axis=1)
model = SimpleModel()

# Create explainer
explainer = FeatureImportanceExplainer(model, X_train, y_train)

# Explain single sample
x_test = np.array([0.5, 0.3, 0.7, 0.2, 0.9])
explanation = explainer.explain_prediction(x_test)

print(f"Feature names: {explanation['feature_names']}")
print(f"Importance scores: {explanation['importance_scores']}")
print(f"Top {explanation['top_k']} features: {explanation['feature_names'][:10]}")
```

**Output:**
```
Feature names: ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
Importance scores: [0.35, 0.28, 0.15, 0.10, 0.12]
Top 10 features: ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
```

### Example 6: SHAP Explanations

```python
from generation.interpretability import SHAPExplainer

# Using the same model and data
explainer = SHAPExplainer(model, X_train, y_train, background_size=50)

# Explain single sample
explanation = explainer.explain_prediction(x_test)
print(f"Base value (expected output): {explanation['base_value']}")
print(f"Prediction: {explanation['prediction']}")
print(f"SHAP values: {explanation['shap_values']}")
print(f"Feature contributions: {dict(zip(explanation['feature_names'], explanation['shap_values']))}")

# Explain multiple samples
explanations = explainer.explain_predictions(X_train[:10])
print(f"Explained {len(explanations)} samples")
for i, exp in enumerate(explanations):
    print(f"  Sample {i}: prediction={exp['prediction']:.2f}, top feature={exp['feature_names'][0]}")
```

### Example 7: LIME Explanations

```python
from generation.interpretability import LIMEExplainer

# Create explainer
explainer = LIMEExplainer(model, X_train, y_train, n_samples=500)

# Explain sample
explanation = explainer.explain_prediction(x_test)
print(f"Local coefficients: {explanation['coefficients']}")
print(f"Feature values: {explanation['feature_values']}")
print(f"Prediction: {explanation['prediction']}")

# Show local importance ranking
coef_importance = np.argsort(np.abs(explanation['coefficients']))[::-1]
print(f"Local feature importance ranking:")
for rank, idx in enumerate(coef_importance[:3]):
    print(f"  {rank+1}. {explanation['feature_names'][idx]}: {explanation['coefficients'][idx]:.4f}")
```

---

## 4. Using Visualization

### Example 8: Generating Plot Specifications

```python
from generation.visualization import PlotGenerator, MetricsFormatter
import numpy as np

# Generate regression plots
actual = np.array([10, 20, 30, 40, 50])
predicted = np.array([9.5, 21.0, 29.5, 40.5, 50.2])

# Residual plot
residual_plot = PlotGenerator.generate_residual_plot(
    residuals=(actual - predicted).tolist(),
    predictions=predicted.tolist()
)
print(f"Residual plot type: {residual_plot['type']}")
print(f"X data: {residual_plot['x']}")
print(f"Y data: {residual_plot['y']}")

# Actual vs Predicted plot
avp_plot = PlotGenerator.generate_actual_vs_predicted_plot(
    actual=actual.tolist(),
    predicted=predicted.tolist()
)
print(f"Title: {avp_plot['title']}")  # Includes R² value

# Feature importance plot
importance_scores = [0.35, 0.28, 0.15, 0.10, 0.12]
feature_names = ["area", "bedrooms", "location", "age", "condition"]
fi_plot = PlotGenerator.generate_feature_importance_plot(
    importance_scores=importance_scores,
    feature_names=feature_names,
    top_k=5
)
print(f"Feature importance plot: {fi_plot['title']}")
print(f"Features: {fi_plot['x']}")
print(f"Scores: {fi_plot['y']}")

# Format metrics for display
metrics = {
    'mae': 15.3,
    'mse': 404.09,
    'rmse': 20.1,
    'r2': 0.87,
    'mape': 8.5
}
formatted = MetricsFormatter.format_metrics_for_display(metrics, task_type="regression")
print(f"Formatted metrics: {formatted}")
```

---

## 5. API Integration (Frontend)

### Example 9: JavaScript/Fetch API

```javascript
// Make prediction
async function predictWithMetrics(datasetId, targetColumn) {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getToken()}`
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            target_column: targetColumn,
            task_type: 'auto',
            include_metrics: true,
            include_interpretability: false
        })
    });
    
    const result = await response.json();
    return result;
}

// Call it
const result = await predictWithMetrics('abc123', 'price');
console.log(`Accuracy: ${result.metrics.r2}`);
console.log(`Predictions: ${result.predictions.slice(0, 5)}`);
```

### Example 10: React Component

```jsx
import React, { useState } from 'react';
import PlotlyChart from 'react-plotly.js';

function PredictionResults({ result }) {
    const [showInterpretability, setShowInterpretability] = useState(false);
    
    if (!result) return <div>Loading...</div>;
    
    return (
        <div className="prediction-results">
            <h2>{result.summary}</h2>
            
            {/* Metrics Section */}
            <section className="metrics">
                <h3>Performance Metrics</h3>
                <div className="metric-cards">
                    {Object.entries(result.metrics).map(([key, value]) => (
                        <div key={key} className="metric-card">
                            <div className="metric-label">{key}</div>
                            <div className="metric-value">
                                {typeof value === 'number' ? value.toFixed(4) : value}
                            </div>
                        </div>
                    ))}
                </div>
            </section>
            
            {/* Visualization Section */}
            {result.metrics_visualization && (
                <section className="visualization">
                    <h3>Visualization</h3>
                    <PlotlyVisualization data={result.metrics_visualization.data} />
                </section>
            )}
            
            {/* Interpretability Section */}
            <section className="interpretability">
                <button onClick={() => setShowInterpretability(!showInterpretability)}>
                    {showInterpretability ? 'Hide' : 'Show'} Interpretability
                </button>
                
                {showInterpretability && result.interpretability && (
                    <div>
                        {result.interpretability.feature_importance && (
                            <div className="feature-importance">
                                <h4>Feature Importance</h4>
                                <FeatureImportanceChart data={result.interpretability.feature_importance} />
                            </div>
                        )}
                        
                        {result.interpretability.shap && (
                            <div className="shap">
                                <h4>SHAP Explanations</h4>
                                <SHAPExplainer data={result.interpretability.shap} />
                            </div>
                        )}
                    </div>
                )}
            </section>
        </div>
    );
}

export default PredictionResults;
```

---

## 6. Database Operations

### Example 11: Saving and Retrieving Results

```python
from api.routes.predict import predict
from db.models import PredictionResult
from db.session import SessionLocal

# The predict endpoint now saves automatically with save_result=True
request = PredictRequest(
    dataset_id="abc123",
    target_column="price",
    save_result=True,
    notes="Testing new parameters"
)

# In route handler:
result = run_prediction(...)  # ... computation

# Save to database
pred_result = PredictionResult(
    user_id=user_id,
    dataset_id=request.dataset_id,
    target_column=request.target_column,
    task_type=result.get("task_type"),
    context_rows=request.context_rows,
    predictions=result.get("predictions", []),
    confidence=result.get("confidence", []),
    metrics=result.get("metrics"),
    notes=request.notes
)
db.add(pred_result)
db.commit()

# Later: retrieve
with SessionLocal() as db:
    saved = db.query(PredictionResult).filter_by(
        user_id=user_id,
        target_column="price"
    ).order_by(PredictionResult.created_at.desc()).first()
    
    print(f"Saved prediction: {saved.id}")
    print(f"Metrics: {saved.metrics}")
    print(f"Notes: {saved.notes}")
```

### Example 12: Batch Result Management

```python
from sqlalchemy import func
from db.models import PredictionResult

def get_statistics(user_id):
    """Get stats on saved predictions"""
    with SessionLocal() as db:
        # Count predictions by task type
        counts = db.query(
            PredictionResult.task_type,
            func.count(PredictionResult.id)
        ).filter(PredictionResult.user_id == user_id).group_by(
            PredictionResult.task_type
        ).all()
        
        print("Predictions by task type:")
        for task_type, count in counts:
            print(f"  {task_type}: {count}")
        
        # Get average metrics
        avg_accuracy = db.query(
            func.avg(func.json_extract(PredictionResult.metrics, '$.accuracy'))
        ).filter(PredictionResult.user_id == user_id).scalar()
        
        print(f"Average accuracy: {avg_accuracy}")
        
        # List pinned predictions
        pinned = db.query(PredictionResult).filter(
            PredictionResult.user_id == user_id,
            PredictionResult.is_pinned == 1
        ).all()
        
        print(f"Pinned predictions: {len(pinned)}")
```

---

## 7. Error Handling

### Example 13: Comprehensive Error Handling

```python
from api.routes.predict import predict, HTTPException
import logging

logger = logging.getLogger(__name__)

def safe_predict(dataset_id, target_column, user_id):
    """Predict with error handling"""
    try:
        result = run_prediction(
            dataset_id=dataset_id,
            user_id=user_id,
            target_column=target_column,
            include_metrics=True,
            include_interpretability=False  # Skip slow computation
        )
        return {"success": True, "result": result}
    
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        return {"success": False, "error": f"Invalid input: {str(e)}", "code": "INVALID_INPUT"}
    
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        return {"success": False, "error": "Dataset not found", "code": "DATASET_NOT_FOUND"}
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {"success": False, "error": "Internal server error", "code": "INTERNAL_ERROR"}

# Usage
response = safe_predict("bad-id", "price", "user123")
if not response["success"]:
    print(f"Error: {response['error']} ({response['code']})")
else:
    print(f"Success: {response['result']['summary']}")
```

---

## 8. Performance Optimization

### Example 14: Optimized Batch Processing

```python
from generation.tabular_predictor import run_prediction
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def batch_predict_optimized(requests, max_workers=4):
    """Process predictions in parallel"""
    
    def predict_task(req):
        return run_prediction(
            dataset_id=req["dataset_id"],
            user_id=req["user_id"],
            target_column=req["target_column"],
            include_metrics=req.get("include_metrics", True),
            include_interpretability=req.get("include_interpretability", False),
            n_samples_explain=req.get("n_samples_explain", 100)
        )
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, predict_task, req)
            for req in requests
        ]
        results = await asyncio.gather(*tasks)
    
    return results

# Usage
requests = [
    {"dataset_id": "d1", "user_id": "u1", "target_column": "price"},
    {"dataset_id": "d2", "user_id": "u1", "target_column": "value"},
    {"dataset_id": "d3", "user_id": "u1", "target_column": "score"},
]

results = asyncio.run(batch_predict_optimized(requests))
print(f"Processed {len(results)} predictions")
```

---

## Summary

These examples cover:
✅ Basic prediction usage
✅ Direct metric calculation
✅ Using explainers independently
✅ Visualization generation
✅ API integration (JS/React)
✅ Database operations
✅ Error handling
✅ Performance optimization

For more details, see:
- PREDICTION_SYSTEM_GUIDE.md (comprehensive guide)
- PREDICTION_API_QUICKREF.md (quick reference)
- REFACTORING_MANIFEST.md (file structure)
