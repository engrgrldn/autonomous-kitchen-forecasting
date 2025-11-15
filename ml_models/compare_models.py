import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("📊 Model Comparison")
print("=" * 70)

# Load metrics
prophet_metrics = pd.read_csv('ml_models/prophet_metrics.csv')
xgboost_metrics = pd.read_csv('ml_models/xgboost_metrics.csv')

prophet_metrics['model'] = 'Prophet'
xgboost_metrics['model'] = 'XGBoost'

# Combine
all_metrics = pd.concat([prophet_metrics, xgboost_metrics])

# Summary
print("\n📈 Average Performance by Model:")
summary = all_metrics.groupby('model')[['mae', 'rmse', 'r2']].mean()
print(summary)

# Best model per location
print("\n🏆 Best Model by Location (by MAE):")
best_mae = all_metrics.loc[all_metrics.groupby('location_id')['mae'].idxmin()]
print(best_mae[['location_id', 'model', 'mae', 'r2']])

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = ['mae', 'rmse', 'r2']
titles = ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'R² Score']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx]
    
    prophet_vals = prophet_metrics[metric].values
    xgboost_vals = xgboost_metrics[metric].values
    
    x = np.arange(len(prophet_vals))
    width = 0.35
    
    ax.bar(x - width/2, prophet_vals, width, label='Prophet', alpha=0.8)
    ax.bar(x + width/2, xgboost_vals, width, label='XGBoost', alpha=0.8)
    
    ax.set_xlabel('Location')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(prophet_metrics['location_id'])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_models/model_comparison.png', dpi=150)
print("\n✅ Comparison chart saved to: ml_models/model_comparison.png")
