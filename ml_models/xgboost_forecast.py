import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🚀 XGBoost Demand Forecasting Model")
print("=" * 70)

# Load data
df = pd.read_csv('ml_models/ml_features.csv')
df['date'] = pd.to_datetime(df['date'])

# Encode categorical variables
le_location = LabelEncoder()
le_weather = LabelEncoder()
df['location_encoded'] = le_location.fit_transform(df['location_id'])
df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'])

# Define features
feature_cols = [
    'location_encoded', 'temperature_c', 'precipitation_mm',
    'is_holiday', 'is_weekend', 'day_of_week', 'month',
    'avg_success_rate', 'total_downtime_minutes',
    'weather_encoded', 'days_since_start',
    'new_customer_rate', 'pizza_quantity', 'salad_quantity'
]

# Remove rows with NaN
df_clean = df.dropna(subset=feature_cols + ['total_orders'])

# Store results
all_predictions = []
metrics = []

for location in df['location_id'].unique():
    print(f"\n📍 Training model for {location}")
    
    # Filter for location
    location_df = df_clean[df_clean['location_id'] == location].copy()
    location_df = location_df.sort_values('date')
    
    # Train/test split (last 14 days for testing)
    train_size = len(location_df) - 14
    train_df = location_df[:train_size]
    test_df = location_df[train_size:]
    
    # Prepare features and target
    X_train = train_df[feature_cols]
    y_train = train_df['total_orders']
    X_test = test_df[feature_cols]
    y_test = test_df['total_orders']
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    metrics.append({
        'location_id': location,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    })
    
    print(f"   MAE:  {mae:.2f} orders")
    print(f"   RMSE: {rmse:.2f} orders")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2:.4f}")
    
    # Store predictions
    pred_df = test_df[['date', 'total_orders']].copy()
    pred_df['predicted_orders'] = y_pred
    pred_df['location_id'] = location
    all_predictions.append(pred_df)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_df['date'], y_test, 'ko-', label='Actual', markersize=4)
    ax.plot(test_df['date'], y_pred, 'ro-', label='Predicted', markersize=4)
    ax.set_title(f'XGBoost Forecast - {location}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Orders')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'ml_models/xgboost_forecast_{location}.png', dpi=150)
    plt.close()

# Save results
all_predictions_df = pd.concat(all_predictions, ignore_index=True)
all_predictions_df.to_csv('ml_models/xgboost_predictions.csv', index=False)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('ml_models/xgboost_metrics.csv', index=False)

print("\n" + "=" * 70)
print("📊 Overall Performance Summary:")
print(metrics_df.to_string(index=False))
print(f"\n📈 Average MAPE: {metrics_df['mape'].mean():.2f}%")
print(f"📈 Average R²: {metrics_df['r2'].mean():.4f}")

print("\n✅ XGBoost forecasting complete!")
print(f"📁 Predictions saved to: ml_models/xgboost_predictions.csv")
print(f"📁 Metrics saved to: ml_models/xgboost_metrics.csv")
