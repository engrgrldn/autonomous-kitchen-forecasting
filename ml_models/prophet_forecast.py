import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🚀 Prophet Demand Forecasting Model")
print("=" * 70)

# Load data
df = pd.read_csv('ml_models/ml_features.csv')
df['date'] = pd.to_datetime(df['date'])

# Store results for all locations
all_forecasts = []
metrics = []

# Train model for each location
for location in df['location_id'].unique():
    print(f"\n📍 Training model for {location}")
    
    # Filter data for this location
    location_df = df[df['location_id'] == location].copy()
    
    # Prepare data for Prophet (needs 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({
        'ds': location_df['date'],
        'y': location_df['total_orders']
    })
    
    # Add external regressors
    prophet_df['temperature'] = location_df['temperature_c'].values
    prophet_df['is_weekend'] = location_df['is_weekend'].values
    prophet_df['is_holiday'] = location_df['is_holiday'].values
    
    # Split train/test (last 14 days for testing)
    train_size = len(prophet_df) - 14
    train_df = prophet_df[:train_size]
    test_df = prophet_df[train_size:]
    
    # Initialize and train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    # Add regressors
    model.add_regressor('temperature')
    model.add_regressor('is_weekend')
    model.add_regressor('is_holiday')
    
    model.fit(train_df)
    
    # Make predictions on test set
    forecast = model.predict(test_df)
    
    # Calculate metrics
    mae = mean_absolute_error(test_df['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))
    mape = np.mean(np.abs((test_df['y'] - forecast['yhat']) / test_df['y'])) * 100
    r2 = r2_score(test_df['y'], forecast['yhat'])
    
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
    
    # Forecast next 14 days
    future = model.make_future_dataframe(periods=14, freq='D')
    
    # Add regressor values for future dates
    future['temperature'] = prophet_df['temperature'].mean()
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    future['is_holiday'] = 0
    
    future_forecast = model.predict(future)
    
    # Store forecast
    forecast_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14)
    forecast_df['location_id'] = location
    all_forecasts.append(forecast_df)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(prophet_df['ds'], prophet_df['y'], 'ko-', label='Actual', markersize=3)
    ax.plot(test_df['ds'], forecast['yhat'], 'ro-', label='Test Predictions', markersize=4)
    
    future_dates = forecast_df['ds']
    ax.plot(future_dates, forecast_df['yhat'], 'go-', label='Future Forecast', markersize=4)
    ax.fill_between(future_dates, forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.3, color='green')
    
    ax.set_title(f'Demand Forecast - {location}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Orders')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'ml_models/prophet_forecast_{location}.png', dpi=150)
    plt.close()

# Save all forecasts
all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
all_forecasts_df.to_csv('ml_models/prophet_forecasts.csv', index=False)

# Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('ml_models/prophet_metrics.csv', index=False)

print("\n" + "=" * 70)
print("📊 Overall Performance Summary:")
print(metrics_df.to_string(index=False))
print(f"\n📈 Average MAPE: {metrics_df['mape'].mean():.2f}%")
print(f"📈 Average R²: {metrics_df['r2'].mean():.4f}")

print("\n✅ Prophet forecasting complete!")
print(f"📁 Forecasts saved to: ml_models/prophet_forecasts.csv")
print(f"📁 Metrics saved to: ml_models/prophet_metrics.csv")
print(f"📁 Charts saved to: ml_models/prophet_forecast_*.png")
