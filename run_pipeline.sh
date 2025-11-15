#!/bin/bash
# Master pipeline orchestrator
# In production, this would be an Airflow DAG

set -e  # Exit on error

echo "🚀 Starting Demand Forecasting Pipeline"
echo "========================================"

echo "📊 Step 1: Generate synthetic data"
python generate_data_optimized.py

echo "🔄 Step 2: Run dbt transformations"
cd kitchen_analytics
dbt run
dbt test
cd ..

echo "📥 Step 3: Export ML features"
python export_ml_features.py

echo "🤖 Step 4: Train XGBoost model"
python ml_models/xgboost_forecast.py

echo "📈 Step 5: Compare models"
python ml_models/compare_models.py

echo "✅ Pipeline complete!"
echo "📁 Results: ml_models/"
