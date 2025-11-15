# Autonomous Kitchen Demand Forecasting

End-to-end demand forecasting system for autonomous kitchen operations using Google Cloud Platform, dbt, and Machine Learning.

## 🎯 Project Overview

Built for **MOSTLY AI / Circus AI Project Manager Application** - demonstrates production-grade data pipeline and ML forecasting capabilities for autonomous food production systems.

## 🏗️ Architecture
```
Raw Data → BigQuery → dbt (Transform) → ML Models → Forecasts
```

## 🛠️ Tech Stack

- **Cloud Platform**: Google Cloud Platform (BigQuery, Vertex AI)
- **Data Transformation**: dbt (data build tool)
- **ML Models**: XGBoost, Prophet
- **Languages**: Python, SQL
- **Orchestration**: Airflow-ready pipeline structure

## 📊 Results

- **Dataset**: 164K+ orders across 3 locations over 3 months
- **Best Model**: XGBoost (R² = 0.20, MAE = ~60 orders)
- **Features Engineered**: 30+ features including lag features, rolling averages, external factors

## 📁 Project Structure
```
├── generate_data_optimized.py       # Synthetic data generation
├── export_ml_features.py            # Export features from BigQuery
├── kitchen_analytics/               # dbt project
│   ├── models/
│   │   ├── staging/                # Data cleaning
│   │   ├── intermediate/           # Aggregations
│   │   └── ml_features/            # ML-ready features
├── ml_models/
│   ├── prophet_forecast.py         # Time series model
│   ├── xgboost_forecast.py         # Gradient boosting model
│   └── compare_models.py           # Model comparison
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Google Cloud Platform account
- Python 3.8+
- dbt-bigquery

### Setup
1. Clone the repository
2. Set up GCP credentials
3. Run data generation
4. Execute dbt models
5. Train ML models

## 📈 Key Features

- ✅ Dimensional data modeling (star schema)
- ✅ Feature engineering with lag features and rolling averages
- ✅ Multiple ML model comparison
- ✅ Production-ready data pipeline
- ✅ Comprehensive testing and validation

## 👤 Author

**Gigi Castillo**
- LinkedIn: [Your LinkedIn]
- Email: castillo.marygeraldine@gmail.com
- Portfolio: [Your Portfolio]

## 📝 License

This project is open source and available under the MIT License.
