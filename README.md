# Autonomous Kitchen Demand Forecasting

End-to-end demand forecasting system for autonomous kitchen operations using Google Cloud Platform, dbt, and Machine Learning.

## Project Overview

demonstrates production-grade data pipeline and ML forecasting capabilities for autonomous food production systems.

## Architecture
```
Raw Data → BigQuery → dbt (Transform) → ML Models → Forecasts
```

## Tech Stack

- **Cloud Platform**: Google Cloud Platform (BigQuery, Vertex AI)
- **Data Transformation**: dbt (data build tool)
- **ML Models**: XGBoost, Prophet
- **Languages**: Python, SQL
- **Orchestration**: Airflow-ready pipeline structure

## Results

- **Dataset**: 164K+ orders across 3 locations over 3 months
- **Best Model**: XGBoost (R² = 0.20, MAE = ~60 orders)
- **Features Engineered**: 30+ features including lag features, rolling averages, external factors

## Project Structure
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

## Getting Started

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

## Key Features

-  Dimensional data modeling (star schema)
-  Feature engineering with lag features and rolling averages
-  Multiple ML model comparison<img width="2250" height="750" alt="model_comparison" src="https://github.com/user-attachments/assets/a7dd6e8d-dd51-4845-90c0-b35a3a799b5e" />

-  Production-ready data pipeline
-  Comprehensive testing and validation

## 🏗️ Data Architecture

### High-Level Architecture
```mermaid
graph LR
    A[Data Generation] --> B[BigQuery Raw]
    B --> C[dbt Transformations]
    C --> D[ML Features]
    D --> E[ML Models]
    E --> F[Forecasts & Metrics]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#fce4ec
    style F fill:#e0f2f1
```

### Data Flow Layers

| Layer | Tools | Purpose | Output |
|-------|-------|---------|--------|
| **Ingestion** | Python | Generate synthetic data | 164K+ orders |
| **Storage** | BigQuery | Cloud data warehouse | Raw tables |
| **Transform** | dbt | Clean & aggregate | 276 daily records |
| **Features** | SQL + dbt | Feature engineering | 30+ ML features |
| **Model** | Python (XGBoost, Prophet) | Demand forecasting | Predictions |
| **Serve** | CSV, Charts | Business insights | Metrics & viz |

## 👤 Author

**Geraldine Castillo**
[- LinkedIn: (https://www.linkedin.com/in/engrgrldn/)

## 📝 License

This project is open source and available under the MIT License.
