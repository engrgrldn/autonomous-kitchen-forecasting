from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "autonomous-kitchen-forecast"

client = bigquery.Client(project=PROJECT_ID)

query = """
SELECT *
FROM `autonomous-kitchen-forecast.analytics.ml_forecast_features`
ORDER BY location_id, date
"""

print("📥 Downloading ML features from BigQuery...")
df = client.query(query).to_dataframe()

# Save to CSV
df.to_csv('ml_models/ml_features.csv', index=False)

print(f"✅ Exported {len(df):,} rows to ml_models/ml_features.csv")
print(f"📊 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"📍 Locations: {df['location_id'].unique().tolist()}")
