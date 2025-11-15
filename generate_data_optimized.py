import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
import random

PROJECT_ID = "autonomous-kitchen-forecast"
DATASET_ID = "raw_data"

START_DATE = datetime(2024, 8, 1)
END_DATE = datetime(2024, 10, 31)
LOCATIONS = ['MUC_001', 'BER_001', 'HAM_001']

MENU_ITEMS = [
    {'item_id': 'PIZZA_001', 'name': 'Margherita Pizza', 'category': 'Pizza', 'price': 12.90, 'prep_time': 12},
    {'item_id': 'PIZZA_002', 'name': 'Pepperoni Pizza', 'category': 'Pizza', 'price': 14.90, 'prep_time': 12},
    {'item_id': 'SALAD_001', 'name': 'Caesar Salad', 'category': 'Salad', 'price': 9.90, 'prep_time': 8},
    {'item_id': 'PASTA_001', 'name': 'Pasta Carbonara', 'category': 'Pasta', 'price': 11.90, 'prep_time': 10},
    {'item_id': 'BOWL_001', 'name': 'Veggie Bowl', 'category': 'Bowl', 'price': 10.90, 'prep_time': 9},
    {'item_id': 'BURGER_001', 'name': 'Classic Burger', 'category': 'Burger', 'price': 13.90, 'prep_time': 11},
]

def generate_orders_data(start_date, end_date):
    print("🔄 Generating orders data...")
    dates = pd.date_range(start_date, end_date, freq='30min')
    orders = []
    
    for timestamp in dates:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        if 11 <= hour < 14:
            base_demand = random.randint(20, 30)
        elif 17 <= hour < 21:
            base_demand = random.randint(25, 35)
        else:
            base_demand = random.randint(2, 8)
        
        if day_of_week in [4, 5]:
            base_demand = int(base_demand * 1.3)
        
        for location_id in LOCATIONS:
            num_orders = max(0, int(np.random.poisson(base_demand)))
            
            for _ in range(num_orders):
                item = random.choice(MENU_ITEMS)
                quantity = random.choice([1, 2])
                
                orders.append({
                    'order_id': f"ORD_{timestamp.strftime('%Y%m%d%H%M')}_{location_id}_{random.randint(1000, 9999)}",
                    'order_timestamp': timestamp,
                    'location_id': location_id,
                    'item_id': item['item_id'],
                    'item_name': item['name'],
                    'category': item['category'],
                    'quantity': quantity,
                    'unit_price': item['price'],
                    'order_total': round(item['price'] * quantity, 2),
                    'channel': random.choice(['app', 'kiosk', 'web']),
                    'customer_type': random.choice(['new', 'returning']),
                    'preparation_time_minutes': item['prep_time'] + random.randint(-2, 3),
                })
    
    df = pd.DataFrame(orders)
    print(f"   ✅ Generated {len(df):,} orders")
    return df

def generate_kitchen_operations(start_date, end_date):
    print("🔄 Generating kitchen operations data...")
    dates = pd.date_range(start_date, end_date, freq='h')
    operations = []
    
    for timestamp in dates:
        for location_id in LOCATIONS:
            has_failure = random.random() < 0.05
            
            operations.append({
                'timestamp': timestamp,
                'location_id': location_id,
                'robot_id': f"{location_id}_ROBOT_1",
                'orders_completed': random.randint(10, 30) if not has_failure else random.randint(0, 5),
                'avg_prep_time_minutes': round(random.uniform(8, 12), 2),
                'success_rate': round(random.uniform(0.95, 0.99), 4) if not has_failure else round(random.uniform(0.70, 0.85), 4),
                'error_count': random.randint(0, 2) if not has_failure else random.randint(3, 10),
                'downtime_minutes': 0 if not has_failure else random.randint(15, 60),
                'ingredient_waste_kg': round(random.uniform(0.5, 2.0), 2),
                'energy_consumption_kwh': round(random.uniform(4.0, 6.0), 2),
                'maintenance_required': 1 if has_failure else 0,
            })
    
    df = pd.DataFrame(operations)
    print(f"   ✅ Generated {len(df):,} operation records")
    return df

def generate_external_factors(start_date, end_date):
    print("🔄 Generating external factors...")
    dates = pd.date_range(start_date, end_date, freq='D')
    factors = []
    
    holidays = ['2024-10-03']
    
    for date in dates:
        for location_id in LOCATIONS:
            factors.append({
                'date': date.date(),
                'location_id': location_id,
                'temperature_c': round(random.uniform(10, 20), 1),
                'weather_condition': random.choice(['sunny', 'cloudy', 'rainy']),
                'precipitation_mm': round(random.uniform(0, 10), 1) if random.random() < 0.3 else 0,
                'is_holiday': 1 if date.strftime('%Y-%m-%d') in holidays else 0,
                'is_school_holiday': 0,
                'has_local_event': 1 if random.random() < 0.08 else 0,
                'day_of_week': date.dayofweek,
                'week_of_year': date.isocalendar()[1],
                'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
            })
    
    df = pd.DataFrame(factors)
    print(f"   ✅ Generated {len(df):,} external factor records")
    return df

def load_to_bigquery(df, table_id):
    print(f"📊 Loading to {table_id}...")
    client = bigquery.Client(project=PROJECT_ID)
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"   ✅ Loaded {len(df):,} rows")

if __name__ == "__main__":
    print("🚀 Autonomous Kitchen Forecasting - Data Generation (Optimized)")
    print("=" * 70)
    
    orders_df = generate_orders_data(START_DATE, END_DATE)
    operations_df = generate_kitchen_operations(START_DATE, END_DATE)
    factors_df = generate_external_factors(START_DATE, END_DATE)
    
    print("\n💾 Saving locally...")
    orders_df.to_csv('data_orders.csv', index=False)
    operations_df.to_csv('data_operations.csv', index=False)
    factors_df.to_csv('data_factors.csv', index=False)
    print("   ✅ Saved to CSV files")
    
    print("\n📊 Loading to BigQuery...")
    print("=" * 70)
    
    load_to_bigquery(orders_df, f"{PROJECT_ID}.{DATASET_ID}.orders_raw")
    load_to_bigquery(operations_df, f"{PROJECT_ID}.{DATASET_ID}.kitchen_operations_raw")
    load_to_bigquery(factors_df, f"{PROJECT_ID}.{DATASET_ID}.external_factors_raw")
    
    print("\n✅ Data generation complete!")
    print("=" * 70)
    print(f"   Orders: {len(orders_df):,} records")
    print(f"   Operations: {len(operations_df):,} records")
    print(f"   External Factors: {len(factors_df):,} records")
    print("\n🎯 Next: Run dbt models to transform the data")
