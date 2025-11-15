import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
import random

# Configuration
PROJECT_ID = "autonomous-kitchen-forecast"
DATASET_ID = "raw_data"

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 10, 31)

LOCATIONS = {
    'MUC_001': {'city': 'Munich', 'lat': 48.1351, 'lon': 11.5820},
    'BER_001': {'city': 'Berlin', 'lat': 52.5200, 'lon': 13.4050},
    'HAM_001': {'city': 'Hamburg', 'lat': 53.5511, 'lon': 9.9937},
    'MUC_002': {'city': 'Munich', 'lat': 48.1549, 'lon': 11.5418},
    'BER_002': {'city': 'Berlin', 'lat': 52.5167, 'lon': 13.3833},
}

MENU_ITEMS = [
    {'item_id': 'PIZZA_001', 'name': 'Margherita Pizza', 'category': 'Pizza', 'price': 12.90, 'prep_time': 12},
    {'item_id': 'PIZZA_002', 'name': 'Pepperoni Pizza', 'category': 'Pizza', 'price': 14.90, 'prep_time': 12},
    {'item_id': 'SALAD_001', 'name': 'Caesar Salad', 'category': 'Salad', 'price': 9.90, 'prep_time': 8},
    {'item_id': 'PASTA_001', 'name': 'Pasta Carbonara', 'category': 'Pasta', 'price': 11.90, 'prep_time': 10},
    {'item_id': 'BOWL_001', 'name': 'Veggie Bowl', 'category': 'Bowl', 'price': 10.90, 'prep_time': 9},
    {'item_id': 'BURGER_001', 'name': 'Classic Burger', 'category': 'Burger', 'price': 13.90, 'prep_time': 11},
]

def generate_orders_data(start_date, end_date):
    """Generate realistic order data"""
    print(" Generating orders data...")
    
    dates = pd.date_range(start_date, end_date, freq='15min')
    orders = []
    
    for timestamp in dates:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        month = timestamp.month
        
        # Demand patterns
        if 6 <= hour < 10:
            base_demand = random.randint(3, 8)
        elif 11 <= hour < 14:
            base_demand = random.randint(20, 40)
        elif 17 <= hour < 21:
            base_demand = random.randint(25, 45)
        elif 21 <= hour < 23:
            base_demand = random.randint(5, 12)
        else:
            base_demand = random.randint(0, 3)
        
        # Weekend boost
        if day_of_week in [4, 5]:
            base_demand = int(base_demand * 1.4)
        elif day_of_week == 6:
            base_demand = int(base_demand * 1.2)
        
        # Seasonal
        if month in [6, 7, 8]:
            base_demand = int(base_demand * 1.15)
        
        for location_id in LOCATIONS.keys():
            num_orders = max(0, int(np.random.poisson(base_demand * random.uniform(0.8, 1.2))))
            
            for _ in range(num_orders):
                item = random.choice(MENU_ITEMS)
                quantity = random.choices([1, 2, 3], weights=[0.7, 0.25, 0.05])[0]
                
                order = {
                    'order_id': f"ORD_{timestamp.strftime('%Y%m%d%H%M')}_{location_id}_{random.randint(1000, 9999)}",
                    'order_timestamp': timestamp + timedelta(seconds=random.randint(0, 899)),
                    'location_id': location_id,
                    'item_id': item['item_id'],
                    'item_name': item['name'],
                    'category': item['category'],
                    'quantity': quantity,
                    'unit_price': item['price'],
                    'order_total': round(item['price'] * quantity, 2),
                    'channel': random.choices(['app', 'kiosk', 'web'], weights=[0.6, 0.3, 0.1])[0],
                    'customer_type': random.choices(['new', 'returning'], weights=[0.3, 0.7])[0],
                    'preparation_time_minutes': item['prep_time'] + random.randint(-2, 3),
                }
                orders.append(order)
    
    df = pd.DataFrame(orders)
    print(f"    Generated {len(df):,} orders")
    return df

def generate_kitchen_operations(start_date, end_date):
    """Generate kitchen robot performance data"""
    print(" Generating kitchen operations data...")
    
    dates = pd.date_range(start_date, end_date, freq='H')
    operations = []
    
    for timestamp in dates:
        for location_id in LOCATIONS.keys():
            num_robots = random.randint(2, 3)
            
            for robot_num in range(1, num_robots + 1):
                has_failure = random.random() < 0.05
                degraded = random.random() < 0.10
                
                operation = {
                    'timestamp': timestamp,
                    'location_id': location_id,
                    'robot_id': f"{location_id}_ROBOT_{robot_num}",
                    'orders_completed': random.randint(5, 25) if not has_failure else random.randint(0, 5),
                    'avg_prep_time_minutes': round(random.uniform(8, 12), 2) * (1.5 if degraded else 1.0),
                    'success_rate': round(random.uniform(0.95, 0.99), 4) if not has_failure else round(random.uniform(0.60, 0.85), 4),
                    'error_count': random.randint(0, 2) if not has_failure else random.randint(3, 10),
                    'downtime_minutes': 0 if not has_failure else random.randint(15, 120),
                    'ingredient_waste_kg': round(random.uniform(0.1, 1.5), 2) * (2.0 if has_failure else 1.0),
                    'energy_consumption_kwh': round(random.uniform(3.5, 6.5), 2),
                    'maintenance_required': 1 if (has_failure or degraded) else 0,
                }
                operations.append(operation)
    
    df = pd.DataFrame(operations)
    print(f"    Generated {len(df):,} operation records")
    return df

def generate_external_factors(start_date, end_date):
    """Generate weather and events"""
    print(" Generating external factors...")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    factors = []
    
    holidays = ['2024-01-01', '2024-05-01', '2024-10-03', '2024-12-25', '2024-12-26',
                '2025-01-01', '2025-05-01', '2025-10-03', '2025-12-25', '2025-12-26']
    
    for date in dates:
        for location_id, location_info in LOCATIONS.items():
            month = date.month
            
            # Seasonal weather
            if month in [12, 1, 2]:
                temp_range = (-5, 8)
                weather_probs = [0.2, 0.3, 0.3, 0.2]
            elif month in [3, 4, 5]:
                temp_range = (5, 18)
                weather_probs = [0.4, 0.3, 0.25, 0.05]
            elif month in [6, 7, 8]:
                temp_range = (15, 30)
                weather_probs = [0.6, 0.25, 0.15, 0.0]
            else:
                temp_range = (8, 18)
                weather_probs = [0.35, 0.35, 0.25, 0.05]
            
            factor = {
                'date': date.date(),
                'location_id': location_id,
                'city': location_info['city'],
                'temperature_c': round(random.uniform(*temp_range), 1),
                'weather_condition': random.choices(['sunny', 'cloudy', 'rainy', 'snowy'], weights=weather_probs)[0],
                'precipitation_mm': round(random.uniform(0, 15), 1) if random.random() < 0.3 else 0,
                'is_holiday': 1 if date.strftime('%Y-%m-%d') in holidays else 0,
                'is_school_holiday': 1 if month in [7, 8, 12] else 0,
                'has_local_event': 1 if random.random() < 0.08 else 0,
                'day_of_week': date.dayofweek,
                'week_of_year': date.isocalendar()[1],
                'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
            }
            factors.append(factor)
    
    df = pd.DataFrame(factors)
    print(f"    Generated {len(df):,} external factor records")
    return df

def load_to_bigquery(df, table_id, description):
    """Load DataFrame to BigQuery"""
    print(f" Loading to {table_id}...")
    
    client = bigquery.Client(project=PROJECT_ID)
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    print(f"    Loaded {len(df):,} rows")

# Main execution
if __name__ == "__main__":
    print(" Autonomous Kitchen Forecasting - Data Generation")
    print("=" * 60)
    
    # Generate datasets
    orders_df = generate_orders_data(START_DATE, END_DATE)
    operations_df = generate_kitchen_operations(START_DATE, END_DATE)
    factors_df = generate_external_factors(START_DATE, END_DATE)
    
    print("\n Loading to BigQuery...")
    print("=" * 60)
    
    # Load to BigQuery
    load_to_bigquery(orders_df, f"{PROJECT_ID}.{DATASET_ID}.orders_raw", "Raw orders data")
    load_to_bigquery(operations_df, f"{PROJECT_ID}.{DATASET_ID}.kitchen_operations_raw", "Kitchen operations")
    load_to_bigquery(factors_df, f"{PROJECT_ID}.{DATASET_ID}.external_factors_raw", "External factors")
    
    print("\n Data generation complete!")
    print("=" * 60)
    print(f"   Orders: {len(orders_df):,} records")
    print(f"   Operations: {len(operations_df):,} records")
    print(f"   External Factors: {len(factors_df):,} records")
    print("\n Next: Run dbt models to transform the data")
