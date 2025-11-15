WITH source AS (
    SELECT * FROM {{ source('raw_data', 'kitchen_operations_raw') }}
),

cleaned AS (
    SELECT
        TIMESTAMP(timestamp) AS timestamp,
        location_id,
        robot_id,
        orders_completed,
        avg_prep_time_minutes,
        success_rate,
        error_count,
        downtime_minutes,
        ingredient_waste_kg,
        energy_consumption_kwh,
        maintenance_required,
        
        -- Derived fields
        DATE(timestamp) AS operation_date,
        EXTRACT(HOUR FROM timestamp) AS operation_hour
        
    FROM source
    WHERE timestamp IS NOT NULL
        AND location_id IS NOT NULL
)

SELECT * FROM cleaned
