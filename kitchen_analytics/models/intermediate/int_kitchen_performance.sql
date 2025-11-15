WITH operations AS (
    SELECT * FROM {{ ref('stg_kitchen_operations') }}
)

SELECT
    operation_date AS date,
    location_id,
    
    -- Performance metrics
    AVG(success_rate) AS avg_success_rate,
    SUM(orders_completed) AS total_orders_completed,
    SUM(downtime_minutes) AS total_downtime_minutes,
    SUM(error_count) AS total_errors,
    AVG(avg_prep_time_minutes) AS avg_prep_time,
    
    -- Resource metrics
    SUM(ingredient_waste_kg) AS total_waste_kg,
    AVG(energy_consumption_kwh) AS avg_energy_consumption,
    SUM(maintenance_required) AS maintenance_events,
    
    -- Efficiency score (0-100)
    ROUND(AVG(success_rate) * 100, 2) AS efficiency_score

FROM operations
GROUP BY 1, 2
