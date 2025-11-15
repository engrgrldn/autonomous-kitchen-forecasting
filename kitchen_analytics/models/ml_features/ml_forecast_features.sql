{{
    config(
        materialized='table'
    )
}}

WITH daily_orders AS (
    SELECT * FROM {{ ref('int_daily_orders') }}
),

kitchen_perf AS (
    SELECT * FROM {{ ref('int_kitchen_performance') }}
),

external AS (
    SELECT * FROM {{ ref('stg_external_factors') }}
)

SELECT
    o.order_date AS date,
    o.location_id,
    
    -- Target variable
    o.total_orders,
    
    -- Order features
    o.total_items_sold,
    o.total_revenue,
    o.avg_order_value,
    SAFE_DIVIDE(o.new_customer_orders, o.total_orders) AS new_customer_rate,
    
    -- Channel features
    o.app_orders,
    o.kiosk_orders,
    o.web_orders,
    
    -- Category features
    o.pizza_quantity,
    o.salad_quantity,
    o.pasta_quantity,
    o.bowl_quantity,
    o.burger_quantity,
    
    -- Kitchen performance features
    COALESCE(k.avg_success_rate, 0.95) AS avg_success_rate,
    COALESCE(k.total_downtime_minutes, 0) AS total_downtime_minutes,
    COALESCE(k.avg_prep_time, 10) AS avg_prep_time,
    
    -- External factors
    COALESCE(e.temperature_c, 15) AS temperature_c,
    COALESCE(e.weather_condition, 'sunny') AS weather_condition,
    COALESCE(e.precipitation_mm, 0) AS precipitation_mm,
    COALESCE(e.is_holiday, 0) AS is_holiday,
    COALESCE(e.is_weekend, 0) AS is_weekend,
    COALESCE(e.day_of_week, 0) AS day_of_week,
    COALESCE(e.week_of_year, 1) AS week_of_year,
    
    -- Lag features (computed in the same query to avoid JOIN issues)
    LAG(o.total_orders, 1) OVER (PARTITION BY o.location_id ORDER BY o.order_date) AS orders_lag_1d,
    LAG(o.total_orders, 7) OVER (PARTITION BY o.location_id ORDER BY o.order_date) AS orders_lag_7d,
    AVG(o.total_orders) OVER (
        PARTITION BY o.location_id 
        ORDER BY o.order_date 
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS orders_rolling_7d_avg,
    
    -- Temporal features
    EXTRACT(MONTH FROM o.order_date) AS month,
    EXTRACT(YEAR FROM o.order_date) AS year,
    DATE_DIFF(o.order_date, DATE('2024-08-01'), DAY) AS days_since_start

FROM daily_orders o
LEFT JOIN kitchen_perf k 
    ON o.order_date = k.date 
    AND o.location_id = k.location_id
LEFT JOIN external e 
    ON o.order_date = e.date 
    AND o.location_id = e.location_id
