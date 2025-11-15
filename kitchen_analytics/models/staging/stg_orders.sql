WITH source AS (
    SELECT * FROM {{ source('raw_data', 'orders_raw') }}
),

cleaned AS (
    SELECT
        order_id,
        TIMESTAMP(order_timestamp) AS order_timestamp,
        location_id,
        item_id,
        item_name,
        category,
        quantity,
        unit_price,
        order_total,
        channel,
        customer_type,
        preparation_time_minutes,
        
        -- Derived temporal fields
        DATE(order_timestamp) AS order_date,
        EXTRACT(HOUR FROM order_timestamp) AS order_hour,
        EXTRACT(DAYOFWEEK FROM order_timestamp) AS day_of_week,
        EXTRACT(WEEK FROM order_timestamp) AS week_of_year,
        EXTRACT(MONTH FROM order_timestamp) AS month,
        EXTRACT(YEAR FROM order_timestamp) AS year,
        
        -- Meal period classification
        CASE 
            WHEN EXTRACT(HOUR FROM order_timestamp) BETWEEN 6 AND 10 THEN 'breakfast'
            WHEN EXTRACT(HOUR FROM order_timestamp) BETWEEN 11 AND 14 THEN 'lunch'
            WHEN EXTRACT(HOUR FROM order_timestamp) BETWEEN 17 AND 21 THEN 'dinner'
            WHEN EXTRACT(HOUR FROM order_timestamp) BETWEEN 21 AND 23 THEN 'late_evening'
            ELSE 'night'
        END AS meal_period,
        
        -- Weekend flag
        CASE 
            WHEN EXTRACT(DAYOFWEEK FROM order_timestamp) IN (1, 7) THEN TRUE
            ELSE FALSE
        END AS is_weekend
        
    FROM source
    WHERE order_timestamp IS NOT NULL
        AND location_id IS NOT NULL
        AND order_total > 0
)

SELECT * FROM cleaned
