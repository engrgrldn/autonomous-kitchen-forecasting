WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
)

SELECT
    order_date,
    location_id,
    
    -- Order counts
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(quantity) AS total_items_sold,
    
    -- Revenue metrics
    SUM(order_total) AS total_revenue,
    AVG(order_total) AS avg_order_value,
    
    -- Customer metrics
    COUNT(DISTINCT CASE WHEN customer_type = 'new' THEN order_id END) AS new_customer_orders,
    COUNT(DISTINCT CASE WHEN customer_type = 'returning' THEN order_id END) AS returning_customer_orders,
    
    -- Channel breakdown
    COUNT(DISTINCT CASE WHEN channel = 'app' THEN order_id END) AS app_orders,
    COUNT(DISTINCT CASE WHEN channel = 'kiosk' THEN order_id END) AS kiosk_orders,
    COUNT(DISTINCT CASE WHEN channel = 'web' THEN order_id END) AS web_orders,
    
    -- Category breakdown
    SUM(CASE WHEN category = 'Pizza' THEN quantity ELSE 0 END) AS pizza_quantity,
    SUM(CASE WHEN category = 'Salad' THEN quantity ELSE 0 END) AS salad_quantity,
    SUM(CASE WHEN category = 'Pasta' THEN quantity ELSE 0 END) AS pasta_quantity,
    SUM(CASE WHEN category = 'Bowl' THEN quantity ELSE 0 END) AS bowl_quantity,
    SUM(CASE WHEN category = 'Burger' THEN quantity ELSE 0 END) AS burger_quantity,
    
    -- Meal period breakdown
    COUNT(DISTINCT CASE WHEN meal_period = 'breakfast' THEN order_id END) AS breakfast_orders,
    COUNT(DISTINCT CASE WHEN meal_period = 'lunch' THEN order_id END) AS lunch_orders,
    COUNT(DISTINCT CASE WHEN meal_period = 'dinner' THEN order_id END) AS dinner_orders,
    COUNT(DISTINCT CASE WHEN meal_period = 'late_evening' THEN order_id END) AS late_evening_orders

FROM orders
GROUP BY 1, 2
