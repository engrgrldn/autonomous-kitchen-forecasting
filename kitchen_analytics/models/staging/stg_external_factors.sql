WITH source AS (
    SELECT * FROM {{ source('raw_data', 'external_factors_raw') }}
)

SELECT
    DATE(date) AS date,
    location_id,
    temperature_c,
    weather_condition,
    precipitation_mm,
    is_holiday,
    is_school_holiday,
    has_local_event,
    day_of_week,
    week_of_year,
    is_weekend
FROM source
WHERE date IS NOT NULL
    AND location_id IS NOT NULL
