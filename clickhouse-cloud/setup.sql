create database if not exists coffeeshop_native;

CREATE TABLE IF NOT EXISTS coffeeshop_native.fact_sales_500m
ENGINE = MergeTree()
ORDER BY (order_date, product_name, location_id)
AS
SELECT
    COALESCE(order_id, '') AS order_id,
    COALESCE(order_line_id, '') AS order_line_id,
    COALESCE(order_date, toDate('1970-01-01')) AS order_date,
    COALESCE(time_of_day, '') AS time_of_day,
    COALESCE(season, '') AS season,
    COALESCE(month, 0) AS month,
    COALESCE(location_id, '') AS location_id,
    COALESCE(region, '') AS region,
    COALESCE(product_name, '') AS product_name,
    COALESCE(quantity, 0) AS quantity,
    COALESCE(sales_amount, 0.0) AS sales_amount,
    COALESCE(discount_percentage, 0) AS discount_percentage,
    COALESCE(product_id, '') AS product_id
FROM icebergS3('s3://clickhouse-datalake-demo/coffeeshop/fact_sales_500m/', '', '');

CREATE TABLE IF NOT EXISTS coffeeshop_native.dim_locations
ENGINE = MergeTree()
ORDER BY (record_id)
AS
SELECT
    COALESCE(record_id, '') AS record_id,
    COALESCE(location_id, '') AS location_id,
    COALESCE(city, '') AS city,
    COALESCE(state, '') AS state,
    COALESCE(country, '') AS country,
    COALESCE(region, '') AS region
FROM icebergS3('s3://clickhouse-datalake-demo/coffeeshop/dim_locations/', '', '');

CREATE TABLE IF NOT EXISTS coffeeshop_native.dim_products
ENGINE = MergeTree()
ORDER BY (record_id)
AS
SELECT
    COALESCE(record_id, '') AS record_id,
    COALESCE(product_id, '') AS product_id,
    COALESCE(name, '') AS name,
    COALESCE(category, '') AS category,
    COALESCE(subcategory, '') AS subcategory,
    COALESCE(standard_cost, 0.0) AS standard_cost,
    COALESCE(standard_price, 0.0) AS standard_price,
    COALESCE(from_date, toDate('1970-01-01')) AS from_date,
    COALESCE(to_date, toDate('9999-12-31')) AS to_date
FROM icebergS3('s3://clickhouse-datalake-demo/coffeeshop/dim_products/', '', '');