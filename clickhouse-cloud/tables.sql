CREATE TABLE dim_locations
(
    record_id   String,
    location_id String,
    city        String,
    state       String,
    country     String,
    region      String
)
ENGINE = MergeTree()
ORDER BY (record_id);

INSERT INTO dim_locations
SELECT
    COALESCE(record_id, '') AS record_id,
    COALESCE(location_id, '') AS location_id,
    COALESCE(city, '') AS city,
    COALESCE(state, '') AS state,
    COALESCE(country, '') AS country,
    COALESCE(region, '') AS region
FROM icebergS3('s3://clickhouse-datasets/coffeeshop/dim_locations/');


CREATE TABLE dim_products
(
    record_id      String,
    product_id     String,
    name           String,
    category       String,
    subcategory    String,
    standard_cost  Float64,
    standard_price Float64,
    from_date      Date,
    to_date        Date
)
ENGINE = MergeTree()
ORDER BY (record_id);

INSERT INTO dim_products
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
FROM icebergS3('s3://clickhouse-datasets/coffeeshop/dim_products/');


CREATE TABLE IF NOT EXISTS fact_sales
(
    order_id             String,
    order_line_id        String,
    order_date           Date,
    time_of_day          String,
    season               String,
    month                Int32,
    location_id          String,
    region               String,
    product_name         String,
    quantity             Int32,
    sales_amount         Float64,
    discount_percentage  Int32,
    product_id           String
)
ENGINE = MergeTree()
ORDER BY (order_id);

INSERT INTO fact_sales
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
FROM icebergS3('s3://clickhouse-datasets/coffeeshop/fact_sales_<SCALE>/');


