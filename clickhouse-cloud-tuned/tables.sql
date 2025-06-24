
CREATE DATABASE dicts;

CREATE DICTIONARY dicts.dict_locations
(
    location_id String,
    record_id   String,
    city        String,
    state       String,
    country     String,
    region      String
)
PRIMARY KEY location_id
SOURCE(CLICKHOUSE(
         QUERY "
            SELECT
                COALESCE(location_id, '') AS location_id,
                COALESCE(record_id, '') AS record_id,
                COALESCE(city, '') AS city,
                COALESCE(state, '') AS state,
                COALESCE(country, '') AS country,
                COALESCE(region, '') AS region
            FROM icebergS3('s3://clickhouse-datasets/coffeeshop/dim_locations/')"))
LIFETIME(0)
LAYOUT(complex_key_hashed());



CREATE OR REPLACE DICTIONARY dicts.dict_products (
    record_id String,
    product_id String,
    name String,
    category String,
    subcategory String,
    standard_cost Float64,
    standard_price Float64,
    from_date Date,
    to_date Date
)
PRIMARY KEY name
SOURCE(CLICKHOUSE(db 'coffeeshop_1b' table 'dim_products'))
LIFETIME(0)
LAYOUT(RANGE_HASHED())
RANGE(MIN from_date MAX to_date);



-- -- messes up content
-- CREATE OR REPLACE DICTIONARY dicts.dict_products (
--     record_id String,
--     product_id String,
--     name String,
--     category String,
--     subcategory String,
--     standard_cost Float64,
--     standard_price Float64,
--     from_date Date,
--     to_date Date
-- )
-- PRIMARY KEY name
-- SOURCE(CLICKHOUSE(
--          QUERY "
--             SELECT
--                 COALESCE(record_id, '') AS record_id,
--                 COALESCE(product_id, '') AS product_id,
--                 COALESCE(name, '') AS name,
--                 COALESCE(category, '') AS category,
--                 COALESCE(subcategory, '') AS subcategory,
--                 COALESCE(standard_cost, 0.0) AS standard_cost,
--                 COALESCE(standard_price, 0.0) AS standard_price,
--                 COALESCE(from_date, toDate('1970-01-01')) AS from_date,
--                 COALESCE(to_date, toDate('9999-12-31')) AS to_date
--             FROM icebergS3('s3://clickhouse-datasets/coffeeshop/dim_products/')"))
-- LIFETIME(0)
-- LAYOUT(RANGE_HASHED())
-- RANGE(MIN from_date MAX to_date);


