-- 1
WITH
    dictGet('dicts.dict_locations', 'city', f.location_id) AS city
SELECT
    f.order_date,
    city,
    SUM(f.sales_amount) AS total_sales,
    AVG(SUM(f.sales_amount)) OVER (
        PARTITION BY city
        ORDER BY f.order_date ASC
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_avg
FROM fact_sales AS f
GROUP BY
    f.order_date,
    city
ORDER BY
    city ASC,
    f.order_date ASC;

-- 2
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', f.order_date) AS sales_month,
        f.product_name,
        SUM(f.sales_amount) AS total_sales
    FROM fact_sales f
    GROUP BY
        DATE_TRUNC('month', f.order_date),
        f.product_name
)
SELECT
    sales_month,
    product_name,
    total_sales,
    RANK() OVER (PARTITION BY sales_month ORDER BY total_sales DESC) AS sales_rank
FROM monthly_sales
ORDER BY sales_month, sales_rank;

-- 3
WITH season_discount AS
(
    SELECT
        f.location_id,
        f.season,
        AVG(f.discount_percentage) AS avg_discount
    FROM fact_sales AS f
    GROUP BY
        f.location_id,
        f.season
),
with_rank AS
(
    SELECT
        dictGet('dicts.dict_locations', 'city', location_id) AS city,
        dictGet('dicts.dict_locations', 'state', location_id) AS state,
        season,
        avg_discount,
        DENSE_RANK() OVER (PARTITION BY season ORDER BY avg_discount DESC) AS discount_rank
    FROM season_discount
)
SELECT *
FROM with_rank
WHERE discount_rank <= 3
ORDER BY season, discount_rank;

-- 4
SELECT
    f.order_date,
    f.product_name,
    dictGet('dicts.dict_products', 'standard_price', f.product_name, f.order_date) AS standard_price,
    dictGet('dicts.dict_products', 'standard_cost', f.product_name, f.order_date) AS standard_cost,
    SUM(f.quantity) AS total_quantity_sold,
    SUM(f.sales_amount) AS total_sales_amount,
    (dictGet('dicts.dict_products', 'standard_price', f.product_name, f.order_date) -
     dictGet('dicts.dict_products', 'standard_cost', f.product_name, f.order_date)) * SUM(f.quantity) AS theoretical_margin
FROM fact_sales f
GROUP BY
    f.order_date,
    f.product_name
ORDER BY
    f.order_date,
    f.product_name;

-- 5
WITH daily_city_qty AS (
    SELECT
        f.order_date,
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        SUM(f.quantity) AS daily_qty
    FROM fact_sales f
    GROUP BY
        f.order_date,
        city
)
SELECT
    order_date,
    city,
    daily_qty,
    SUM(daily_qty) OVER (
        PARTITION BY city
        ORDER BY order_date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_30day_qty
FROM daily_city_qty
ORDER BY city, order_date;

-- 6
CREATE OR REPLACE TABLE query06
ORDER BY sales_month AS
WITH monthly_cat AS (
    SELECT
        DATE_TRUNC('month', f.order_date) AS sales_month,
        dictGet('dicts.dict_products', 'category', f.product_name, f.order_date) AS category,
        SUM(f.sales_amount) AS monthly_revenue
    FROM fact_sales f
    GROUP BY
        DATE_TRUNC('month', f.order_date),
        category
)
SELECT
    COALESCE(sales_month, DATE('1970-01-01')) AS sales_month,
    category,
    monthly_revenue
FROM monthly_cat;

-- 7
WITH yearly_sales AS (
    SELECT
        f.location_id,
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        dictGet('dicts.dict_locations', 'state', f.location_id) AS state,
        toYear(f.order_date) AS sales_year,
        SUM(f.sales_amount) AS total_sales_year
    FROM fact_sales f
    GROUP BY
        f.location_id,
        city,
        state,
        sales_year
)
SELECT
    city,
    state,
    SUM(CASE WHEN sales_year = 2023 THEN total_sales_year ELSE 0 END) AS sales_2023,
    SUM(CASE WHEN sales_year = 2024 THEN total_sales_year ELSE 0 END) AS sales_2024,
    SUM(CASE WHEN sales_year = 2024 THEN total_sales_year ELSE 0 END)
    - SUM(CASE WHEN sales_year = 2023 THEN total_sales_year ELSE 0 END) AS yoy_diff
FROM yearly_sales
GROUP BY
    city,
    state
ORDER BY
    city,
    state;

-- 8
WITH city_quarter_subcat AS (
    SELECT
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        dictGet('dicts.dict_products', 'subcategory', f.product_name, f.order_date) AS subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM fact_sales f
    GROUP BY
        city,
        sales_quarter,
        subcategory
)
SELECT
    city,
    sales_quarter,
    subcategory,
    total_sales,
    RANK() OVER (PARTITION BY city, sales_quarter ORDER BY total_sales DESC) AS subcat_rank
FROM city_quarter_subcat
ORDER BY city, sales_quarter, subcat_rank;

-- 9
WITH daily_discount AS (
    SELECT
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        f.order_date,
        AVG(f.discount_percentage) AS avg_discount
    FROM fact_sales f
    GROUP BY
        city,
        f.order_date
)
SELECT
    city,
    order_date,
    avg_discount,
    AVG(avg_discount) OVER (
        PARTITION BY city
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_avg_discount
FROM daily_discount
ORDER BY city, order_date;

-- 10
CREATE OR REPLACE TABLE query10
ORDER BY (city, order_date) AS
WITH daily_orders AS (
    SELECT
        f.order_date,
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        COUNT(DISTINCT f.order_id) AS daily_distinct_orders
    FROM fact_sales f
    GROUP BY
        city,
        f.order_date
)
SELECT
    COALESCE(order_date, DATE('1970-01-01')) AS order_date,
    COALESCE(city, '') AS city,
    daily_distinct_orders,
    SUM(daily_distinct_orders) OVER (
        PARTITION BY city
        ORDER BY order_date
        ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
    ) AS rolling_90d_distinct_orders
FROM daily_orders
ORDER BY city, order_date;

-- 11
WITH city_quarter_subcat AS (
    SELECT
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        dictGet('dicts.dict_products', 'subcategory', f.product_name, f.order_date) AS subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM fact_sales f
    WHERE city IN ('Charlotte', 'Houston')
      AND isNotNull(subcategory)
    GROUP BY
        city,
        sales_quarter,
        subcategory
)
SELECT
    city,
    sales_quarter,
    subcategory,
    total_sales,
    RANK() OVER (PARTITION BY city, sales_quarter ORDER BY total_sales DESC) AS subcat_rank
FROM city_quarter_subcat
ORDER BY city, sales_quarter, subcat_rank;

-- 12
WITH city_quarter_subcat AS (
    SELECT
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        dictGet('dicts.dict_products', 'subcategory', f.product_name, f.order_date) AS subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM fact_sales f
    WHERE city IN ('Charlotte', 'Houston')
      AND DATE_TRUNC('quarter', f.order_date) IN (
            DATE('2023-01-01'), DATE('2023-04-01'),
            DATE('2024-01-01'), DATE('2024-04-01')
      )
      AND isNotNull(subcategory)
    GROUP BY
        city,
        sales_quarter,
        subcategory
)
SELECT
    city,
    sales_quarter,
    subcategory,
    total_sales,
    RANK() OVER (PARTITION BY city, sales_quarter ORDER BY total_sales DESC) AS subcat_rank
FROM city_quarter_subcat
ORDER BY city, sales_quarter, subcat_rank;

-- 13
WITH city_quarter_subcat AS (
    SELECT
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        dictGet('dicts.dict_products', 'subcategory', f.product_name, f.order_date) AS subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM fact_sales f
    WHERE
        city = 'Austin'
        AND DATE_TRUNC('quarter', f.order_date) IN (
            DATE('2023-01-01'), DATE('2023-04-01'),
            DATE('2024-01-01'), DATE('2024-04-01')
        )
    GROUP BY
        city,
        sales_quarter,
        subcategory
)
SELECT
    city,
    sales_quarter,
    subcategory,
    total_sales,
    RANK() OVER (PARTITION BY city, sales_quarter ORDER BY total_sales DESC) AS subcat_rank
FROM city_quarter_subcat
ORDER BY city, sales_quarter, subcat_rank;

-- 14
WITH city_quarter_subcat AS (
    SELECT
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        dictGet('dicts.dict_products', 'subcategory', f.product_name, f.order_date) AS subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM fact_sales f
    WHERE
        DATE_TRUNC('quarter', f.order_date) IN (
            DATE('2023-01-01'), DATE('2023-04-01'),
            DATE('2024-01-01'), DATE('2024-04-01')
        )
    GROUP BY
        city,
        sales_quarter,
        subcategory
)
SELECT
    city,
    sales_quarter,
    subcategory,
    total_sales,
    RANK() OVER (PARTITION BY city, sales_quarter ORDER BY total_sales DESC) AS subcat_rank
FROM city_quarter_subcat
ORDER BY city, sales_quarter, subcat_rank;

-- 15
CREATE OR REPLACE TABLE query15
ORDER BY (city, product_name, sales_quarter) AS
WITH base_data AS (
    SELECT
        f.location_id,
        dictGet('dicts.dict_locations', 'city', f.location_id) AS city,
        f.product_name,
        toStartOfQuarter(f.order_date) AS sales_quarter,
        SUM(f.sales_amount) AS total_sales,
        SUM(f.sales_amount * (f.discount_percentage / 100.0)) AS total_discount,
        SUM(f.quantity * dictGet('dicts.dict_products', 'standard_cost', f.product_name, f.order_date)) AS total_cogs
    FROM fact_sales AS f
    WHERE f.order_date BETWEEN toDate('2022-01-01') AND toDate('2024-12-31')
    GROUP BY
        f.location_id,
        city,
        f.product_name,
        sales_quarter
),
with_profit AS (
    SELECT
        *,
        total_sales - total_discount - total_cogs AS profit
    FROM base_data
)
SELECT
    city,
    product_name,
    sales_quarter,
    profit,
    lagInFrame(profit) OVER w AS prev_profit,
    round(
        IF(prev_profit = 0 OR prev_profit IS NULL,
           NULL,
           100.0 * (profit - prev_profit) / prev_profit),
        2
    ) AS yoy_profit_pct
FROM with_profit
WINDOW w AS (
    PARTITION BY location_id, product_name
    ORDER BY sales_quarter
);

-- 16
WITH seasonal_data AS (
    SELECT
        dictGet('dicts.dict_locations', 'state', f.location_id) AS state,
        f.season,
        dictGet('dicts.dict_products', 'category', f.product_name, f.order_date) AS category,
        SUM(f.sales_amount) AS total_sales,
        SUM(f.quantity) AS total_units,
        COUNT(DISTINCT f.order_id) AS order_count
    FROM fact_sales f
    WHERE f.order_date BETWEEN toDate('2023-01-01') AND toDate('2024-06-30')
    GROUP BY
        state,
        f.season,
        category
),
ranked AS (
    SELECT
        *,
        DENSE_RANK() OVER (PARTITION BY state, season ORDER BY total_sales DESC) AS category_rank
    FROM seasonal_data
)
SELECT *
FROM ranked
WHERE category_rank <= 3
ORDER BY state, season, category_rank;

-- 17
WITH raw_agg AS (
    SELECT
        dictGet('dicts.dict_locations', 'state', f.location_id) AS state,
        f.season,
        dictGet('dicts.dict_products', 'category', f.product_name, f.order_date) AS category,
        SUM(f.sales_amount) AS total_sales,
        SUM(f.quantity) AS total_units,
        COUNT(DISTINCT f.order_id) AS order_count
    FROM fact_sales AS f
    WHERE
        dictGet('dicts.dict_locations', 'region', f.location_id) = 'West'
        AND f.time_of_day = 'Morning'
        AND f.product_name = 'Frappe'
        AND f.order_date BETWEEN toDate('2023-01-01') AND toDate('2024-06-30')
    GROUP BY
        state,
        f.season,
        category
),
seasonal_data AS (
    SELECT
        state,
        season,
        category,
        total_sales,
        total_units,
        order_count,
        SUM(total_sales) OVER (PARTITION BY state, season) AS season_total_sales
    FROM raw_agg
),
ranked AS (
    SELECT
        state,
        season,
        category,
        total_sales,
        total_units,
        order_count,
        season_total_sales,
        ROUND(100.0 * total_sales / season_total_sales, 2) AS pct_of_season,
        DENSE_RANK() OVER (
            PARTITION BY state, season
            ORDER BY total_sales DESC
        ) AS category_rank
    FROM seasonal_data
)
SELECT
    state,
    season,
    category,
    total_sales,
    total_units,
    order_count,
    pct_of_season,
    category_rank
FROM ranked
WHERE category_rank <= 3
ORDER BY
    state,
    season,
    category_rank;