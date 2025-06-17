-- 1
SELECT
    f.order_date,
    l.city,
    SUM(f.sales_amount) AS total_sales,
    AVG(SUM(f.sales_amount)) OVER (
        PARTITION BY l.city
        ORDER BY f.order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_avg
FROM coffeeshop.fact_sales_1b f
JOIN coffeeshop.dim_locations l
    ON f.location_id = l.location_id
GROUP BY
    f.order_date,
    l.city
ORDER BY
    l.city,
    f.order_date;

-- 2
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', f.order_date) AS sales_month,
        f.product_name,
        SUM(f.sales_amount) AS total_sales
    FROM coffeeshop.fact_sales_1b f
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
WITH season_discount AS (
    SELECT
        l.city,
        l.state,
        f.season,
        AVG(f.discount_percentage) AS avg_discount
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    GROUP BY
        l.city,
        l.state,
        f.season
)
SELECT
    city,
    state,
    season,
    avg_discount,
    discount_rank
FROM (
    SELECT
        city,
        state,
        season,
        avg_discount,
        DENSE_RANK() OVER (PARTITION BY season ORDER BY avg_discount DESC) AS discount_rank
    FROM season_discount
) t
WHERE discount_rank <= 3
ORDER BY season, discount_rank;

-- 4
SELECT
    f.order_date,
    f.product_name,
    p.standard_price,
    p.standard_cost,
    SUM(f.quantity) AS total_quantity_sold,
    SUM(f.sales_amount) AS total_sales_amount,
    (p.standard_price - p.standard_cost) * SUM(f.quantity) AS theoretical_margin
FROM coffeeshop.fact_sales_1b f
JOIN coffeeshop.dim_products p
    ON f.product_name = p.name
    AND f.order_date BETWEEN p.from_date AND p.to_date
GROUP BY
    f.order_date,
    f.product_name,
    p.standard_price,
    p.standard_cost
ORDER BY
    f.order_date,
    f.product_name;

-- 5
WITH daily_city_qty AS (
    SELECT
        f.order_date,
        l.city,
        SUM(f.quantity) AS daily_qty
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    GROUP BY
        f.order_date,
        l.city
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
CREATE OR REPLACE TABLE query06 ORDER BY sales_month AS
WITH monthly_cat AS (
    SELECT
        DATE_TRUNC('month', f.order_date) AS sales_month,
        p.category,
        SUM(f.sales_amount) AS monthly_revenue
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND f.order_date BETWEEN p.from_date AND p.to_date
    GROUP BY
        DATE_TRUNC('month', f.order_date),
        p.category
)
SELECT
    coalesce(sales_month, DATE('1970-01-01')) AS sales_month,
    category,
    monthly_revenue
FROM monthly_cat;

-- 7
WITH yearly_sales AS (
    SELECT
        l.location_id,
        l.city,
        l.state,
        YEAR(f.order_date) AS sales_year,
        SUM(f.sales_amount) AS total_sales_year
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    GROUP BY
        l.location_id,
        l.city,
        l.state,
        YEAR(f.order_date)
)
SELECT
    city,
    state,
    SUM(CASE WHEN sales_year = 2023 THEN total_sales_year ELSE 0 END) AS sales_2023,
    SUM(CASE WHEN sales_year = 2024 THEN total_sales_year ELSE 0 END) AS sales_2024,
    (SUM(CASE WHEN sales_year = 2024 THEN total_sales_year ELSE 0 END)
     - SUM(CASE WHEN sales_year = 2023 THEN total_sales_year ELSE 0 END)) AS yoy_diff
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
        l.city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        p.subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND f.order_date BETWEEN p.from_date AND p.to_date
    GROUP BY
        l.city,
        DATE_TRUNC('quarter', f.order_date),
        p.subcategory
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
        l.city,
        f.order_date,
        AVG(f.discount_percentage) AS avg_discount
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    GROUP BY
        l.city,
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
CREATE OR REPLACE TABLE query10 ORDER BY (city, order_date) AS
WITH daily_orders AS (
    SELECT
        f.order_date,
        l.city,
        COUNT(DISTINCT f.order_id) AS daily_distinct_orders
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    GROUP BY
        l.city,
        f.order_date
        
)
SELECT
    coalesce(order_date, DATE('1970-01-01')) AS order_date,
    coalesce(city, '') AS city,
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
        l.city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        p.subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND f.order_date BETWEEN p.from_date AND p.to_date
    WHERE l.city IN ('Charlotte', 'Houston')
    GROUP BY
        l.city,
        DATE_TRUNC('quarter', f.order_date),
        p.subcategory
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
        l.city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        p.subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND f.order_date BETWEEN p.from_date AND p.to_date
    WHERE l.city IN ('Charlotte', 'Houston')
      AND DATE_TRUNC('quarter', f.order_date) IN (
            DATE('2023-01-01'), DATE('2023-04-01'),
            DATE('2024-01-01'), DATE('2024-04-01')
      )
    GROUP BY
        l.city,
        DATE_TRUNC('quarter', f.order_date),
        p.subcategory
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
        l.city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        p.subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND f.order_date BETWEEN p.from_date AND p.to_date
    WHERE l.city = 'Austin'
      AND DATE_TRUNC('quarter', f.order_date) IN (
            DATE('2023-01-01'), DATE('2023-04-01'),
            DATE('2024-01-01'), DATE('2024-04-01')
      )
    GROUP BY
        l.city,
        DATE_TRUNC('quarter', f.order_date),
        p.subcategory
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
        l.city,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        p.subcategory,
        SUM(f.sales_amount) AS total_sales
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND f.order_date BETWEEN p.from_date AND p.to_date
    WHERE DATE_TRUNC('quarter', f.order_date) IN (
            DATE('2023-01-01'), DATE('2023-04-01'),
            DATE('2024-01-01'), DATE('2024-04-01')
      )
    GROUP BY
        l.city,
        DATE_TRUNC('quarter', f.order_date),
        p.subcategory
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
CREATE OR REPLACE TABLE query15 AS
WITH base_data AS (
    SELECT
        f.location_id,
        l.city,
        f.product_name,
        DATE_TRUNC('quarter', f.order_date) AS sales_quarter,
        SUM(f.sales_amount) AS total_sales,
        SUM(f.sales_amount * (f.discount_percentage / 100.0)) AS total_discount,
        SUM(f.quantity * p.standard_cost) AS total_cogs
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND f.order_date BETWEEN p.from_date AND p.to_date
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    WHERE f.order_date BETWEEN '2022-01-01' AND '2024-12-31'
    GROUP BY f.location_id, l.city, f.product_name, DATE_TRUNC('quarter', f.order_date)
),
with_profit AS (
    SELECT
        *,
        total_sales - total_discount - total_cogs AS profit
    FROM base_data
),
with_yoy AS (
    SELECT
        *,
        LAG(profit) OVER (PARTITION BY location_id, product_name ORDER BY sales_quarter) AS prev_profit,
        ROUND(
            CASE
                WHEN LAG(profit) OVER (PARTITION BY location_id, product_name ORDER BY sales_quarter) = 0 THEN NULL
                ELSE 100.0 * (profit - LAG(profit) OVER (PARTITION BY location_id, product_name ORDER BY sales_quarter)) /
                     LAG(profit) OVER (PARTITION BY location_id, product_name ORDER BY sales_quarter)
            END, 2
        ) AS yoy_profit_pct
    FROM with_profit
)
SELECT
    city,
    product_name,
    sales_quarter,
    profit,
    prev_profit,
    yoy_profit_pct
FROM with_yoy;

-- 16
WITH seasonal_data AS (
    SELECT
        l.state,
        f.season,
        p.category,
        SUM(f.sales_amount) AS total_sales,
        SUM(f.quantity) AS total_units,
        COUNT(DISTINCT f.order_id) AS order_count
    FROM coffeeshop.fact_sales_1b f
    JOIN coffeeshop.dim_products p
        ON f.product_name = p.name
        AND DATE(f.order_date) BETWEEN DATE(p.from_date) AND DATE(p.to_date)
    JOIN coffeeshop.dim_locations l
        ON f.location_id = l.location_id
    WHERE DATE(f.order_date) BETWEEN DATE('2023-01-01') AND DATE('2024-06-30')
    GROUP BY l.state, f.season, p.category
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
        l.state,
        f.season,
        p.category,
        SUM(f.sales_amount)       AS total_sales,
        SUM(f.quantity)           AS total_units,
        COUNT(DISTINCT f.order_id) AS order_count
    FROM coffeeshop.fact_sales_1b AS f
    JOIN coffeeshop.dim_products AS p
      ON f.product_id = p.product_id
     AND DATE(f.order_date) BETWEEN DATE(p.from_date) AND DATE(p.to_date)
    JOIN coffeeshop.dim_locations AS l
      ON f.location_id = l.location_id
    WHERE 
        l.region        = 'West'
      AND f.time_of_day  = 'Morning'
      AND p.name         = 'Frappe'
      AND DATE(f.order_date)  BETWEEN DATE('2023-01-01') AND DATE('2024-06-30')
    GROUP BY
        l.state,
        f.season,
        p.category
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