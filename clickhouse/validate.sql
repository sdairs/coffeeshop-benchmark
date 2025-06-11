--This is not neccesary, but a good sanity check.
SELECT
  sales.order_date,
  products.name,
  locations.city,
  sales.time_of_day,
  products.category,
  SUM(sales.quantity) AS quantity,
  SUM(sales.sales_amount) AS revenue,
  COUNT(*) AS row_count
FROM coffeeshop_native.fact_sales_500m sales
JOIN coffeeshop_native.dim_locations locations
  ON sales.location_id = locations.location_id
JOIN coffeeshop_native.dim_products products
  ON sales.product_id = products.product_id
  AND sales.order_date BETWEEN DATE(products.from_date) AND DATE(products.to_date)
WHERE sales.order_date BETWEEN DATE('2023-01-01') AND DATE('2024-12-31')
GROUP BY
  sales.order_date,
  products.name,
  locations.city,
  sales.time_of_day,
  products.category
