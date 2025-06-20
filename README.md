# Coffee Shop Benchmark

This is a modified version of the [Coffee Shop Benchmark](https://github.com/JosueBogran/coffeeshopdatageneratorv2) by Josue Bogran.

The data generator has been updated to run on AWS Glue ETL PySpark, using AWS Glue catalog and outputting Iceberg tables. You can generate the data yourself, or download it from the public bucket below.

## Get the data

If you want to run the benchmark on your own system, the data is available as Iceberg tables on the public bucket below:

```
s3://clickhouse-datasets/coffeeshop/dim_locations
s3://clickhouse-datasets/coffeeshop/dim_products
s3://clickhouse-datasets/coffeeshop/fact_sales_500m
s3://clickhouse-datasets/coffeeshop/fact_sales_1b
s3://clickhouse-datasets/coffeeshop/fact_sales_5b
```

The bucket is provided by [ClickHouse](https://clickhouse.com).

## Note on Snowflake & Databricks results

These are the results from the original benchmark posted to LinkedIn by the original author (see [credits](#credits)). They have been extracted from the images and put into CSVs so we can replot them. See the original post for methodology on those results. Any differences between the original results and the results in this repository are human error when extracting (if you see something, please let me know so I can fix it).

## Credits

Original idea & code by Josue Bogran
- [Coffee Shop Benchmark](https://github.com/JosueBogran/coffeeshopdatageneratorv2)
- [Databricks vs Snowflake SQL Performance Test Day 1](https://www.linkedin.com/pulse/databricks-vs-snowflake-sql-performance-test-day-1-721m-bogran-lsboe/)
- [Databricks vs Snowflake Gen 1 & 2 SQL Performance Test Day 2](https://www.linkedin.com/pulse/databricks-vs-snowflake-gen-1-2-sql-performance-test-day-bogran-ddmhe/)
