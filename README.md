# Coffee Shop Benchmark

This is a modified version of the [Coffee Shop Benchmark](https://github.com/JosueBogran/coffeeshopdatageneratorv2) by Josue Bogran.

The data generator has been updated to run on AWS Glue ETL PySpark, using AWS Glue catalog and outputting Iceberg tables.

## Directory structure

Each database tested has its own directory, with the following structure:

- `setup.sql` - The setup script to change settings, load data, etc.
- `validate.sql` - The validation queries to check the data is correct.
- `queries.sql` - The set of 17 benchmarking queries.
- `results` - The directory containing the results of the benchmarking queries.
- `results/{size}.results.csv` - The results of the benchmarking queries for a specific compute size.

If testing the same database with different storage formats (e.g. ClickHouse with MergeTree vs ClickHouse with Iceberg), add them as different top level directories (e.g. `clickhouse` and `clickhouse_iceberg`).

## Contributions

If you'd like to add another database, please submit a PR that adds a new directory for the database with the updated queries and results.

## Credits

Original idea & code by Josue Bogran
- [Coffee Shop Benchmark](https://github.com/JosueBogran/coffeeshopdatageneratorv2)
- [Databricks vs Snowflake SQL Performance Test Day 1](https://www.linkedin.com/pulse/databricks-vs-snowflake-sql-performance-test-day-1-721m-bogran-lsboe/)
- [Databricks vs Snowflake Gen 1 & 2 SQL Performance Test Day 2](https://www.linkedin.com/pulse/databricks-vs-snowflake-gen-1-2-sql-performance-test-day-bogran-ddmhe/)
