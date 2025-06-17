create database if not exists coffeeshop;

create table if not exists coffeeshop.dim_products
engine = IcebergS3('s3://clickhouse-datalake-demo/coffeeshop/dim_products/', '', '');

create table if not exists coffeeshop.dim_locations
engine = IcebergS3('s3://clickhouse-datalake-demo/coffeeshop/dim_locations/', '', '');

create table if not exists coffeeshop.fact_sales_500m
engine = IcebergS3('s3://clickhouse-datalake-demo/coffeeshop/fact_sales_500m/', '', '');

-- create table if not exists coffeeshop.fact_sales_1b
-- engine = IcebergS3('s3://clickhouse-datalake-demo/coffeeshop/fact_sales_1b/', '', '');

-- create table if not exists coffeeshop.fact_sales_5b
-- engine = IcebergS3('s3://clickhouse-datalake-demo/coffeeshop/fact_sales_5b/', '', '');