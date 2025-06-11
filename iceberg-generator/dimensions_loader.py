#!/usr/bin/env python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.sql import SparkSession
from awsglue.job import Job
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

spark = (
    SparkSession.builder
    .appName("dimensions_loader")
    .config("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.glue_catalog.warehouse", "s3://clickhouse-coffeeshop-benchmark/coffeeshop/")
    .config("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
    .config("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .getOrCreate()
)

# -------------------- Parameters --------------------
input_s3_path = "s3://clickhouse-coffeeshop-benchmark/csv/"

dim_locations_file = "dim_locations.csv"
dim_products_file = "dim_products.csv"

output_database = "coffeeshop"
output_locations_table_name = "dim_locations"
output_products_table_name = "dim_products"

# -------------------- Locations --------------------
locations_schema = StructType([
    StructField("record_id", StringType(), True),
    StructField("location_id", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("country", StringType(), True),
    StructField("region", StringType(), True)
])

locations_df = spark.read.format("csv") \
    .option("header", "true") \
    .schema(locations_schema) \
    .load(input_s3_path + dim_locations_file)

locations_target_iceberg_table = f"glue_catalog.{output_database}.{output_locations_table_name}"

locations_df.writeTo(locations_target_iceberg_table) \
    .using("iceberg") \
    .createOrReplace()

print(f"Successfully wrote data to Iceberg table: {locations_target_iceberg_table}")

# -------------------- Products --------------------

products_schema = StructType([
    StructField("record_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("subcategory", StringType(), True),
    StructField("standard_cost", DoubleType(), True),
    StructField("standard_price", DoubleType(), True),
    StructField("from_date", DateType(), True),
    StructField("to_date", DateType(), True)
])

products_df = spark.read.format("csv") \
    .option("header", "true") \
    .schema(products_schema) \
    .load(input_s3_path + dim_products_file)

products_target_iceberg_table = f"glue_catalog.{output_database}.{output_products_table_name}"

products_df.writeTo(products_target_iceberg_table) \
    .using("iceberg") \
    .createOrReplace()

print(f"Successfully wrote data to Iceberg table: {products_target_iceberg_table}")