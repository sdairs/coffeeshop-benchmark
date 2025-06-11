SET allow_experimental_database_glue_catalog = 1;
SET use_iceberg_partition_pruning = 1;

CREATE DATABASE glue
ENGINE = DataLakeCatalog
SETTINGS
    catalog_type = 'glue',
    region = 'us-east-1',
    aws_access_key_id = '',
    aws_secret_access_key = '';

USE glue;