# ClickHouse Cloud benchmark runner

This repo contains scripts to automate the benchmark on ClickHouse Cloud, using the public dataset and query suite originally used to compare Databricks and Snowflake.

## Overview

This benchmark runs a fixed set of 17 SQL queries against ClickHouse Cloud, using different service sizes and dataset scales. All queries are executed multiple times, and performance is logged alongside estimated cost based on ClickHouse Cloud pricing.

The core script is `main.sh`, which:
- (Optionally) loads data
- Runs the full benchmark query suite
- Logs runtime and cost metrics

## ⚠️ Prerequisites

Before running any benchmarks, **you must create a ClickHouse Cloud service**, and export credentials as environment variables.

### 1. Spin up a ClickHouse Cloud service

Go to [ClickHouse Cloud](https://clickhouse.com/cloud), provision a service, and note:
- Cloud provider (e.g., AWS or GCP)
- Region (e.g., `us-east-2`)
- Number of nodes, CPU, and RAM per node
- Your service host, username, and password

> Make sure your service size matches what you plan to benchmark.

### 2. Export required environment variables

These are used by all scripts to connect to your service:

```bash
export CLICKHOUSE_HOST="your-service-name.region.clickhouse.cloud"
export CLICKHOUSE_USER="your-username"
export CLICKHOUSE_PASSWORD="your-password"
```

## Running the benchmark

Once your service is ready and the environment variables are set, run the benchmark using:
```bash
./main.sh <version> <scale> <load_data?> <repeats> <nodes> <vcpus> <ram_gb> <cloud> <region> <scale_price> <enterprise_price>
```

### Parameters

| Parameter          | Example     | Description                                      |
|--------------------|-------------|--------------------------------------------------|
| `version`          | `25.4.1`    | ClickHouse version to log for the run           |
| `scale`            | `500m`      | Dataset scale: 500m, 1b, 5b                      |
| `load_data?`       | `true`      | Whether to ingest the dataset                   |
| `repeats`          | `5`         | Number of times to run each query               |
| `nodes`            | `2`         | Number of compute nodes in the service          |
| `vcpus`            | `30`        | vCPUs per node                                  |
| `ram_gb`           | `120`       | RAM per node in GB                              |
| `cloud`            | `AWS`       | Cloud provider                                  |
| `region`           | `us-east-2` | Deployment region                               |
| `scale_price`      | `0.2985`    | Price per node/hour for Scale tier              |
| `enterprise_price` | `0.3903`    | Price per node/hour for Enterprise tier         |

> You can find up-to-date compute pricing [here](https://clickhouse.com/pricing)

### Examples

Run benchmark at 500m scale on a 2-node service where each node has 30 CPU Cores and 120 GB RAM:
```bash
./main.sh 25.4.1 500m true 5 2 30 120 AWS us-east-2 0.2985 0.3903
```


Benchmark 500m, 1b, and 5b on a 16-node service, where each node has 30 CPU Cores and 120 GB RAM:
```bash
./main.sh 25.4.1 500m true 5 16 30 120 AWS us-east-2 0.2985 0.3903
./main.sh 25.4.1 1b true 5 16 30 120 AWS us-east-2 0.2985 0.3903
./main.sh 25.4.1 5b true 5 16 30 120 AWS us-east-2 0.2985 0.3903
```

##  File structure

| File            | Description                                      |
|------------------|--------------------------------------------------|
| `main.sh`        | Orchestrates the benchmark for one run           |
| `load_data.sh`   | Ingests data for selected scale                  |
| `benchmark.sh`   | Helper script for individual query tests         |
| `run_queries.sh` | Runs each query and logs runtimes                |
| `queries.sql`    | Query suite (17 SQL JOIN-heavy queries)          |
| `tables.sql`     | Table schema definitions                         |


##   Output

Each run generates a JSON result file saved under the `results/` folder with:
- Raw results for each query  
- Total runtime and cost  
- Cost per query and per second  
- Metadata for reproducibility
