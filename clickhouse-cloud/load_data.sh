#!/bin/bash
set -e

# Required environment variables
: "${CLICKHOUSE_HOST:?CLICKHOUSE_HOST is required}"
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER is required}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD is required}"

: "${CREATE_FILE:=tables.sql}"  # Defaults to tables.sql if unset

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 {500m|1b|5b}"
  exit 1
fi

SCALE=$1


DB_NAME="coffeeshop_${SCALE}"

echo "Creating database: $DB_NAME"
clickhouse client \
  --host "$CLICKHOUSE_HOST" \
  --user "$CLICKHOUSE_USER" \
  --password "$CLICKHOUSE_PASSWORD" \
  --secure \
  --query "CREATE DATABASE IF NOT EXISTS $DB_NAME"

echo "Creating tables in $DB_NAME"

# Replace <SCALE> in SQL file and pipe into clickhouse-client
sed "s|<SCALE>|$SCALE|g" "$CREATE_FILE" | clickhouse client \
  --host "$CLICKHOUSE_HOST" \
  --user "$CLICKHOUSE_USER" \
  --password "$CLICKHOUSE_PASSWORD" \
  --secure \
  --database "$DB_NAME" \
  --multiquery

echo "✅ Loaded tables for $DB_NAME (scale: $SCALE)"