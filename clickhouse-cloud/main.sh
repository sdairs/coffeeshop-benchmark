#!/bin/bash

set -euo pipefail

# Required environment variables
: "${CLICKHOUSE_HOST:?CLICKHOUSE_HOST is required}"
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER is required}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD is required}"

# Usage check
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <SCALE>"
    echo ""
    echo "Available SCALE options:"
    echo "  500m    Run benchmark on 500 million rows scale"
    echo "  1b      Run benchmark on 1 billion rows scale"
    echo "  5b      Run benchmark on 5 billion rows scale"
    exit 1
fi


# Input
SCALE="$1"
DB_NAME="coffeeshop_${SCALE}"
REPEATS=5

if [[ "$SCALE" != "500m" && "$SCALE" != "1b" && "$SCALE" != "5b" ]]; then
    echo "❌ Invalid SCALE: $SCALE"
    echo "Valid options are: 500m, 1b, 5b"
    exit 1
fi

# Internal variables
SERVER_SYSTEM="ClickHouse Cloud 25.4 30 vCPU and 120 GiB per replica / 3 replicas"
DATE_TODAY=$(date +"%Y-%m-%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SKIP_LOAD=false

# Ensure results directory exists
mkdir -p results

# Filenames
RESULT_FILE_RUNTIMES="results/result_runtime_${SCALE}_${TIMESTAMP}.json"
RESULT_FILE_JSON="results/result_${SCALE}_${TIMESTAMP}.json"

# Optional data loading
if [[ "$SKIP_LOAD" != "true" ]]; then
    echo "📦 Loading data for scale $SCALE..."
    ./load_data.sh "$SCALE"
else
    echo "⚠️ Skipping data load (SKIP_LOAD=$SKIP_LOAD)"
fi

# Run benchmark
echo "🚀 Running benchmarks..."
./benchmark.sh "$DB_NAME" "$REPEATS" "$RESULT_FILE_RUNTIMES"

# Read runtime results
RUNTIMES=$(cat "$RESULT_FILE_RUNTIMES")

# Compute fastest times and sum
FASTEST=$(echo "$RUNTIMES" | jq '[.[] | min]')
SUM_FASTEST=$(echo "$FASTEST" | jq 'add | tonumber')

# Write final pretty JSON
jq -n \
  --arg server_system "$SERVER_SYSTEM" \
  --arg date "$DATE_TODAY" \
  --arg scale "$SCALE" \
  --argjson result "$RUNTIMES" \
  --argjson fastest "$FASTEST" \
  --argjson sum_of_fastest "$SUM_FASTEST" \
  '{
    server_system: $server_system,
    date: $date,
    scale: $scale,
    result: $result,
    fastest: $fastest,
    sum_of_fastest: ($sum_of_fastest | tonumber)
  }' > "$RESULT_FILE_JSON"

# Clean up intermediate runtime file
rm -f "$RESULT_FILE_RUNTIMES"

echo "✅ JSON results written to $RESULT_FILE_JSON"
