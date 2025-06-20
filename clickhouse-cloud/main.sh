#!/bin/bash

set -euo pipefail

SKIP_LOAD=true
REPEATS=5

# Internal variables
CSP="AWS"
REGION="us-east-2"
TIER_SCALE_PRICE_PER_UNIT_PER_HR="0.2985"
TIER_ENTERPRISE_PRICE_PER_UNIT_PER_HR="0.3903"
DATE_TODAY=$(date +"%Y-%m-%d")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Required environment variables
: "${CLICKHOUSE_HOST:?CLICKHOUSE_HOST is required}"
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER is required}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD is required}"

# Usage check
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <SCALE> [NUM_NODES] [CPU_CORES_PER_NODE] [MEMORY_PER_NODE]"
    echo ""
    echo "Available SCALE options:"
    echo "  500m    Run benchmark on 500 million rows scale"
    echo "  1b      Run benchmark on 1 billion rows scale"
    echo "  5b      Run benchmark on 5 billion rows scale"
    echo ""
    echo "Optional:"
    echo "  NUM_NODES "
    echo "  CPU_CORES_PER_NODE "
    echo "  MEMORY_PER_NODE "
    exit 1
fi

# Input
SCALE="$1"
DB_NAME="coffeeshop_${SCALE}"

# Optional overrides via arguments
NUM_NODES="${2:-8}"
CPU_CORES_PER_NODE="${3:-60}"
MEMORY_PER_NODE="${4:-240}"


if [[ "$SCALE" != "500m" && "$SCALE" != "1b" && "$SCALE" != "5b" ]]; then
    echo "❌ Invalid SCALE: $SCALE"
    echo "Valid options are: 500m, 1b, 5b"
    exit 1
fi




# Ensure results directory exists
mkdir -p results
mkdir -p logs

# Filenames
FILENAME_SUFFIX="${SCALE}_${NUM_NODES}n_${CPU_CORES_PER_NODE}c_${MEMORY_PER_NODE}g_${TIMESTAMP}"

RESULT_FILE_RUNTIMES="logs/result_runtime_${FILENAME_SUFFIX}.json"
RESULT_FILE_JSON="results/result_${FILENAME_SUFFIX}.json"

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
FASTEST=$(echo "$RUNTIMES" | jq '[.[] | (map(select(. != 0)) | min // 0.0)]')
SUM_FASTEST=$(echo "$FASTEST" | jq 'add | tonumber') # in seconds

# Calculate units used
TOTAL_CPU=$((NUM_NODES * CPU_CORES_PER_NODE))
TOTAL_RAM=$((NUM_NODES * MEMORY_PER_NODE))  # GiB
CPU_UNITS=$(awk "BEGIN { printf \"%.2f\", $TOTAL_CPU / 2 }")
RAM_UNITS=$(awk "BEGIN { printf \"%.2f\", $TOTAL_RAM / 8 }")
UNITS_USED=$(awk "BEGIN { printf \"%.2f\", ($TOTAL_CPU / 2 > $TOTAL_RAM / 8) ? $TOTAL_CPU / 2 : $TOTAL_RAM / 8 }")

# Calculate per-tier cost (in seconds → hours)
SUM_FASTEST_HOURS=$(awk "BEGIN { printf \"%.10f\", $SUM_FASTEST / 3600 }")

TIER_SCALE_COST_FMT=$(awk "BEGIN { printf \"%.5f\", $SUM_FASTEST_HOURS * $UNITS_USED * $TIER_SCALE_PRICE_PER_UNIT_PER_HR }")
TIER_ENTERPRISE_COST_FMT=$(awk "BEGIN { printf \"%.5f\", $SUM_FASTEST_HOURS * $UNITS_USED * $TIER_ENTERPRISE_PRICE_PER_UNIT_PER_HR }")

# Compute cost per query for each tier
TIER_SCALE_COSTS_PER_QUERY=$(echo "$FASTEST" | jq --arg units "$UNITS_USED" --arg price "$TIER_SCALE_PRICE_PER_UNIT_PER_HR" '
  map((. / 3600) * ($units | tonumber) * ($price | tonumber))
')

TIER_ENTERPRISE_COSTS_PER_QUERY=$(echo "$FASTEST" | jq --arg units "$UNITS_USED" --arg price "$TIER_ENTERPRISE_PRICE_PER_UNIT_PER_HR" '
  map((. / 3600) * ($units | tonumber) * ($price | tonumber))
')


# Write final pretty JSON
jq -n \
  --arg date "$DATE_TODAY" \
  --arg scale "$SCALE" \
  --arg csp "$CSP" \
  --arg region "$REGION" \
  --argjson num_nodes "$NUM_NODES" \
  --argjson cpu_cores_per_node "$CPU_CORES_PER_NODE" \
  --argjson memory_per_node "$MEMORY_PER_NODE" \
  --argjson result "$RUNTIMES" \
  --argjson fastest "$FASTEST" \
  --argjson sum_of_fastest "$SUM_FASTEST" \
  --argjson units_used "$UNITS_USED" \
  --arg tier_scale_price_per_unit_per_hr "$TIER_SCALE_PRICE_PER_UNIT_PER_HR" \
  --arg tier_enterprise_price_per_unit_per_hr "$TIER_ENTERPRISE_PRICE_PER_UNIT_PER_HR" \
  --arg tier_scale_cost "$TIER_SCALE_COST_FMT" \
  --arg tier_enterprise_cost "$TIER_ENTERPRISE_COST_FMT" \
  --argjson tier_scale_cost_per_query "$TIER_SCALE_COSTS_PER_QUERY" \
  --argjson tier_enterprise_cost_per_query "$TIER_ENTERPRISE_COSTS_PER_QUERY" \
  '{
    date: $date,
    scale: $scale,
    csp: $csp,
    region: $region,
    num_nodes: $num_nodes,
    cpu_cores_per_node: $cpu_cores_per_node,
    memory_per_node: $memory_per_node,
    result: $result,
    fastest: $fastest,
    sum_of_fastest: $sum_of_fastest,
    units_used: ($units_used | tonumber),
    tier_scale_price_per_unit_per_hr: ($tier_scale_price_per_unit_per_hr | tonumber),
    tier_enterprise_price_per_unit_per_hr: ($tier_enterprise_price_per_unit_per_hr | tonumber),
    tier_scale_cost: ($tier_scale_cost | tonumber),
    tier_enterprise_cost: ($tier_enterprise_cost | tonumber),
    tier_scale_cost_per_query: $tier_scale_cost_per_query,
    tier_enterprise_cost_per_query: $tier_enterprise_cost_per_query
  }' > "$RESULT_FILE_JSON"


echo "✅ JSON results written to $RESULT_FILE_JSON"