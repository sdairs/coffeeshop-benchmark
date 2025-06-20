#!/bin/bash

set -euo pipefail

# Required environment variables
: "${CLICKHOUSE_HOST:?CLICKHOUSE_HOST is required}"
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER is required}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD is required}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <DB_NAME> <REPEATS>"
    exit 1
fi

DB_NAME="$1"
REPEATS="$2"

QUERY_FILE="queries.sql"
query_number=1

# Split queries using '-- #' headers
awk '
    /^-- [0-9]+/ { if (query) print query "\0"; query = ""; next }
    { query = query $0 "\n" }
    END { if (query) print query "\0" }
' "$QUERY_FILE" | while IFS= read -r -d '' query; do
    query="$(echo "$query" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [[ -z "$query" ]]; then
        continue
    fi

    echo "👉 Running query:"
    echo "Q${query_number}"
#     echo "$query"
    echo ""

    for i in $(seq 1 "$REPEATS"); do
        if ! echo "$query" | clickhouse-client \
            --host "${CLICKHOUSE_HOST}" \
            --user "${CLICKHOUSE_USER}" \
            --password "${CLICKHOUSE_PASSWORD}" \
            --secure \
            --database "$DB_NAME" \
            --time --memory-usage --format=Null --progress 0; then

            echo "❌ Query failed on attempt #$i, skipping to next query"
            remaining=$((REPEATS - i))
            for _ in $(seq 1 "$remaining"); do
                echo "0.00"
                echo "0"
                echo ""
            done
            break
        fi
        echo ""
    done
    ((query_number++))
done