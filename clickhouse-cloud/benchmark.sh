#!/bin/bash


# Required environment variables
: "${CLICKHOUSE_HOST:?CLICKHOUSE_HOST is required}"
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER is required}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD is required}"

# Usage check
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <DB_NAME> <REPEATS> <RESULT_FILE_RUNTIMES>"
    exit 1
fi

# Args
DB_NAME="$1"
REPEATS="$2"
RESULT_FILE_RUNTIMES="$3"

QUERY_LOG_FILE="_query_log_${DB_NAME}.txt"

echo "Running queries on database: $DB_NAME with $REPEATS repetitions"
#
# Run and log
./run_queries.sh "$DB_NAME" "$REPEATS" 2>&1 | tee "$QUERY_LOG_FILE"

# Extract only runtime lines (first line in each result pair), and group by REPEATS
RUNTIME_RESULTS=$(awk -v n="$REPEATS" '
    /^[0-9]+\.[0-9]+$/ {
        runtimes[total++] = $1;
    }
    END {
        print "[";
        for (i = 0; i < total; i += n) {
            printf "  [";
            for (j = 0; j < n; ++j) {
                printf "%s%.3f", (j > 0 ? ", " : ""), runtimes[i + j];
            }
            printf "]%s\n", (i + n < total ? "," : "");
        }
        print "]";
    }
' "$QUERY_LOG_FILE")

# Write output
echo "$RUNTIME_RESULTS" > "$RESULT_FILE_RUNTIMES"
echo "Runtime results written to $RESULT_FILE_RUNTIMES"