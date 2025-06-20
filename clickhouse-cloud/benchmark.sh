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

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"  # ensure the folder exists
QUERY_LOG_FILE="${LOG_DIR}/_query_log_${DB_NAME}_${TIMESTAMP}.txt"

echo "Running queries on database: $DB_NAME with $REPEATS repetitions"
#
# Run and log
./run_queries.sh "$DB_NAME" "$REPEATS" 2>&1 | tee "$QUERY_LOG_FILE"

# === Extract runtimes into JSON ===
RUNTIME_RESULTS=$(awk -v repeats="$REPEATS" '
BEGIN {
    FS = "\n";
    RS = "👉 Running query:\n";
    print "["
    first = 1
}
NR > 1 {
    n = 0;
    delete runtimes;

    # Extract only runtimes (first of each runtime–memory pair)
    for (i = 1; i <= NF; ++i) {
        if ($i ~ /^[0-9]+(\.[0-9]+)?$/) {
            runtimes[n++] = $i + 0;
            ++i;  # Skip memory line
        }
    }

    # Pad with 0.00 if too few
    while (n < repeats) {
        runtimes[n++] = 0.00;
    }

    # Output JSON array
    if (!first) {
        printf ",\n";
    }
    first = 0;

    printf "  [";
    for (j = 0; j < repeats; ++j) {
        printf "%s%.3f", (j > 0 ? ", " : ""), runtimes[j];
    }
    printf "]";
}
END {
    print "\n]"
}
' "$QUERY_LOG_FILE")

# === Write to output file ===
echo "$RUNTIME_RESULTS" > "$RESULT_FILE_RUNTIMES"
echo "Runtime results written to $RESULT_FILE_RUNTIMES"

