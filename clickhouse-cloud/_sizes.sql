SELECT
    database,
    table,
    formatReadableQuantity(sum(rows)) AS rows,
    formatReadableSize(sum(data_uncompressed_bytes)) AS data_size_uncompressed,
    formatReadableSize(sum(data_compressed_bytes)) AS data_size_compressed
FROM system.parts
WHERE active AND startsWith(database, 'coffeeshop_') AND startsWith(table, 'fact_')
GROUP BY database, table
ORDER BY sum(data_compressed_bytes);