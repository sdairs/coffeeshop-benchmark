import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors # For color mapping
import numpy as np # For handling potential NaNs if a query is missing for a config
import re # For parsing compute configuration strings

RESULTS_DIRS = [
    "../clickhouse-cloud/results/",
    "../clickhouse-cloud-iceberg/results/",
    "../databricks/results/",
    "../snowflake/results/"
]

OUTPUT_DIR = "output_plots"

# Static costs for Databricks and Snowflake based on provided image
# Assumes compute_size for Databricks are 'S', 'XS'
# Assumes compute_size for Snowflake are 'S_Gen1', 'S_Gen2', 'XS_Gen1', 'XS_Gen2'
# (derived from filenames like 'S_Gen1.results.csv')
STATIC_COSTS = {
    ("databricks", "DBX_S"): {1: 0.023, 2: 0.007, 3: 0.011, 4: 0.008, 5: 0.005, 6: 0.018, 7: 0.005, 8: 0.010, 9: 0.006, 11: 0.008, 12: 0.006, 13: 0.004, 14: 0.006, 15: 0.023, 17: 0.013},
    ("databricks", "DBX_XS"): {1: 0.014, 2: 0.004, 3: 0.007, 4: 0.005, 5: 0.004, 6: 0.011, 7: 0.003, 8: 0.007, 9: 0.004, 11: 0.005, 12: 0.004, 13: 0.002, 14: 0.004, 15: 0.016, 17: 0.013},
    ("snowflake", "SF_S_Gen1"): {1: 0.011, 2: 0.009, 3: 0.008, 4: 0.016, 5: 0.006, 6: 0.018, 7: 0.018, 8: 0.025, 9: 0.009, 11: 0.015, 12: 0.009, 13: 0.006, 14: 0.015, 15: 0.033, 17: 0.012},
    ("snowflake", "SF_S_Gen2"): {1: 0.014, 2: 0.008, 3: 0.008, 4: 0.016, 5: 0.006, 6: 0.017, 7: 0.015, 8: 0.023, 9: 0.009, 11: 0.013, 12: 0.007, 13: 0.006, 14: 0.014, 15: 0.032, 17: 0.009},
    ("snowflake", "SF_XS_Gen1"): {1: 0.010, 2: 0.008, 3: 0.008, 4: 0.015, 5: 0.005, 6: 0.017, 7: 0.027, 8: 0.025, 9: 0.008, 11: 0.014, 12: 0.008, 13: 0.005, 14: 0.014, 15: 0.033, 17: 0.013},
    ("snowflake", "SF_XS_Gen2"): {1: 0.010, 2: 0.007, 3: 0.008, 4: 0.016, 5: 0.005, 6: 0.016, 7: 0.015, 8: 0.023, 9: 0.007, 11: 0.012, 12: 0.007, 13: 0.004, 14: 0.012, 15: 0.029, 17: 0.008}
}


def extract_query_number(query_str):
    """Extracts the number from a query string like 'Query X' or 'X'."""
    if isinstance(query_str, (int, float)):
        return int(query_str)
    try:
        return int(str(query_str).lower().replace('query', '').strip())
    except ValueError:
        return query_str # Return original if not convertible


def generate_visualizations(df_to_plot, metric_column, y_label, title_suffix, plot_filename_base, table_filename_base, dataset_size, output_dir):
    """Generates and saves a plot and a CSV table for the given metric."""
    
    pivot_table = df_to_plot.pivot_table(index='config_compute', 
                                         columns='query_num', 
                                         values=metric_column,
                                         aggfunc='mean')

    # Sort columns (queries) numerically if they are not already
    if not pivot_table.empty:
        pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    if pivot_table.empty:
        print(f"Pivot table for {metric_column} is empty for dataset {dataset_size}. Skipping plot.")
        return

    # Save the pivot table as CSV
    table_filename = os.path.join(output_dir, f"{table_filename_base}_{dataset_size}.csv")
    try:
        pivot_table.to_csv(table_filename)
        print(f"Saved table ({metric_column}) to {table_filename}")
    except Exception as e:
        print(f"Error saving table {table_filename}: {e}")

    # Generate Bar Chart
    num_queries = len(pivot_table.columns)
    num_configs = len(pivot_table.index)

    if num_queries == 0 or num_configs == 0:
        print(f"Not enough data to plot for dataset {dataset_size}, metric {metric_column}. Queries: {num_queries}, Configs: {num_configs}")
        return

    fig, ax = plt.subplots(figsize=(max(15, num_queries * 1.5), 10))
    
    bar_width = 0.8 / num_configs
    if bar_width * num_configs > 0.8:
         bar_width = 0.8 / num_configs
    if bar_width < 0.1:
        bar_width = 0.1

    indices = np.arange(num_queries)

    for i, config_name in enumerate(pivot_table.index):
        bar_values = pivot_table.loc[config_name].fillna(0).values # Fill NaN with 0 for plotting
        ax.bar(indices + i * bar_width - (num_configs * bar_width / 2) + bar_width / 2, 
               bar_values, 
               bar_width, 
               label=config_name)

    ax.set_xlabel("Query Number")
    ax.set_ylabel(y_label)
    ax.set_title(f"Benchmark {title_suffix} for Dataset: {dataset_size}")
    ax.set_xticks(indices)
    ax.set_xticklabels(pivot_table.columns.astype(str))
    ax.legend(title="DB Config & Compute Size", bbox_to_anchor=(1.05, 1), loc='upper left')

    if metric_column == 'cost':
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.4f'))
        table_data_str = pivot_table.fillna('N/A').round(4).astype(str)
    else: # seconds
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f s'))
        table_data_str = pivot_table.fillna('N/A').round(2).astype(str)

    plt.subplots_adjust(left=0.05, bottom=0.30, right=0.8, top=0.9) # Increased bottom margin

    cell_text_values = table_data_str.values.tolist()
    col_labels = [f"Q{col}" for col in table_data_str.columns]
    row_labels = table_data_str.index

    color_map_definition = [
        (0.0, mcolors.to_rgba('lightgreen')), (0.33, mcolors.to_rgba('yellow')),
        (0.66, mcolors.to_rgba('orange')), (1.0, mcolors.to_rgba('darkred'))
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", color_map_definition)
    nan_color = mcolors.to_rgba('lightgrey')
    default_cell_color = mcolors.to_rgba('white')

    num_table_rows = len(pivot_table.index)
    num_table_cols = len(pivot_table.columns)
    cell_colors_list = [[default_cell_color for _ in range(num_table_cols)] for _ in range(num_table_rows)]

    for col_idx, query_col_name in enumerate(pivot_table.columns):
        column_data = pivot_table[query_col_name]
        valid_data = column_data.dropna()
        if valid_data.empty:
            for row_idx in range(num_table_rows):
                cell_colors_list[row_idx][col_idx] = nan_color
            continue
        min_val = valid_data.min()
        max_val = valid_data.max()
        for row_idx in range(num_table_rows):
            value = pivot_table.iloc[row_idx, col_idx]
            if pd.isna(value):
                cell_colors_list[row_idx][col_idx] = nan_color
            else:
                if min_val == max_val:
                    cell_colors_list[row_idx][col_idx] = custom_cmap(0.0)
                else:
                    norm_value = (value - min_val) / (max_val - min_val)
                    cell_colors_list[row_idx][col_idx] = custom_cmap(norm_value)

    table_height_inches = max(2, len(row_labels) * 0.3 + 0.5)
    current_fig_width, current_fig_height = fig.get_size_inches()
    fig.set_size_inches(current_fig_width, current_fig_height + table_height_inches)

    the_table = plt.table(cellText=cell_text_values, 
                          rowLabels=row_labels, colLabels=col_labels,
                          cellColours=cell_colors_list, loc='bottom',
                          bbox=[0, -0.40 - 0.05 * len(row_labels), 1, 0.3 + 0.05 * len(row_labels)]) # Made y-position more negative
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1, 1.2)

    plot_filename = os.path.join(output_dir, f"{plot_filename_base}_{dataset_size}.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot ({metric_column}) to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)


def generate_total_metric_visualizations(df_to_plot, metric_column, y_label, title_suffix, plot_filename_base, table_filename_base, dataset_size, output_dir):
    """Generates and saves a plot and a CSV table for the total sum of a given metric, grouped by configuration."""
    print(f"--- Generating TOTAL visualization for metric: {metric_column}, dataset: {dataset_size} ---")

    # Calculate total metric per config_compute, ensure to sum only valid numbers
    # Using min_count=1 so that if all values for a group are NaN, the sum is NaN, not 0.
    grouped_sum = df_to_plot.groupby('config_compute')[metric_column].sum(min_count=1)
    grouped_sum.dropna(inplace=True) # Drop NaNs from the sum results first

    if grouped_sum.empty:
        print(f"Total metrics for '{metric_column}', dataset '{dataset_size}' are empty after dropping NaNs. Skipping plot.")
        return

    # For the plot (smallest bar at the top because barh plots first item in series at bottom of y-axis)
    series_for_plot = grouped_sum.sort_values(ascending=False)

    # For the table and CSV (smallest value at the top)
    series_for_table_csv = grouped_sum.sort_values(ascending=True)

    # Convert Series to DataFrame for table saving (using ascending order)
    total_metrics_df_for_csv = series_for_table_csv.reset_index()
    total_metrics_df_for_csv.columns = ['config_compute', f'total_{metric_column}']

    table_filename_csv = os.path.join(output_dir, f"{table_filename_base}_{dataset_size}.csv")
    try:
        total_metrics_df_for_csv.to_csv(table_filename_csv, index=False)
        print(f"Saved total metrics table ({metric_column}) to {table_filename_csv}")
    except Exception as e:
        print(f"Error saving total metrics table {table_filename_csv}: {e}")

    num_configs = len(series_for_plot.index) # Use series_for_plot for plot dimensions
    # Adjust figsize for horizontal bars: height might depend on num_configs, width can be more fixed or adapt to label length
    fig, ax = plt.subplots(figsize=(12, max(6, num_configs * 0.5))) # Width, Height

    ax.barh(series_for_plot.index, series_for_plot.values, color='teal')

    ax.set_ylabel("DB Config & Compute Size") # Swapped
    ax.set_xlabel(y_label) # Swapped
    ax.set_title(f"Benchmark {title_suffix} for Dataset: {dataset_size}")
    # plt.xticks is not needed here for rotation as y-labels in barh are typically horizontal
    # Ensure y-labels (config names) are fully visible if they are long
    plt.yticks(fontsize=8) # Adjust fontsize if needed
    
    # Define colors for table (consistent with other plot)
    color_map_definition = [
        (0.0, mcolors.to_rgba('lightgreen')), (0.33, mcolors.to_rgba('yellow')),
        (0.66, mcolors.to_rgba('orange')), (1.0, mcolors.to_rgba('darkred'))
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", color_map_definition)
    nan_color = mcolors.to_rgba('lightgrey')
    default_cell_color = mcolors.to_rgba('white')

    # Table display uses series_for_table_csv (ascending order)
    table_values_for_display = series_for_table_csv.copy()
    if metric_column == 'cost':
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('$%.4f')) # Changed to xaxis
        table_values_for_display = table_values_for_display.round(4)
    else: # seconds
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f s')) # Changed to xaxis
        table_values_for_display = table_values_for_display.round(2)
    
    cell_text_str_list = table_values_for_display.fillna('N/A').astype(str).tolist()
    cell_text_values = [[val] for val in cell_text_str_list] # Define cell_text_values for the table
    row_labels = series_for_table_csv.index.tolist() # Use ascending order for table row labels
    col_labels = [f"Total {metric_column.capitalize()}"]

    cell_colors_list = [[default_cell_color] for _ in range(len(row_labels))]
    valid_data_for_color = table_values_for_display.dropna()

    if not valid_data_for_color.empty:
        min_val_color = valid_data_for_color.min()
        max_val_color = valid_data_for_color.max()
        for i, value in enumerate(table_values_for_display):
            if pd.isna(value):
                cell_colors_list[i][0] = nan_color
            else:
                if min_val_color == max_val_color:
                    cell_colors_list[i][0] = custom_cmap(0.0)
                else:
                    norm_value = (value - min_val_color) / (max_val_color - min_val_color)
                    cell_colors_list[i][0] = custom_cmap(norm_value)
    else:
        for i in range(len(row_labels)):
            cell_colors_list[i][0] = nan_color

    # Adjust layout for table and potentially long y-axis labels
    plt.subplots_adjust(left=0.3, bottom=0.35) # Increased bottom margin for plot, to make space for table

    table_height_inches = max(1.5, len(row_labels) * 0.25 + 0.5)
    current_fig_width, current_fig_height = fig.get_size_inches()
    # fig.set_size_inches(current_fig_width, current_fig_height + table_height_inches) # This can distort existing plot, use bbox carefully

    # Position table using bbox relative to axes. The axes itself might need adjustment.
    # The bbox coordinates are (left, bottom, width, height) relative to the axes.
    # Negative bottom places it below the axes. Height of bbox needs to be dynamic.
    dynamic_bbox_bottom = -0.35 - (0.03 * len(row_labels)) # Made more negative to push table further down
    dynamic_bbox_height = 0.2 + (0.03 * len(row_labels))  # Heuristic for height

    the_table = plt.table(cellText=cell_text_values,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          cellColours=cell_colors_list,
                          loc='bottom',
                          bbox=[0, dynamic_bbox_bottom, 1, dynamic_bbox_height] # Full width of axes, below it
                         )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1, 1.2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to prevent table overlap if bbox_inches='tight' is not enough

    plot_filename = os.path.join(output_dir, f"{plot_filename_base}_{dataset_size}.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved total metrics plot ({metric_column}) to {plot_filename}")
    except Exception as e:
        print(f"Error saving total metrics plot {plot_filename}: {e}")
    plt.close(fig)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for relative_results_dir_path in RESULTS_DIRS:
        results_dir_abs_path = os.path.abspath(os.path.join(script_dir, relative_results_dir_path))
        db_config_with_parent = os.path.basename(os.path.dirname(results_dir_abs_path))
        csv_pattern = os.path.join(results_dir_abs_path, "*.results.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            print(f"Warning: No CSV files found in {results_dir_abs_path} matching pattern {csv_pattern}")
            continue

        for csv_file_path in csv_files:
            filename = os.path.basename(csv_file_path)
            compute_size = filename.replace(".results.csv", "")
            if not compute_size: compute_size = "default"
            
            try:
                df_temp = pd.read_csv(csv_file_path)
                df_temp['db_config'] = db_config_with_parent
                df_temp['compute_size'] = compute_size
                df_temp['cost'] = np.nan

                if db_config_with_parent in ["clickhouse-cloud", "clickhouse-cloud-iceberg"]:
                    match = re.match(r'(?:[a-zA-Z0-9-]+\.)?(\d+)x(\d+)c(\d+)m', compute_size) # Updated regex for optional provider prefix
                    if match:
                        nodes = int(match.group(1))
                        ram_per_node_gb = int(match.group(3))
                        compute_units_per_node = ram_per_node_gb / 8.0
                        total_compute_units = nodes * compute_units_per_node
                        cost_per_hour_for_config = total_compute_units * 0.298
                        cost_per_second_for_config = cost_per_hour_for_config / 3600.0
                        df_temp_seconds_numeric = pd.to_numeric(df_temp['seconds'], errors='coerce')
                        df_temp['cost'] = cost_per_second_for_config * df_temp_seconds_numeric
                    else:
                        print(f"Warning: Could not parse compute_size '{compute_size}' for {db_config_with_parent} in {csv_file_path} for ClickHouse cost calculation. Cost will be NaN.")
                elif db_config_with_parent in ["databricks", "snowflake"]:
                    config_key = (db_config_with_parent, compute_size)
                    if config_key in STATIC_COSTS:
                        # Apply static costs by mapping query_num to cost
                        # extract_query_number is applied later to df_master, so apply here temporarily for lookup
                        query_num_series = df_temp['query'].apply(extract_query_number)
                        df_temp['cost'] = query_num_series.map(STATIC_COSTS[config_key]).astype(float)
                        
                        # Check for any queries that didn't get a cost (e.g., if a query from CSV is not in STATIC_COSTS for this config)
                        if df_temp['cost'].isnull().any():
                            unmapped_queries = query_num_series[df_temp['cost'].isnull()].unique()
                            # Filter out NaN if query_num itself was NaN (e.g. from bad query string)
                            unmapped_queries = [q for q in unmapped_queries if pd.notna(q)] 
                            if unmapped_queries: # Only print warning if there are actual query numbers that were not mapped
                                print(f"Warning: For configuration {config_key}, static costs were not found or failed to map for query numbers: {unmapped_queries}. Their cost will be NaN.")
                    else:
                        print(f"Warning: Static cost data not found for configuration: {config_key}. Cost will be NaN for all queries in this file.")
                
                all_data.append(df_temp)
            except pd.errors.EmptyDataError:
                print(f"Warning: CSV file {csv_file_path} is empty.")
            except Exception as e:
                print(f"Error reading {csv_file_path}: {e}")

    if not all_data:
        print("No data loaded. Exiting.")
        return

    df_master = pd.concat(all_data, ignore_index=True)
    df_master['seconds'] = pd.to_numeric(df_master['seconds'], errors='coerce')
    df_master['cost'] = pd.to_numeric(df_master['cost'], errors='coerce')
    df_master['query_num'] = df_master['query'].apply(extract_query_number)
    df_master.dropna(subset=['seconds', 'query_num'], inplace=True)
    df_master['query_num'] = df_master['query_num'].astype(int)
    df_master['config_compute'] = df_master['db_config'] + "_" + df_master['compute_size']

    queries_to_exclude = [10, 15, 16] # Temporarily excluding Query 15 as well
    print(f"\nExcluding queries: {queries_to_exclude} from the results.")
    df_master = df_master[~df_master['query_num'].isin(queries_to_exclude)]

    if df_master.empty:
        print("No data remaining after excluding queries. Exiting.")
        return

    dataset_sizes = df_master['dataset'].unique()

    for dataset_size in dataset_sizes:
        print(f"Processing dataset: {dataset_size}")
        df_dataset_current = df_master[df_master['dataset'] == dataset_size].copy()

        if df_dataset_current.empty:
            print(f"No data for dataset size {dataset_size}. Skipping.")
            continue

        # Generate visualization for 'seconds'
        generate_visualizations(
            df_to_plot=df_dataset_current,
            metric_column='seconds',
            y_label='Seconds (Lower is Better)',
            title_suffix='Performance (Time)',
            plot_filename_base='plot_results',
            table_filename_base='table_results',
            dataset_size=dataset_size,
            output_dir=OUTPUT_DIR
        )

        # Generate visualization for 'cost'
        df_cost_dataset = df_dataset_current[df_dataset_current['cost'].notna()].copy()
        if not df_cost_dataset.empty:
            generate_visualizations(
                df_to_plot=df_cost_dataset,
                metric_column='cost',
                y_label='Cost ($) (Lower is Better)',
                title_suffix='Query Cost',
                plot_filename_base='plot_cost',
                table_filename_base='table_cost',
                dataset_size=dataset_size,
                output_dir=OUTPUT_DIR
            )
        else:
            print(f"No cost data available for dataset {dataset_size} to generate cost plot.")

        # Generate total time visualization
        generate_total_metric_visualizations(
            df_to_plot=df_dataset_current,
            metric_column='seconds',
            y_label='Total Seconds (Sum of All Queries)',
            title_suffix='Total Performance (Time)',
            plot_filename_base='plot_total_time',
            table_filename_base='table_total_time',
            dataset_size=dataset_size,
            output_dir=OUTPUT_DIR
        )

        # Generate total cost visualization
        # df_cost_dataset is already filtered for notna costs from the per-query section
        if not df_cost_dataset.empty:
            generate_total_metric_visualizations(
                df_to_plot=df_cost_dataset, # Use the same df_cost_dataset
                metric_column='cost',
                y_label='Total Cost ($) (Sum of All Queries)',
                title_suffix='Total Query Cost',
                plot_filename_base='plot_total_cost',
                table_filename_base='table_total_cost',
                dataset_size=dataset_size,
                output_dir=OUTPUT_DIR
            )
        else:
            print(f"No cost data available for dataset {dataset_size} to generate total cost plot.")


if __name__ == "__main__":
    main()