import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors # For color mapping
import numpy as np # For handling potential NaNs if a query is missing for a config

RESULTS_DIRS = [
    "../clickhouse-cloud/results/",
    "../clickhouse-cloud-iceberg/results/",
    "../databricks/results/",
    "../snowflake/results/"
]

OUTPUT_DIR = "output_plots"

def extract_query_number(query_str):
    """Extracts the number from a query string like 'Query X' or 'X'."""
    if isinstance(query_str, (int, float)):
        return int(query_str)
    try:
        return int(str(query_str).lower().replace('query', '').strip())
    except ValueError:
        return query_str # Return original if not convertible

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for relative_results_dir_path in RESULTS_DIRS:
        # Construct absolute path for results_dir to reliably get parent
        results_dir_abs_path = os.path.abspath(os.path.join(script_dir, relative_results_dir_path))
        db_config_with_parent = os.path.basename(os.path.dirname(results_dir_abs_path))

        # Find all CSV files matching the pattern
        # Ensure the pattern correctly captures files like DBX_M.results.csv
        csv_pattern = os.path.join(results_dir_abs_path, "*.results.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            print(f"Warning: No CSV files found in {results_dir_abs_path} matching pattern {csv_pattern}")
            continue

        for csv_file_path in csv_files:
            filename = os.path.basename(csv_file_path)
            # Extract compute_size: anything before '.results.csv'
            compute_size = filename.replace(".results.csv", "")
            if not compute_size: # Handle cases like just 'results.csv'
                compute_size = "default" # Or some other placeholder
            
            try:
                df_temp = pd.read_csv(csv_file_path)
                df_temp['db_config'] = db_config_with_parent
                df_temp['compute_size'] = compute_size
                all_data.append(df_temp)
            except pd.errors.EmptyDataError:
                print(f"Warning: CSV file {csv_file_path} is empty.")
            except Exception as e:
                print(f"Error reading {csv_file_path}: {e}")

    if not all_data:
        print("No data loaded. Exiting.")
        return

    df_master = pd.concat(all_data, ignore_index=True)

    # Ensure 'seconds' is numeric
    df_master['seconds'] = pd.to_numeric(df_master['seconds'], errors='coerce')
    # Extract query number for sorting and consistent naming
    df_master['query_num'] = df_master['query'].apply(extract_query_number)
    df_master.dropna(subset=['seconds', 'query_num'], inplace=True) # Drop rows where conversion failed or data is missing
    df_master['query_num'] = df_master['query_num'].astype(int)

    # Create a combined identifier for plotting
    df_master['config_compute'] = df_master['db_config'] + "_" + df_master['compute_size']

    # Exclude specific queries (10 and 16)
    queries_to_exclude = [10, 16]
    print(f"\nExcluding queries: {queries_to_exclude} from the results.")
    df_master = df_master[~df_master['query_num'].isin(queries_to_exclude)]

    if df_master.empty:
        print("No data remaining after excluding queries. Exiting.")
        return

    dataset_sizes = df_master['dataset'].unique()

    for dataset_size in dataset_sizes:
        print(f"Processing dataset: {dataset_size}")
        df_dataset = df_master[df_master['dataset'] == dataset_size].copy()

        if df_dataset.empty:
            print(f"No data for dataset size {dataset_size}. Skipping.")
            continue

        # Pivot table for plotting and table output
        # Using mean in case there are duplicate entries for the same query/config/dataset (should not happen with this structure)
        pivot_table = df_dataset.pivot_table(index='config_compute', 
                                             columns='query_num', 
                                             values='seconds',
                                             aggfunc='mean')

        # Sort columns (queries) numerically if they are not already
        pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

        if pivot_table.empty:
            print(f"Pivot table is empty for dataset {dataset_size}. Skipping plot.")
            continue

        # Save the pivot table as CSV
        table_filename = os.path.join(OUTPUT_DIR, f"table_results_{dataset_size}.csv")
        pivot_table.to_csv(table_filename)
        print(f"Saved table to {table_filename}")

        # Generate Bar Chart
        num_queries = len(pivot_table.columns)
        num_configs = len(pivot_table.index)

        if num_queries == 0 or num_configs == 0:
            print(f"Not enough data to plot for dataset {dataset_size}. Queries: {num_queries}, Configs: {num_configs}")
            continue

        fig, ax = plt.subplots(figsize=(max(15, num_queries * 1.5), 10))
        
        bar_width = 0.8 / num_configs # Adjust bar width based on number of configs
        if bar_width * num_configs > 0.8: # Cap total width
             bar_width = 0.8 / num_configs
        if bar_width < 0.1:
            bar_width = 0.1 # Minimum bar width

        indices = np.arange(num_queries)

        for i, config_name in enumerate(pivot_table.index):
            # Get values for this config, fill NaNs with 0 for plotting if a query is missing
            # or decide to skip/highlight missing data.
            # For now, plotting 0, but this might skew perception if data is truly missing.
            # A better approach might be to not plot missing bars or use a different color.
            bar_values = pivot_table.loc[config_name].fillna(0).values 
            ax.bar(indices + i * bar_width - (num_configs * bar_width / 2) + bar_width / 2, 
                   bar_values, 
                   bar_width, 
                   label=config_name)

        ax.set_xlabel("Query Number")
        ax.set_ylabel("Seconds (Lower is Better)")
        ax.set_title(f"Benchmark Results for Dataset: {dataset_size}")
        ax.set_xticks(indices)
        ax.set_xticklabels(pivot_table.columns.astype(str)) # Use query numbers as x-tick labels
        ax.legend(title="DB Config & Compute Size", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f s'))

        # Add table below the plot
        # Format pivot_table values for display
        table_data_str = pivot_table.fillna('N/A').round(2).astype(str)
        
        # Add a title for the table part if needed, or incorporate into main title
        # The table will be placed at the bottom of the figure.
        # We need to make space for it. `plt.tight_layout()` might not be enough alone.
        # One way is to adjust subplot parameters.
        plt.subplots_adjust(left=0.05, bottom=0.25, right=0.8, top=0.9) # Adjust right for legend, bottom for table

        # Add the table - colWidths can be adjusted
        # Convert DataFrame to list of lists for cellText, which plt.table expects
        cell_text_values = table_data_str.values.tolist()
        
        # Define column labels from pivot_table columns (query numbers)
        col_labels = [f"Q{col}" for col in table_data_str.columns]
        row_labels = table_data_str.index

        # --- Add color grading to table cells ---
        # Define colors for the gradient: Green (best) -> Yellow -> Orange -> Red (worst)
        color_map_definition = [
            (0.0, mcolors.to_rgba('lightgreen')),  # Best value
            (0.33, mcolors.to_rgba('yellow')),
            (0.66, mcolors.to_rgba('orange')),
            (1.0, mcolors.to_rgba('darkred'))    # Worst value
        ]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", color_map_definition)
        nan_color = mcolors.to_rgba('lightgrey')
        default_cell_color = mcolors.to_rgba('white')

        num_table_rows = len(pivot_table.index)
        num_table_cols = len(pivot_table.columns)
        cell_colors_list = [[default_cell_color for _ in range(num_table_cols)] for _ in range(num_table_rows)]

        for col_idx, query_col_name in enumerate(pivot_table.columns):
            column_data = pivot_table[query_col_name] # This is a pandas Series of numeric data
            valid_data = column_data.dropna()

            if valid_data.empty:
                for row_idx in range(num_table_rows):
                    cell_colors_list[row_idx][col_idx] = nan_color
                continue

            min_val = valid_data.min()
            max_val = valid_data.max()

            for row_idx in range(num_table_rows): # Iterate using numeric row index
                value = pivot_table.iloc[row_idx, col_idx]

                if pd.isna(value):
                    cell_colors_list[row_idx][col_idx] = nan_color
                else:
                    if min_val == max_val: # All valid values are the same or only one valid value
                        cell_colors_list[row_idx][col_idx] = custom_cmap(0.0) # Color as best (light green)
                    else:
                        norm_value = (value - min_val) / (max_val - min_val)
                        cell_colors_list[row_idx][col_idx] = custom_cmap(norm_value)
        # --- End of color grading ---

        # Increase figure height to accommodate table
        table_height_inches = max(2, len(row_labels) * 0.3 + 0.5) # Minimum 2 inches, or scaled by rows
        current_fig_width, current_fig_height = fig.get_size_inches()
        fig.set_size_inches(current_fig_width, current_fig_height + table_height_inches)

        the_table = plt.table(cellText=cell_text_values, 
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              cellColours=cell_colors_list, # Add the calculated cell colors
                              loc='bottom', # Place table at the bottom of the axes area
                              bbox=[0, -0.35 - 0.05 * len(row_labels), 1, 0.3 + 0.05 * len(row_labels)] # x, y, width, height relative to axes
                              )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1, 1.2) # Scale table to fit, adjust as needed

        # Re-adjust layout slightly if needed after adding table and resizing figure
        # plt.tight_layout(rect=[0, 0.1, 0.85, 0.95]) # rect=[left, bottom, right, top]
        # The bbox for legend and table might need careful manual adjustment or using GridSpec

        plot_filename = os.path.join(OUTPUT_DIR, f"plot_results_{dataset_size}.png")
        try:
            # bbox_inches='tight' is important here to include the table and legend
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)

if __name__ == "__main__":
    main()