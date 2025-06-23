import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.gridspec as gridspec
import numpy as np
from enum import Enum


# Assuming the script is in a 'charts' directory and data is in sibling directories
BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), '..')

DATABASE_DIRS = {
    "clickhouse-cloud": os.path.join(BASE_DATA_PATH, "clickhouse-cloud"),
    "databricks": os.path.join(BASE_DATA_PATH, "databricks"),
    "snowflake": os.path.join(BASE_DATA_PATH, "snowflake"),
}

DATABASE_COLORS = {
    "clickhouse-cloud": ["#faff69", "#E2D25D", "#BBFF69", "#FFDC69", "#7CFF69"],
    "databricks": ["#FF3621", "#FF21DF", "#FF8521"],
    "snowflake": ["#00A1D9", "#00D9C9", "#005ED9", "#001AD9"],
    # Add more specific colors if needed for different tiers/hardware
}

DATABASE_LABELS = {
    "clickhouse-cloud": "ClickHouse Cloud",
    "databricks": "Databricks",
    "snowflake": "Snowflake",
}

# Standardized query names (Q1 to Q17)
QUERY_NAMES = [f"Query_{i:02d}" for i in range(1, 18)] # Query_01 to Query_17

# ClickHouse data filtering configuration
# We benchmarked a lot of instance types out of interest, but we don't want to produce too many bars or the charts are unreadable
CLICKHOUSE_EXCLUDE_PATTERNS = [
    "60c",  # We dont need to include the 60 vCPU results as these instances were far larger than the sf/dbx
]

# Dataset-size-specific filtering for ClickHouse hardware configurations
# The original benchmark scaled the hardware size with the workload and did not bench small instances on larger datasets / large instaces on smaller datasets
CLICKHOUSE_DATASET_FILTERS = {
    "500m": [
        "16n",  # Exclude 16-node configs for 500m
        "8n",   # Exclude 8-node configs for 500m
    ],
    "1b": [
        "16n",  # Exclude 16-node configs for 1b
        "8n",   # Exclude 8-node configs for 1b
    ],
    "5b": [
        "2n",   # Exclude 2-node configs for 5b
        "4n",   # Exclude 4-node configs for 5b
    ]
}

class ChartType(Enum):
    """Enum to define different chart types and their behaviors"""
    QUERY_PERFORMANCE = "query_performance"
    QUERY_COST = "query_cost"
    TOTAL_PERFORMANCE = "total_performance"
    TOTAL_COST = "total_cost"


class ChartConfig:
    """Configuration class to define chart behavior based on chart type"""
    
    @staticmethod
    def get_config(chart_type: ChartType):
        configs = {
            ChartType.QUERY_PERFORMANCE: {
                'is_total_chart': False,
                'is_cost_chart': False,
                'sort_ascending': True,
                'show_secondary_values': False,
                'currency_format': False
            },
            ChartType.QUERY_COST: {
                'is_total_chart': False,
                'is_cost_chart': True,
                'sort_ascending': True,
                'show_secondary_values': False,
                'currency_format': True
            },
            ChartType.TOTAL_PERFORMANCE: {
                'is_total_chart': True,
                'is_cost_chart': False,
                'sort_ascending': True,
                'show_secondary_values': True,
                'currency_format': False
            },
            ChartType.TOTAL_COST: {
                'is_total_chart': True,
                'is_cost_chart': True,
                'sort_ascending': True,
                'show_secondary_values': True,
                'currency_format': True
            }
        }
        return configs.get(chart_type, configs[ChartType.QUERY_PERFORMANCE])


# --- Data Loading Functions ---

def load_snowflake_databricks_data(db_name, data_type):
    """Loads performance or cost data for Snowflake or Databricks."""
    # data_type can be 'results' (performance) or 'costs'
    file_path = os.path.join(DATABASE_DIRS[db_name], f"{data_type}.json")
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return []
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    # Example structure from user: {"SF L Gen 1 @ $24": {"500m": {"Query_01": 0.5, ...}}}
    # We need to flatten this into a list of records
    for dataset_size_key, hardware_data in data.items(): # Outer loop is dataset size (e.g., "500m")
        for hardware_config_key, query_values in hardware_data.items(): # Inner loop is hardware config (e.g., "SF_S_Gen1")
            if not isinstance(query_values, list):
                print(f"Warning: Expected a list of query results for {db_name} - {dataset_size_key} - {hardware_config_key}, but got {type(query_values)}. Skipping this entry.")
                continue

            for i, value in enumerate(query_values):
                if value is None:
                    continue
                
                if i < len(QUERY_NAMES):
                    query_name = QUERY_NAMES[i]
                    record = {
                        'database': db_name,
                        'hardware_config': hardware_config_key, # Corrected assignment
                        'dataset_size': dataset_size_key,    # Corrected assignment
                        'query_name': query_name,
                        'metric_type': 'performance' if data_type == 'results' else 'cost',
                        'value': float(value)
                    }
                    processed_data.append(record)
                else:
                    print(f"Warning: Found more results ({len(query_values)}) than defined query names ({len(QUERY_NAMES)}) for {db_name} - {dataset_size_key} - {hardware_config_key}. Index {i} (value: {value}) is out of bounds. Skipping extra data.")
    return processed_data

def load_clickhouse_data(data_type):
    """Loads performance or cost data for ClickHouse Cloud."""
    # data_type can be 'performance' or 'cost'
    results_dir = os.path.join(DATABASE_DIRS["clickhouse-cloud"], "results")
    print(f"Loading ClickHouse data from: {results_dir}")
    if not os.path.exists(results_dir):
        print(f"Warning: ClickHouse results directory not found - {results_dir}")
        return []

    processed_data = []
    for filename in os.listdir(results_dir):
        if filename.startswith("result_") and filename.endswith(".json"):
            file_path = os.path.join(results_dir, filename)
            # Extract scale and hardware from filename
            # New pattern: result_v25_4_1_1b_2n_30c_120g_20250620_151900.json
            # Old pattern: result_1b_2n_60c_240g_....json
            parts = filename.replace("result_", "").replace(".json", "").split('_')
            
            # Check if this is the new pattern (starts with version like v25_4_1)
            if parts[0].startswith('v'):
                # New pattern: skip version parts (v25, 4, 1) and get dataset size
                dataset_size = parts[3]  # 1b is at index 3 after v25, 4, 1
                # Hardware config starts after dataset size, ends before timestamp (last 2 parts)
                hardware_config_ch = "_".join(parts[4:-2])
            else:
                # Old pattern: dataset size is first, hardware config follows
                dataset_size = parts[0]
                hardware_config_ch = "_".join(parts[1:-2])
            # Map to a more generic hardware label if needed, for now use raw
            # For ClickHouse, hardware might be like '2n_60c_240g'

            # Apply filtering configuration
            if any(pattern in hardware_config_ch for pattern in CLICKHOUSE_EXCLUDE_PATTERNS):
                print(f"Skipping ClickHouse file: {filename} due to excluded pattern in hardware config: {hardware_config_ch}")
                continue

            # Apply dataset-size-specific filtering
            if dataset_size in CLICKHOUSE_DATASET_FILTERS:
                if any(pattern in hardware_config_ch for pattern in CLICKHOUSE_DATASET_FILTERS[dataset_size]):
                    print(f"Skipping ClickHouse file: {filename} due to dataset-size-specific exclusion in hardware config: {hardware_config_ch}")
                    continue

            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if data_type == 'performance':
                # User: "clickhouse cloud results to use are under the `fastest` key"
                if 'fastest' in data and isinstance(data['fastest'], list):
                    for i, value in enumerate(data['fastest']):
                        if value is None: continue
                        if i < len(QUERY_NAMES):
                            query_name = QUERY_NAMES[i]
                            record = {
                                'database': 'clickhouse-cloud',
                                'hardware_config': f"CH {hardware_config_ch}", # Prefix to distinguish
                                'dataset_size': dataset_size,
                                'query_name': query_name,
                                'metric_type': 'performance',
                                'value': float(value)
                            }
                            processed_data.append(record)
                        else:
                            print(f"Warning: CH 'fastest' - Found more results ({len(data['fastest'])}) than defined query names ({len(QUERY_NAMES)}) for {filename}. Index {i} out of bounds. Skipping extra data.")
                elif 'fastest' in data: # It exists but is not a list
                     print(f"Warning: CH 'fastest' data in {filename} is not a list as expected. Type: {type(data['fastest'])}. Skipping 'fastest' data.")

            elif data_type == 'cost':
                # User: "tier_scale_cost_per_query` and `tier_enterprise_cost_per_query`"
                cost_keys = {
                    'tier_scale_cost_per_query': 'cost_scale',
                    'tier_enterprise_cost_per_query': 'cost_enterprise'
                }
                for ch_cost_key, metric_label in cost_keys.items():
                    if ch_cost_key in data and isinstance(data[ch_cost_key], list):
                        cost_values = data[ch_cost_key]
                        for i, value in enumerate(cost_values):
                            if value is None: continue
                            if i < len(QUERY_NAMES):
                                query_name = QUERY_NAMES[i]
                                record = {
                                    'database': 'clickhouse-cloud',
                                    'hardware_config': f"CH {hardware_config_ch} ({metric_label.split('_')[1]})",
                                    'dataset_size': dataset_size,
                                    'query_name': query_name,
                                    'metric_type': metric_label,
                                    'value': float(value)
                                }
                                processed_data.append(record)
                            else:
                                print(f"Warning: CH '{ch_cost_key}' - Found more results ({len(cost_values)}) than defined query names ({len(QUERY_NAMES)}) for {filename}. Index {i} out of bounds. Skipping extra data.")
                    elif ch_cost_key in data: # It exists but is not a list
                        print(f"Warning: CH '{ch_cost_key}' data in {filename} is not a list as expected. Type: {type(data[ch_cost_key])}. Skipping this cost data.")

    return processed_data

def generate_chart_and_table(df_filtered, title, output_filename_base, queries_to_plot, chart_type=ChartType.QUERY_PERFORMANCE, horizontal_bars=False):
    """Generates and saves a bar chart with a data table below it."""
    print(f"\n--- Generating chart: {title} ---")
    print(f"Chart type: {chart_type.value}")
    print(f"Input df_filtered ({len(df_filtered)} records) head for '{title}':")
    if not df_filtered.empty:
        print(df_filtered.head().to_string())
        print(f"Unique 'dataset_size' in input df_filtered: {df_filtered['dataset_size'].unique()}")
        print(f"Unique 'hardware_config' in input df_filtered: {df_filtered['hardware_config'].unique()}")
        print(f"Unique 'database' in input df_filtered: {df_filtered['database'].unique()}")
        # Show counts per database for this specific chart's input
        print("Counts per database in input df_filtered:")
        print(df_filtered.groupby('database')['hardware_config'].count())
    else:
        print("Input df_filtered is EMPTY.")
    print("--- End of input data for chart --- \n")

    # Get chart configuration
    config = ChartConfig.get_config(chart_type)

    # Original print for the function start
    # print(f"\nGenerating chart: {title}") # This can be removed or kept as is
    if df_filtered.empty:
        print(f"Skipping chart generation for '{title}' as there is no data.")
        return

    # Ensure 'value' is numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
    df_filtered.dropna(subset=['value'], inplace=True)

    # Pivot table for plotting and for the table display
    # Index: hardware_config, Columns: query_name, Values: value
    pivot_df = df_filtered.pivot_table(index='hardware_config', columns='query_name', values='value')
    pivot_df = pivot_df[queries_to_plot] # Ensure correct query order and selection

    # For 'Total' charts, sort by value (best to worst, so ascending for data but descending for display)
    if config['is_total_chart'] and not pivot_df.empty and len(pivot_df.columns) == 1:
        # Sort by the single column of values, descending so best (lowest) values appear at top of horizontal bars
        pivot_df = pivot_df.sort_values(by=pivot_df.columns[0], ascending=False)
    
    # Sort hardware_config for consistent plotting order if desired (e.g., by a predefined list or name)
    # For now, default pandas sort is used for non-total charts or if sorting by value isn't applicable.

    num_configs = len(pivot_df.index)
    num_queries = len(pivot_df.columns)

    if num_configs == 0 or num_queries == 0:
        print(f"Skipping chart '{title}' due to no data after pivoting.")
        return

    fig = plt.figure(figsize=(max(15, num_queries * 1.5), 12)) # Increased height for better spacing
    fig.patch.set_facecolor('#1C1C1A') # Set figure background

    # Use different layouts for total vs non-total charts
    if config['is_total_chart']:
        # Single subplot for total charts (no table)
        ax_chart = fig.add_subplot(1, 1, 1)
    else:
        # Two subplots for non-total charts (chart + table)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.1) # Increased spacing between chart and table
        ax_chart = fig.add_subplot(gs[0]) # Axis for the bar chart
    
    ax_chart.set_facecolor('#1C1C1A') # Set chart area background

    # Set spine colors and visibility - only show bottom spine
    ax_chart.spines['bottom'].set_color('white')
    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)
    ax_chart.spines['left'].set_visible(False)

    # Remove vertical axis ticks for total charts only
    if config['is_total_chart']:
        ax_chart.set_yticks([])
    else:
        # For per-query charts, keep y-axis ticks and add grid lines
        ax_chart.tick_params(axis='y', colors='white')
        ax_chart.grid(True, axis='y', linestyle='--', color='white', alpha=0.3)

    # Bar chart
    bar_width = 0.8 / num_configs # Adjust bar width based on number of configs
    indices = np.arange(num_queries) # Use numpy arange for consistent bar positioning

    # --- Color variation logic --- 
    config_colors = {}
    db_config_counts = {} # Tracks count of configs for each base DB type in this chart

    # Sort index to make color assignment more consistent if order changes slightly
    # This ensures that if 'SF_S_Gen1' and 'SF_S_Gen2' are present, 'SF_S_Gen1' (if it comes first alphabetically)
    # is more likely to get the base color.
    # A more robust way would be to pre-define an order for hardware configs if strictness is needed.
    sorted_pivot_index = sorted(pivot_df.index.tolist())

    for config_name_in_pivot in sorted_pivot_index:
        current_db_base = 'unknown' # Default to 'unknown'

        # Try to match config_name_in_pivot with keys from DATABASE_COLORS
        # Config names are like "SF_S_Gen1", "DBX_L", "CH 2n_60c_240g", "CH 2n_60c_240g (scale)"
        # DATABASE_COLORS keys are 'snowflake', 'databricks', 'clickhouse-cloud'
        
        config_lower = config_name_in_pivot.lower()
        if 'sf' in config_lower or 'snowflake' in config_lower:
            current_db_base = 'snowflake'
        elif 'dbx' in config_lower or 'databricks' in config_lower:
            current_db_base = 'databricks'
        elif 'ch' in config_lower or 'clickhouse' in config_lower:
            current_db_base = 'clickhouse-cloud'
        # Add more specific checks if the above are too broad or conflict

        count = db_config_counts.get(current_db_base, 0)
        color_list_for_db = DATABASE_COLORS.get(current_db_base)

        if isinstance(color_list_for_db, list) and color_list_for_db: # Check if it's a non-empty list
            # Cycle through the predefined colors for this database
            color_index = count % len(color_list_for_db)
            config_colors[config_name_in_pivot] = color_list_for_db[color_index]
            db_config_counts[current_db_base] = count + 1
        else:
            # Fallback color if no predefined colors are available
            config_colors[config_name_in_pivot] = '#CCCCCC'

    # Plot bars for each hardware config
    for i, config_name in enumerate(pivot_df.index):
        values = pivot_df.loc[config_name].values
        positions = indices + i * bar_width
        color = config_colors.get(config_name, '#CCCCCC')
        
        if horizontal_bars:
            bars = ax_chart.barh(positions, values, bar_width, label=config_name, color=color)
        else:
            bars = ax_chart.bar(positions, values, bar_width, label=config_name, color=color)

        # Add value labels on bars for total charts
        if config['is_total_chart']:
            for j, (bar, value) in enumerate(zip(bars, values)):
                if pd.notna(value):
                    if horizontal_bars:
                        # For horizontal bars, text goes to the right of the bar with fixed padding
                        x_pos = bar.get_width()  # Fixed padding for all bars
                        y_pos = bar.get_y() + bar.get_height() / 2
                        
                        # Add hardware config label to the left of the bar
                        label_x_pos = -max(values) * 0.02  # Position to the left of the chart area
                        ax_chart.text(label_x_pos, y_pos, config_name, 
                                    ha='right', va='center', fontsize=10, color='white', weight='bold')
                        
                        # Show secondary values if available and configured
                        if config['show_secondary_values'] and 'total_cost_value' in df_filtered.columns and chart_type == ChartType.TOTAL_PERFORMANCE:
                            # Show cost value for performance charts
                            cost_row = df_filtered[df_filtered['hardware_config'] == config_name]
                            if not cost_row.empty and 'total_cost_value' in cost_row.columns:
                                cost_value = cost_row['total_cost_value'].iloc[0]
                                if pd.notna(cost_value):
                                    ax_chart.text(x_pos, y_pos, f'{value:.3f}s\n(${cost_value:.3f})', 
                                                ha='left', va='center', fontsize=9, color='white', weight='bold')
                                else:
                                    ax_chart.text(x_pos, y_pos, f'{value:.3f}s', 
                                                ha='left', va='center', fontsize=9, color='white', weight='bold')
                            else:
                                ax_chart.text(x_pos, y_pos, f'{value:.3f}s', 
                                            ha='left', va='center', fontsize=9, color='white', weight='bold')
                        elif config['show_secondary_values'] and 'total_perf_value' in df_filtered.columns and chart_type == ChartType.TOTAL_COST:
                            # Show performance value for cost charts
                            perf_row = df_filtered[df_filtered['hardware_config'] == config_name]
                            if not perf_row.empty and 'total_perf_value' in perf_row.columns:
                                perf_value = perf_row['total_perf_value'].iloc[0]
                                if pd.notna(perf_value):
                                    ax_chart.text(x_pos, y_pos, f'${value:.3f}\n({perf_value:.3f}s)', 
                                                ha='left', va='center', fontsize=9, color='white', weight='bold')
                                else:
                                    ax_chart.text(x_pos, y_pos, f'${value:.3f}', 
                                                ha='left', va='center', fontsize=9, color='white', weight='bold')
                            else:
                                ax_chart.text(x_pos, y_pos, f'${value:.3f}', 
                                            ha='left', va='center', fontsize=9, color='white', weight='bold')
                        else:
                            # Standard value display
                            if config['currency_format']:
                                ax_chart.text(x_pos, y_pos, f'${value:.3f}', 
                                            ha='left', va='center', fontsize=9, color='white', weight='bold')
                            else:
                                ax_chart.text(x_pos, y_pos, f'{value:.3f}s', 
                                            ha='left', va='center', fontsize=9, color='white', weight='bold')
                    else:
                        # For vertical bars, text goes above the bar
                        x_pos = bar.get_x() + bar.get_width() / 2
                        y_pos = bar.get_height() + max(values) * 0.01
                        
                        if config['currency_format']:
                            ax_chart.text(x_pos, y_pos, f'${value:.3f}', 
                                        ha='center', va='bottom', fontsize=9, color='white', weight='bold')
                        else:
                            ax_chart.text(x_pos, y_pos, f'{value:.3f}s', 
                                        ha='center', va='bottom', fontsize=9, color='white', weight='bold')

    # Chart formatting
    if horizontal_bars:
        # Remove y-tick labels since we removed y-ticks entirely
        
        # Adjust x-axis limits for total charts to make room for labels on the left
        if config['is_total_chart']:
            current_xlim = ax_chart.get_xlim()
            max_bar_value = pivot_df.values.max() if not pivot_df.empty else 1
            # Extend left side for labels and right side for value labels
            # ax_chart.set_xlim(-max_bar_value * 0.15, max_bar_value * 1.15)
            
            # Hide negative x-axis ticks - only show ticks from 0 onwards
            # current_ticks = ax_chart.get_xticks()
            # positive_ticks = [tick for tick in current_ticks if tick >= 0]
            # ax_chart.set_xticks(positive_ticks)
            
            # Shorten bottom spine to start at x=0 instead of extending to the left
            ax_chart.spines['bottom'].set_bounds(0, max_bar_value * 1.15)
    else:
        ax_chart.set_xticks(indices + bar_width * (num_configs - 1) / 2)
        
        # Transform query labels for per-query charts (not total charts)
        if not config['is_total_chart']:
            # Transform Query_02 format to Q2 format for per-query charts
            simplified_labels = []
            for label in queries_to_plot:
                if label.startswith('Query_'):
                    # Extract the number part and convert Query_02 to Q2
                    query_num = label.replace('Query_', '')
                    simplified_labels.append(f'Q{int(query_num)}')
                else:
                    simplified_labels.append(label)
            ax_chart.set_xticklabels(simplified_labels, color='white')  # No rotation for per-query charts
        else:
            # Keep original labels and rotation for total charts
            ax_chart.set_xticklabels(queries_to_plot, rotation=45, ha='right', color='white')
            
    ax_chart.set_title(title, color='white', fontsize=16, weight='bold')
    ax_chart.tick_params(colors='white')

    # Reduce gap between bars and bottom axis
    ax_chart.margins(y=0.01)

    # Data table - only for non-total charts
    if not config['is_total_chart']:
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis('off')

        # Create table data
        def format_number(value):
            """Format numbers based on chart type"""
            if pd.isna(value):
                return 'N/A'
            if config['currency_format']:
                return f'${value:.3f}'
            else:
                return f'{value:.3f}'

        # Prepare table data
        table_data = []
        headers = ['Color', 'Hardware Config'] + list(pivot_df.columns)
        
        for config_name in pivot_df.index:
            row = ['', config_name]  # Empty string for color swatch, will be filled with color
            for query in pivot_df.columns:
                value = pivot_df.loc[config_name, query]
                row.append(format_number(value))
            table_data.append(row)

        # Create table
        if table_data:
            table = ax_table.table(cellText=table_data, colLabels=headers, 
                                  cellLoc='center', loc='center',
                                  colColours=['#333333'] * len(headers))
            table.auto_set_font_size(False)
            table.set_fontsize(9)  # Slightly smaller font
            table.scale(1, 1.5)    # Reduced vertical scaling
            
            # Adjust column widths - make Hardware Config column wider
            num_data_cols = len(pivot_df.columns)
            color_col_width = 0.08  # Color swatch column
            config_col_width = 0.11  # Hardware Config column (slightly wider than default)
            remaining_width = 1.0 - color_col_width - config_col_width
            data_col_width = remaining_width / num_data_cols if num_data_cols > 0 else 0.1
            
            # Set column widths
            for j in range(len(headers)):
                if j == 0:  # Color column
                    width = color_col_width
                elif j == 1:  # Hardware Config column
                    width = config_col_width
                else:  # Data columns
                    width = data_col_width
                
                # Apply width to all cells in this column
                for i in range(len(table_data) + 1):  # +1 for header row
                    if (i, j) in table.get_celld():
                        table.get_celld()[(i, j)].set_width(width)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#555555')
                else:
                    cell.set_text_props(color='white')
                    if j == 0:  # Color column - use the hardware config's color
                        config_name = pivot_df.index[i-1]  # i-1 because header is row 0
                        color = config_colors.get(config_name, '#CCCCCC')
                        cell.set_facecolor(color)
                        cell.set_text_props(color=color)  # Hide text by making it same color as background
                    else:
                        cell.set_facecolor('#2A2A2A')
                cell.set_edgecolor('white')
        elif len(pivot_df.columns) == 0:
            ax_table.text(0.5, 0.5, 'No data available for table', 
                         ha='center', va='center', transform=ax_table.transAxes, 
                         color='white', fontsize=12)

    # Save the chart
    # output_path = f"{output_filename_base}.png"
    output_path = os.path.join(BASE_DATA_PATH, 'charts', f"{output_filename_base}.png")
    plt.subplots_adjust(hspace=0.2)  # Adjust spacing instead of tight_layout
    plt.savefig(output_path, facecolor='#1C1C1A', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved: {output_path}")


# --- Main Orchestration --- 

def main():
    all_perf_data = []
    all_cost_data = []

    # Load Snowflake and Databricks data
    for db in ['snowflake', 'databricks']:
        all_perf_data.extend(load_snowflake_databricks_data(db, 'results'))
        all_cost_data.extend(load_snowflake_databricks_data(db, 'costs'))

    # Load ClickHouse data
    all_perf_data.extend(load_clickhouse_data('performance'))
    all_cost_data.extend(load_clickhouse_data('cost'))

    if not all_perf_data and not all_cost_data:
        print("No data loaded. Exiting.")
        return

    df_perf = pd.DataFrame(all_perf_data)
    df_cost = pd.DataFrame(all_cost_data)

    # Standardize dataset_size to lowercase for consistent filtering
    if not df_perf.empty and 'dataset_size' in df_perf.columns:
        df_perf['dataset_size'] = df_perf['dataset_size'].astype(str).str.lower()
    if not df_cost.empty and 'dataset_size' in df_cost.columns:
        df_cost['dataset_size'] = df_cost['dataset_size'].astype(str).str.lower()

    # Filter out ClickHouse 'enterprise' costs, keep only 'scale' or non-ClickHouse
    if not df_cost.empty:
        # Condition: database is not 'clickhouse-cloud' OR (it is 'clickhouse-cloud' AND 'hardware_config' contains '(scale)')
        is_not_ch_enterprise_cost = ~(df_cost['database'].str.contains('clickhouse-cloud', case=False, na=False) & 
                                      df_cost['hardware_config'].str.contains('enterprise', case=False, na=False))
        df_cost = df_cost[is_not_ch_enterprise_cost].copy() # Use .copy() to avoid SettingWithCopyWarning later

        # Simplify ClickHouse hardware_config names for cost charts (remove '(scale)')
        # This will affect legend and table row labels for cost charts.
        def simplify_ch_cost_label(config_name):
            if 'clickhouse-cloud' in df_cost[df_cost['hardware_config'] == config_name]['database'].iloc[0] and \
               '(scale)' in config_name:
                return config_name.replace(' (scale)', '')
            return config_name
        
        if 'hardware_config' in df_cost.columns:
             df_cost['hardware_config'] = df_cost['hardware_config'].apply(simplify_ch_cost_label)

    print("\n--- Initial Data Load Summary ---")
    if not df_perf.empty:
        print("Performance Data (df_perf):")
        print(f"  Total records: {len(df_perf)}")
        print(f"  Unique DBs: {df_perf['database'].unique()}")
        print(f"  Unique Hardware Configs: {df_perf['hardware_config'].unique()}")
        print(f"  Sample for Snowflake:\n{df_perf[df_perf['database'] == 'snowflake'].head().to_string()}")
        print(f"  Sample for Databricks:\n{df_perf[df_perf['database'] == 'databricks'].head().to_string()}")
        print(f"  Sample for ClickHouse:\n{df_perf[df_perf['database'] == 'clickhouse-cloud'].head().to_string()}")
    else:
        print("Performance Data: Empty")
    
    if not df_cost.empty:
        print("\nCost Data (df_cost):")
        print(f"  Total records: {len(df_cost)}")
        print(f"  Unique DBs: {df_cost['database'].unique()}")
        print(f"  Unique Hardware Configs: {df_cost['hardware_config'].unique()}")
        print(f"  Sample for Snowflake:\n{df_cost[df_cost['database'] == 'snowflake'].head().to_string()}")
        print(f"  Sample for Databricks:\n{df_cost[df_cost['database'] == 'databricks'].head().to_string()}")
        print(f"  Sample for ClickHouse:\n{df_cost[df_cost['database'] == 'clickhouse-cloud'].head().to_string()}")
    else:
        print("Cost Data: Empty")
    print("--- End Initial Data Load Summary ---\n")

    # --- Define queries for different chart groups ---
    all_queries = sorted(list(set(df_perf['query_name']))) # Get all unique query names
    queries_group1_2 = [q for q in all_queries if q not in ['Query_10', 'Query_16']]
    queries_group3 = ['Query_10', 'Query_16']

    dataset_sizes = ['500m', '1b', '5b']

    # 1. Query performance (except Q10, Q16)
    for ds_size in dataset_sizes:
        df_filtered = df_perf[(df_perf['dataset_size'] == ds_size) & (df_perf['metric_type'] == 'performance')]
        if not queries_group1_2 or df_filtered.empty:
            print(f"No data or queries for perf chart (non Q10/16) for {ds_size}")
            continue
        generate_chart_and_table(df_filtered, 
                                 title=f"Query Performance (excl. Q10, Q16) - {ds_size} (Seconds)", 
                                 output_filename_base=f"perf_excl_q10_q16_{ds_size}",
                                 queries_to_plot=queries_group1_2,
                                 chart_type=ChartType.QUERY_PERFORMANCE)

    # 2. Query cost (except Q10, Q16)
    for ds_size in dataset_sizes:
        df_filtered = df_cost[df_cost['dataset_size'] == ds_size]
        # For cost, metric_type can be 'cost', 'cost_scale', 'cost_enterprise'
        # The plotting function will group by 'hardware_config' which includes tier for CH
        if not queries_group1_2 or df_filtered.empty:
            print(f"No data or queries for cost chart (non Q10/16) for {ds_size}")
            continue
        generate_chart_and_table(df_filtered, 
                                 title=f"Query Cost (excl. Q10, Q16) - {ds_size} ($)", 
                                 output_filename_base=f"cost_excl_q10_q16_{ds_size}",
                                 queries_to_plot=queries_group1_2,
                                 chart_type=ChartType.QUERY_COST)

    # 3. Query performance & cost for Q10, Q16
    for ds_size in dataset_sizes:
        # Perf for Q10, Q16
        df_filtered_perf = df_perf[(df_perf['dataset_size'] == ds_size) & (df_perf['metric_type'] == 'performance')]
        if not queries_group3 or df_filtered_perf.empty:
            print(f"No data or queries for perf chart (Q10/16) for {ds_size}")
        else:
            generate_chart_and_table(df_filtered_perf, 
                                     title=f"Query Performance (Q10, Q16) - {ds_size} (Seconds)", 
                                     output_filename_base=f"perf_q10_q16_{ds_size}",
                                     queries_to_plot=queries_group3,
                                     chart_type=ChartType.QUERY_PERFORMANCE)
        # Cost for Q10, Q16
        df_filtered_cost = df_cost[df_cost['dataset_size'] == ds_size]
        if not queries_group3 or df_filtered_cost.empty:
            print(f"No data or queries for cost chart (Q10/16) for {ds_size}")
        else:
            generate_chart_and_table(df_filtered_cost, 
                                     title=f"Query Cost (Q10, Q16) - {ds_size} ($)", 
                                     output_filename_base=f"cost_q10_q16_{ds_size}",
                                     queries_to_plot=queries_group3,
                                     chart_type=ChartType.QUERY_COST)

    # 4. Total performance (sum of all queries)
    df_total_perf_data_for_labels = None # Initialize to store perf data for labels
    if not df_perf.empty and all_queries:
        df_total_perf_grouped = df_perf.groupby(['database', 'hardware_config', 'dataset_size', 'metric_type'])['value'].sum().reset_index()
        df_total_perf_grouped['query_name'] = 'Total_Performance' # Use a single 'query' for plotting
        df_total_perf_data_for_labels = df_total_perf_grouped.copy() # Save for use in cost chart labels

        # Prepare cost data for labels in performance charts, if available
        df_cost_labels_prepared = pd.DataFrame()
        if not df_cost.empty:
            df_total_cost_for_labels = df_cost.groupby(['database', 'hardware_config', 'dataset_size', 'metric_type'])['value'].sum().reset_index()
            df_cost_labels_prepared = df_total_cost_for_labels[['hardware_config', 'dataset_size', 'value']].copy()
            df_cost_labels_prepared.rename(columns={'value': 'total_cost_value'}, inplace=True)

        for ds_size in dataset_sizes:
            df_filtered_perf_total = df_total_perf_grouped[(df_total_perf_grouped['dataset_size'] == ds_size) & (df_total_perf_grouped['metric_type'] == 'performance')].copy()
            if df_filtered_perf_total.empty:
                print(f"No data for total perf chart for {ds_size}")
                continue
            
            # Merge cost labels if cost data was prepared
            if not df_cost_labels_prepared.empty:
                current_ds_cost_labels = df_cost_labels_prepared[df_cost_labels_prepared['dataset_size'] == ds_size]
                if not current_ds_cost_labels.empty:
                    df_filtered_perf_total = pd.merge(
                        df_filtered_perf_total,
                        current_ds_cost_labels[['hardware_config', 'total_cost_value']], # Only merge these two columns
                        on='hardware_config',
                        how='left'
                    )
            
            generate_chart_and_table(df_filtered_perf_total,
                                     title=f"Total Query Performance - {ds_size} (Seconds)",
                                     output_filename_base=f"total_perf_{ds_size}",
                                     queries_to_plot=['Total_Performance'],
                                     chart_type=ChartType.TOTAL_PERFORMANCE,
                                     horizontal_bars=True)
    else:
        print("Skipping total performance charts due to no performance data or no queries identified.")

    # 5. Total cost (sum of all queries)
    if not df_cost.empty and all_queries:
        df_total_cost_grouped = df_cost.groupby(['database', 'hardware_config', 'dataset_size', 'metric_type'])['value'].sum().reset_index()
        df_total_cost_grouped['query_name'] = 'Total_Cost'

        # Prepare performance data for labels, if available
        df_perf_labels_prepared = pd.DataFrame()
        if df_total_perf_data_for_labels is not None and not df_total_perf_data_for_labels.empty:
            df_perf_labels_prepared = df_total_perf_data_for_labels[
                df_total_perf_data_for_labels['metric_type'] == 'performance'
            ][['hardware_config', 'dataset_size', 'value']].copy()
            df_perf_labels_prepared.rename(columns={'value': 'total_perf_value'}, inplace=True)

        for ds_size in dataset_sizes:
            # Filter total cost data for the current dataset size
            df_filtered_cost_total = df_total_cost_grouped[df_total_cost_grouped['dataset_size'] == ds_size].copy()
            if df_filtered_cost_total.empty:
                print(f"No data for total cost chart for {ds_size}")
                continue

            # Merge performance labels if performance data was prepared
            if not df_perf_labels_prepared.empty:
                current_ds_perf_labels = df_perf_labels_prepared[df_perf_labels_prepared['dataset_size'] == ds_size]
                if not current_ds_perf_labels.empty:
                    df_filtered_cost_total = pd.merge(
                        df_filtered_cost_total,
                        current_ds_perf_labels[['hardware_config', 'total_perf_value']], # Only merge these two columns
                        on='hardware_config',
                        how='left'
                    )
            
            generate_chart_and_table(df_filtered_cost_total,
                                     title=f"Total Query Cost - {ds_size} ($)",
                                     output_filename_base=f"total_cost_{ds_size}",
                                     queries_to_plot=['Total_Cost'],
                                     chart_type=ChartType.TOTAL_COST,
                                     horizontal_bars=True)
    else:
        print("Skipping total cost charts due to no cost data or no queries identified.")

if __name__ == "__main__":
    main()
