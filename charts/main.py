import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.gridspec as gridspec
import numpy as np


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
            # Extract scale and hardware from filename, e.g., result_1b_2n_60c_240g_....json
            parts = filename.replace("result_", "").split('_')
            dataset_size = parts[0]
            # Assuming hardware config is everything between the first and last two parts (date, time)
            hardware_config_ch = "_" .join(parts[1:-2]) 
            # Map to a more generic hardware label if needed, for now use raw
            # For ClickHouse, hardware might be like '2n_60c_240g'

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

    """Darkens a given hex color by a factor using JCh color space for better perceptual results."""
    if not isinstance(hex_color, str) or not hex_color.startswith('#') or len(hex_color) != 7:
        # print(f"Invalid hex color format: {hex_color}, returning as is.")
        return hex_color # Not a valid hex color string
    try:
        # Convert hex to RGB tuple (0-1 range)
        rgb_color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        
        # Convert RGB to JCh (luminance, chroma, hue)
        jch_color = cspace_converter("sRGB1", "JCh")(rgb_color)
        
        # Reduce luminance (J)
        new_j = jch_color[0] * factor
        
        # Prevent making colors too dark or black, ensure some minimum luminance
        if jch_color[0] < 10 and factor < 1.0: # if original luminance is very low
             new_j = max(2, jch_color[0] * 0.95) # Darken very slightly, min luminance 2
        elif new_j < 5 and factor < 1.0: # Don't let it become black
             new_j = 5 # Minimum luminance of 5
        new_j = max(0, new_j) # Ensure J is not negative

        jch_darkened_color = (new_j, jch_color[1], jch_color[2])
        
        # Convert back to RGB
        rgb_darkened = cspace_converter("JCh", "sRGB1")(jch_darkened_color)
        
        # Clip values to [0, 1] and convert to hex string
        hex_darkened = '#' + ''.join(f'{int(max(0, min(1, c)) * 255):02x}' for c in rgb_darkened)
        return hex_darkened
    except Exception as e:
        # print(f"Error darkening color {hex_color} with factor {factor}: {e}") # Optional debug
        return hex_color # Fallback to original color on error


# --- Plotting Function (Placeholder) ---

def generate_chart_and_table(df_filtered, title, output_filename_base, queries_to_plot, horizontal_bars=False):
    """Generates and saves a bar chart with a data table below it."""
    print(f"\n--- Generating chart: {title} ---")
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

    # For 'Total' charts, sort by value (best to worst, so ascending)
    if ('total query performance' in title.lower() or 'total query cost' in title.lower()) and not pivot_df.empty and len(pivot_df.columns) == 1:
        # Sort by the single column of values, ascending (lowest is best)
        pivot_df = pivot_df.sort_values(by=pivot_df.columns[0], ascending=True)
    
    # Sort hardware_config for consistent plotting order if desired (e.g., by a predefined list or name)
    # For now, default pandas sort is used for non-total charts or if sorting by value isn't applicable.

    num_configs = len(pivot_df.index)
    num_queries = len(pivot_df.columns)

    if num_configs == 0 or num_queries == 0:
        print(f"Skipping chart '{title}' due to no data after pivoting.")
        return

    fig = plt.figure(figsize=(max(15, num_queries * 1.5), 10)) # Increased height for table
    fig.patch.set_facecolor('#1C1C1A') # Set figure background

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15) # 2 rows, 1 col; chart gets 3 parts, table 1 part. hspace adds space.
    ax_chart = fig.add_subplot(gs[0]) # Axis for the bar chart
    ax_chart.set_facecolor('#1C1C1A') # Set chart area background

    # Set spine colors to white
    ax_chart.spines['bottom'].set_color('white')
    ax_chart.spines['top'].set_color('white') 
    ax_chart.spines['right'].set_color('white')
    ax_chart.spines['left'].set_color('white')

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
            # Cycle through the predefined colors if there are more configs than colors
            final_color_for_cfg = color_list_for_db[count % len(color_list_for_db)]
        elif isinstance(color_list_for_db, str): # Fallback if a single color string is still defined for some reason
            final_color_for_cfg = color_list_for_db 
        else: # Ultimate fallback to grey if no valid color/list found
            final_color_for_cfg = '#808080' 

        config_colors[config_name_in_pivot] = final_color_for_cfg
        db_config_counts[current_db_base] = count + 1

        # Debug print for color assignment (BaseColor concept changes here)
        print(f"  ColorDebug: Config='{config_name_in_pivot}', DB_Base='{current_db_base}', ColorIndex={count}, FinalColor='{final_color_for_cfg}'")
    # --- End of color variation logic ---

    # Iterate through the original pivot_df index order for plotting to maintain user's expected bar order
    for i, (config_name, row) in enumerate(pivot_df.iterrows()):
        color = config_colors.get(config_name, '#808080') # Get pre-calculated color
        
        position_offset = (i - num_configs / 2 + 0.5) * bar_width
        positions = indices + position_offset

        if horizontal_bars:
            # For horizontal bars, 'indices' are y-coordinates, 'row.values' are widths
            bars = ax_chart.barh(positions, row.values, height=bar_width, label=config_name, color=color, align='center', edgecolor='white', linewidth=0.5) # Add subtle white edge to bars

            # Add performance labels if this is a Total Cost chart and data is available
            if horizontal_bars and 'total_perf_value' in df_filtered.columns and 'total query cost' in title.lower():
                cost_value = row.values[0] # This is the bar's width (the cost)
                y_bar_center = positions[0]    # This is the y-coordinate of the bar's center

                # Retrieve the performance value for this specific config_name from the original df_filtered
                perf_series = df_filtered.loc[df_filtered['hardware_config'] == config_name, 'total_perf_value']
                
                if not perf_series.empty and pd.notna(perf_series.iloc[0]):
                    perf_value = perf_series.iloc[0]
                    label_text = f"(time: {perf_value:.1f}s)" # Format as, e.g., "(time: 123.4s)"
                    
                    # Determine text position and alignment dynamically
                    current_xlim = ax_chart.get_xlim() # xmin, xmax for data
                    plot_data_range = current_xlim[1] - current_xlim[0]
                    if plot_data_range == 0: # Avoid division by zero if range is zero
                        plot_data_range = 1 # Default to 1 to prevent errors, though unlikely for a bar chart
                    
                    padding_abs = plot_data_range * 0.015 # 1.5% of range as padding, slightly more than before

                    # Threshold for placing text inside: if bar is longer than 85% of x-axis max value
                    threshold_for_inside = current_xlim[1] * 0.85

                    text_x = 0
                    ha = 'left' # Default horizontal alignment

                    if cost_value > threshold_for_inside:
                        # Bar is long, place text inside, to the left of the bar's end
                        text_x = cost_value - padding_abs
                        ha = 'right'
                    else:
                        # Bar is not "too long", place text outside, to the right of the bar's end
                        text_x = cost_value + padding_abs
                        ha = 'left'

                    if cost_value == 0:
                        text_x = padding_abs # Place at padding distance from origin
                        ha = 'left'
                    elif cost_value < padding_abs and ha == 'right': 
                        # If bar is extremely short AND logic decided to place inside (unlikely with threshold_for_inside)
                        # revert to placing outside to avoid text appearing left of y-axis or cramped.
                        text_x = cost_value + padding_abs
                        ha = 'left'
                    
                    text_label_color = 'white' # Default to white
                    if ha == 'right': # Text is inside the bar
                        text_label_color = 'black' # Use black for better contrast inside light bars
                    
                    ax_chart.text(text_x, y_bar_center, label_text,
                                  va='center', ha=ha, color=text_label_color, fontsize=8)
            
            # Add cost labels if this is a Total Performance chart and cost data is available
            elif horizontal_bars and 'total_cost_value' in df_filtered.columns and 'total query performance' in title.lower():
                perf_value = row.values[0] # This is the bar's width (the performance time)
                y_bar_center = positions[0]    # This is the y-coordinate of the bar's center

                # For performance charts, we need to find the corresponding cost value
                # We'll need to look up the cost from the cost data based on hardware_config and dataset_size
                # For now, let's add a placeholder that can be enhanced when cost data is available
                # This would require passing cost data to the function or having it available in df_filtered
                
                # Check if we have cost data available in the dataframe (this would need to be added in the main function)
                cost_series = df_filtered.loc[df_filtered['hardware_config'] == config_name, 'total_cost_value']
                
                if not cost_series.empty and pd.notna(cost_series.iloc[0]):
                    cost_value = cost_series.iloc[0]
                    label_text = f"(cost: ${cost_value:.2f})" # Format as, e.g., "(cost: $1.23)"
                    
                    # Use similar positioning logic as for cost charts
                    current_xlim = ax_chart.get_xlim()
                    plot_data_range = current_xlim[1] - current_xlim[0]
                    if plot_data_range == 0:
                        plot_data_range = 1
                        
                    padding_abs = plot_data_range * 0.015
                    threshold_for_inside = current_xlim[1] * 0.85

                    text_x = 0
                    ha = 'left'

                    if perf_value > threshold_for_inside:
                        text_x = perf_value - padding_abs
                        ha = 'right'
                    else:
                        text_x = perf_value + padding_abs
                        ha = 'left'

                    if perf_value == 0:
                        text_x = padding_abs
                        ha = 'left'
                    elif perf_value < padding_abs and ha == 'right':
                        text_x = perf_value + padding_abs
                        ha = 'left'
                    
                    text_label_color = 'white'
                    if ha == 'right':
                        text_label_color = 'black'
                    
                    ax_chart.text(text_x, y_bar_center, label_text,
                                  va='center', ha=ha, color=text_label_color, fontsize=8)
        else:
            ax_chart.bar(positions, row.values, width=bar_width, label=config_name, color=color, edgecolor='white', linewidth=0.5) # Add subtle white edge to bars

    if horizontal_bars:
        ax_chart.invert_yaxis() # To have Query 01 at the top
        ax_chart.grid(True, axis='x', linestyle='--', color='white', alpha=0.5) # Grid for horizontal bars
        ax_chart.set_yticks([]) # Remove y-axis ticks
    else:
        ax_chart.set_xticks(indices)
        # Check if this is a per-query chart (not a total chart)
        if not ('total query performance' in title.lower() or 'total query cost' in title.lower()):
            # Transform Query_02 format to Q2 format for per-query charts
            simplified_labels = []
            for label in pivot_df.columns:
                if label.startswith('Query_'):
                    # Extract the number part and convert Query_02 to Q2
                    query_num = label.replace('Query_', '')
                    simplified_labels.append(f'Q{int(query_num)}')
                else:
                    simplified_labels.append(label)
            ax_chart.set_xticklabels(simplified_labels, color='white')  # No rotation for per-query charts
        else:
            ax_chart.set_xticklabels(pivot_df.columns, rotation=45, ha="right", color='white')  # Keep rotation for total charts
        ax_chart.grid(True, axis='y', linestyle='--', color='white', alpha=0.5) # Grid for vertical bars

    # Tick parameter colors
    ax_chart.tick_params(axis='x', colors='white')
    ax_chart.tick_params(axis='y', colors='white')

    ax_chart.set_title(title, fontsize=16, color='white')
    # legend = ax_chart.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left') # Legend removed
    # legend.get_title().set_color('white')
    # for text in legend.get_texts():
    #     text.set_color('white')
    # legend.get_frame().set_facecolor('#333333') # Darker background for legend box
    # legend.get_frame().set_edgecolor('white')

    # --- Table creation (uses a new axis area) ---
    ax_table_area = fig.add_subplot(gs[1]) # Axis for the table
    ax_table_area.axis('off') # No visible axis for the table area
    ax_table_area.set_facecolor('#1C1C1A') # Ensure table area background is also dark

    # Format table data to show up to 3 decimal places with trailing zeros for consistency
    def format_number(value):
        if pd.isna(value):
            return 'N/A'
        try:
            # Format to exactly 3 decimal places with trailing zeros for consistency
            formatted = f"{float(value):.3f}"
            return formatted
        except (ValueError, TypeError):
            return str(value)
    
    # Apply formatting to all numeric values in the pivot table
    table_data = []
    for row in pivot_df.values:
        formatted_row = [format_number(val) for val in row]
        table_data.append(formatted_row)
    
    row_labels = pivot_df.index.tolist()
    col_labels = pivot_df.columns.tolist()

    # Add column headers: one for color swatch (empty), one for row labels (empty), then data col_labels
    table_col_labels_with_header = ['', ''] + col_labels

    the_table = Table(ax_table_area, bbox=[0, 0, 1, 1]) # Table fills its allocated subplot area
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)

    num_table_rows = len(row_labels) + 1 # +1 for header
    # num_table_cols = len(table_col_labels_with_header) # Recalculate based on new structure

    cell_height = 1.0 / num_table_rows
    
    # Define widths for columns:
    swatch_col_width = 0.04  # Narrow column for color swatch
    label_name_col_width = 0.21 # Width for the hardware_config name
    num_data_cols = len(col_labels)
    
    if num_data_cols > 0:
        remaining_width_for_data = 1.0 - swatch_col_width - label_name_col_width
        data_col_individual_width = remaining_width_for_data / num_data_cols
        col_widths = [swatch_col_width, label_name_col_width] + [data_col_individual_width] * num_data_cols
    elif num_data_cols == 0 and ('total query' in title.lower()): # Total charts have one data column conceptually, but pivot_df might be empty of data cols
        # For 'Total' charts, the single value is effectively the data. The 'col_labels' might be ['Total_Cost'] or ['Total_Performance']
        # If pivot_df.columns was used for col_labels, it might be 1. Let's assume it's 1 data column for totals.
        remaining_width_for_data = 1.0 - swatch_col_width - label_name_col_width
        # Ensure remaining_width_for_data is not negative if label_name_col_width is too large
        data_col_individual_width = max(0, remaining_width_for_data) 
        col_widths = [swatch_col_width, label_name_col_width, data_col_individual_width]
    else: # No data columns at all, e.g. if something went wrong or no queries_to_plot
        col_widths = [swatch_col_width, 1.0 - swatch_col_width] # Swatch and full width for label name

    num_table_cols = len(col_widths) # Update based on actual col_widths array

    table_header_color = '#404040' # Dark grey for header background
    table_edge_color = '#666666' # Lighter grey for cell edges
    table_text_color = 'white'

    # Add header row
    for j, label_text in enumerate(table_col_labels_with_header):
        # Ensure we don't try to access col_widths[j] if j >= len(col_widths)
        # This can happen if table_col_labels_with_header has more items than col_widths implies (e.g. for Total charts)
        if j < len(col_widths):
            width_for_cell = col_widths[j]
            cell = the_table.add_cell(0, j, width_for_cell, cell_height, text=label_text, loc='center',
                                      facecolor=table_header_color, edgecolor=table_edge_color)
            cell.get_text().set_color(table_text_color)
        # else: print(f"Warning: Skipping header cell for index {j} due to col_width mismatch.")

    # Add data rows
    for i, row_label_text in enumerate(row_labels):
        # Column 0: Color Swatch
        swatch_color = config_colors.get(row_label_text, '#808080') # Get color from pre-calculated dict
        cell_swatch = the_table.add_cell(i + 1, 0, col_widths[0], cell_height, text='', loc='center',
                                         facecolor=swatch_color, edgecolor=table_edge_color)
        # cell_swatch.get_text().set_color(table_text_color) # No text in swatch cell

        # Column 1: Row Label Text (hardware_config name)
        # Use the same background color as other headers for consistency, or a different one if preferred
        cell_label_text = the_table.add_cell(i + 1, 1, col_widths[1], cell_height, text=row_label_text, loc='left',
                                             facecolor=table_header_color, edgecolor=table_edge_color)
        cell_label_text.get_text().set_color(table_text_color)
        cell_label_text.PAD = 0.02 # Add a bit of padding to the left of the text

        # Data cells (starting from column 2)
        for j, val in enumerate(table_data[i]):
            # Ensure we don't try to access col_widths[j+2] if it's out of bounds
            if (j + 2) < len(col_widths):
                width_for_data_cell = col_widths[j+2]
                cell_data = the_table.add_cell(i + 1, j + 2, width_for_data_cell, cell_height, text=str(val), loc='center',
                                                 facecolor='#1C1C1A', edgecolor=table_edge_color) # Cell background same as chart
                cell_data.get_text().set_color(table_text_color)
            # else: print(f"Warning: Skipping data cell for row {i}, col {j+2} due to col_width mismatch.")
    
    ax_table_area.add_table(the_table)
    # --- End of table creation changes ---

    plt.tight_layout(pad=1.0, h_pad=2.0) # Adjust overall layout

    output_path = os.path.join(BASE_DATA_PATH, 'charts_output', f"{output_filename_base}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Chart saved to {output_path}")
    plt.close(fig)

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
                                 title=f"Query Performance (excl. Q10, Q16) - {ds_size}", 
                                 output_filename_base=f"perf_excl_q10_q16_{ds_size}",
                                 queries_to_plot=queries_group1_2)

    # 2. Query cost (except Q10, Q16)
    for ds_size in dataset_sizes:
        df_filtered = df_cost[df_cost['dataset_size'] == ds_size]
        # For cost, metric_type can be 'cost', 'cost_scale', 'cost_enterprise'
        # The plotting function will group by 'hardware_config' which includes tier for CH
        if not queries_group1_2 or df_filtered.empty:
            print(f"No data or queries for cost chart (non Q10/16) for {ds_size}")
            continue
        generate_chart_and_table(df_filtered, 
                                 title=f"Query Cost (excl. Q10, Q16) - {ds_size}", 
                                 output_filename_base=f"cost_excl_q10_q16_{ds_size}",
                                 queries_to_plot=queries_group1_2)

    # 3. Query performance & cost for Q10, Q16
    for ds_size in dataset_sizes:
        # Perf for Q10, Q16
        df_filtered_perf = df_perf[(df_perf['dataset_size'] == ds_size) & (df_perf['metric_type'] == 'performance')]
        if not queries_group3 or df_filtered_perf.empty:
            print(f"No data or queries for perf chart (Q10/16) for {ds_size}")
        else:
            generate_chart_and_table(df_filtered_perf, 
                                     title=f"Query Performance (Q10, Q16) - {ds_size}", 
                                     output_filename_base=f"perf_q10_q16_{ds_size}",
                                     queries_to_plot=queries_group3)
        # Cost for Q10, Q16
        df_filtered_cost = df_cost[df_cost['dataset_size'] == ds_size]
        if not queries_group3 or df_filtered_cost.empty:
            print(f"No data or queries for cost chart (Q10/16) for {ds_size}")
        else:
            generate_chart_and_table(df_filtered_cost, 
                                     title=f"Query Cost (Q10, Q16) - {ds_size}", 
                                     output_filename_base=f"cost_q10_q16_{ds_size}",
                                     queries_to_plot=queries_group3)

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
                                     title=f"Total Query Performance - {ds_size}",
                                     output_filename_base=f"total_perf_{ds_size}",
                                     queries_to_plot=['Total_Performance'],
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
                                     title=f"Total Query Cost - {ds_size}",
                                     output_filename_base=f"total_cost_{ds_size}",
                                     queries_to_plot=['Total_Cost'],
                                     horizontal_bars=True)
    else:
        print("Skipping total cost charts due to no cost data or no queries identified.")

if __name__ == "__main__":
    main()
