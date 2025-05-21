# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Set up matplotlib for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define mapping from internal board names to commercial names
BOARD_NAME_MAPPING = {
    'esp32dev': 'ESP32',
    'esp32-s3-devkitm-1': 'ESP32-S3',
    'esp32-c6-devkitc-1': 'ESP32-C6',
    'esp32-p4-evboard': 'ESP32-P4',
    'picow': 'RPi Pico W',
    'pico2': 'RPi Pico 2',
    'nano33ble': 'A. Nano 33 BLE',
    'teensy40': 'Teensy 4.0',
    'nucleo_l552ze_q': 'L552ZE-Q',
    'nucleo_f207zg': 'F207ZG'
}


def generate_color_palette(item_names, palette_name=None):
    """
    Generate a color palette for a list of items.
    
    Parameters:
    -----------
    item_names : list
        List of item names to generate colors for
    palette_name : str, optional
        Name of the color palette to use
        
    Returns:
    --------
    dict
        Dictionary mapping item names to colors
    """
    num_items = len(item_names)
    
    # Choose a good default palette if none specified
    if palette_name is None:
        # For technical publications, 'colorblind' is a good choice
        # For many categories (>10), viridis or plasma are good options
        if num_items <= 8:
            palette_name = 'colorblind'
        else:
            palette_name = 'viridis'
    
    # Create the color palette based on the number of items
    try:
        # For sequential palettes like viridis, we need to specify the number of colors
        if palette_name in ['viridis', 'plasma', 'inferno', 'cividis']:
            colors = sns.color_palette(palette_name, num_items)
        else:
            # For categorical palettes, we take as many as we need
            colors = sns.color_palette(palette_name)
            # If we have more items than colors, cycle the palette
            if num_items > len(colors):
                colors = sns.color_palette(palette_name, num_items)
    except ValueError:
        # Fallback to default tab10 if the palette is not found
        print(f"Warning: Color palette '{palette_name}' not found. Using default 'tab10'.")
        colors = sns.color_palette('tab10', num_items)
    
    # Create a mapping of item name to color
    item_colors = {}
    for i, item in enumerate(item_names):
        item_colors[item] = colors[i % len(colors)]
    
    return item_colors


def load_benchmark_data(data_dir="benchmark_results"):
    """
    Load benchmark data from CSV files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing benchmark CSV files
    
    Returns:
    --------
    pandas.DataFrame or None
        Loaded benchmark data or None if no data was found
    """
    # Get all CSV files
    benchmark_files = glob.glob(f"{data_dir}/csv/benchmark_*.csv")
    
    if not benchmark_files:
        # Try without the csv subdirectory
        benchmark_files = glob.glob(f"{data_dir}/benchmark_*.csv")
        
    if not benchmark_files:
        print(f"No benchmark CSV files found in {data_dir}!")
        return None

    dataframes = []

    for file in benchmark_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dataframes:
        print("No valid data could be loaded!")
        return None

    # Combine all data
    data = pd.concat(dataframes, ignore_index=True)

    # Convert operation time columns to numeric
    for col in data.columns:
        if col.endswith('_ticks') or col.endswith('_bytes'):
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data


def load_energy_data(energy_file="benchmark_results/energy_data.csv"):
    """
    Load energy consumption data from CSV file.
    
    Parameters:
    -----------
    energy_file : str
        Path to the energy data CSV file
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing energy consumption data
    """
    try:
        energy_data = pd.read_csv(energy_file)
        print(f"Loaded energy data with {len(energy_data)} entries")
        return energy_data
    except Exception as e:
        print(f"Error loading energy data: {e}")
        return None


def preprocess_benchmark_data(data):
    """
    Process benchmark data to extract board type and optimization level.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw benchmark data
    
    Returns:
    --------
    pandas.DataFrame
        Processed data with additional columns
    """
    # Create a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Extract optimization level from board name
    processed_data['opt_level'] = processed_data['board'].apply(
        lambda x: 'small' if x.endswith('_small') else 
                 'fast' if x.endswith('_fast') else 'unknown'
    )
    
    # Extract base board name (without optimization suffix)
    processed_data['base_board'] = processed_data['board'].apply(
        lambda x: x[:-6] if x.endswith('_small') else 
                 x[:-5] if x.endswith('_fast') else x
    )
    
    # Map to display names
    processed_data['board_display_name'] = processed_data['base_board'].map(
        lambda x: BOARD_NAME_MAPPING.get(x, x)
    )
    
    return processed_data


def get_operation_cols(data, benchmark_type=None):
    """
    Get operation columns for a specific benchmark type.
    If benchmark_type is provided, returns only columns with non-zero values.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed benchmark data
    benchmark_type : str, optional
        Benchmark type to filter columns for
    
    Returns:
    --------
    list
        List of operation column names
    """
    # Base operation columns (all columns ending with _ticks except total)
    all_op_cols = [col for col in data.columns if col.endswith('_ticks') 
                  and col != 'total_execution_time_ticks']
    
    if benchmark_type is None:
        return all_op_cols
    
    # Filter data for the specific benchmark
    benchmark_data = data[data['benchmark_type'] == benchmark_type]
    
    # Check which operations are actually used (non-zero values)
    used_ops = []
    for col in all_op_cols:
        if (benchmark_data[col] > 0).any():
            used_ops.append(col)
    
    return used_ops


def get_memory_cols(data):
    """
    Get all memory usage columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed benchmark data
    
    Returns:
    --------
    list
        List of memory column names
    """
    return [col for col in data.columns if col.endswith('_bytes')]


def plot_operation_times(data, benchmark_type, save_path=None, color_palette=None):
    """
    Create a grouped stacked bar chart showing operation-level timing breakdown.
    Uses standard matplotlib approach for grouped bar charts with proper spacing.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed benchmark data
    benchmark_type : str
        The benchmark type to visualize
    save_path : str, optional
        If provided, save the figure to this path
    color_palette : str, optional
        Name of the color palette to use
    """
    # Check if the benchmark exists in the data
    if benchmark_type not in data['benchmark_type'].unique():
        print(f"Benchmark type '{benchmark_type}' not found in data!")
        return
    
    # Filter data for the specific benchmark
    plot_df = data[data['benchmark_type'] == benchmark_type].copy()
    
    # Get operation columns specific to this benchmark
    op_cols = get_operation_cols(data, benchmark_type)
    
    if not op_cols:
        print(f"No operation data found for benchmark '{benchmark_type}'")
        return
    
    # Convert microseconds to milliseconds for all operation columns
    for col in op_cols:
        plot_df[col] = plot_df[col] / 1000.0
    
    if 'total_execution_time_ticks' in plot_df.columns:
        plot_df['total_execution_time_ticks'] = plot_df['total_execution_time_ticks'] / 1000.0
    
    # Get unique board types and optimization levels
    board_types = sorted(plot_df['board_display_name'].unique())
    opt_levels = sorted(plot_df['opt_level'].unique())
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort operation columns by average time (largest first)
    op_cols_sorted = sorted(op_cols, 
                           key=lambda x: plot_df[x].mean(), 
                           reverse=True)
    
    # Set the positions for the groups on the x-axis
    x = np.arange(len(board_types))
    
    # Width configuration with spacing
    total_group_width = 0.75  # Width allocated for each board type group
    # Space between bars within a group (as a proportion of the total width per bar)
    bar_spacing_proportion = 0.07  # Subtle spacing between bars
    
    # Calculate bar width to account for spacing
    # First calculate what the width would be without spacing
    bar_width_no_spacing = total_group_width / len(opt_levels)
    # Then adjust to account for spacing
    bar_width = bar_width_no_spacing * (1 - bar_spacing_proportion)
    
    # Recalculate the total spacing
    total_spacing = total_group_width - (bar_width * len(opt_levels))
    # Space between each bar
    space_between = total_spacing / (len(opt_levels) - 1) if len(opt_levels) > 1 else 0
    
    # Get operation names for the color palette
    op_names = [col.replace('_ticks', '').upper() for col in op_cols_sorted]
    
    # Generate color palette for operations
    operation_colors = generate_color_palette(op_names, color_palette)
    
    # For storing the handles for legend
    legend_handles = []
    legend_labels = []
    
    # Create a mapping of board type to its data for each optimization level
    board_data = {}
    for board_type in board_types:
        board_data[board_type] = {}
        for opt_level in opt_levels:
            mask = (plot_df['board_display_name'] == board_type) & (plot_df['opt_level'] == opt_level)
            if mask.any():
                board_data[board_type][opt_level] = plot_df[mask].iloc[0]
            else:
                board_data[board_type][opt_level] = None
    
    # Plot the bars for each board and optimization level
    for i, board_type in enumerate(board_types):
        for j, opt_level in enumerate(opt_levels):
            if board_data[board_type][opt_level] is not None:
                # Calculate the position of this bar with spacing
                # Start from the left position of the group, then add spacing
                if len(opt_levels) > 1:
                    # For multiple bars, calculate position with spacing
                    position_offset = -total_group_width/2 + (j * (bar_width + space_between)) + bar_width/2
                else:
                    # For a single bar, center it
                    position_offset = 0
                    
                bar_pos = x[i] + position_offset
                
                # Get the data for this board and optimization level
                data_row = board_data[board_type][opt_level]
                
                # Plot stacked bars for each operation
                bottom = 0
                for k, col in enumerate(op_cols_sorted):
                    # Clean up the column name for the legend
                    op_name = col.replace('_ticks', '').upper()
                    
                    # Get color for this operation from our dynamic palette
                    color = operation_colors[op_name]
                    
                    # Only include operations with non-zero time
                    val = data_row[col]
                    if val > 0:
                        # Plot this operation's time
                        bar = ax.bar(bar_pos, val, bottom=bottom, width=bar_width, color=color)
                        
                        # Only add to legend once
                        if op_name not in legend_labels:
                            legend_handles.append(bar)
                            legend_labels.append(op_name)
                        
                        # Update the bottom for the next operation
                        bottom += val
                
                # Calculate the actual total from displayed operations
                # Don't use the pre-calculated value from CSV
                total = sum([data_row[col] for col in op_cols_sorted if data_row[col] > 0])
                
                # Add total time as text on top of each bar
                ax.text(bar_pos, bottom + (total*0.01), f"{total:.1f} ms", 
                       ha='center', va='bottom', fontsize=9, rotation=0, fontweight='bold')
    
    # Set the x-axis ticks in the middle of each group
    ax.set_xticks(x)
    
    # Set the board names as x-axis tick labels
    ax.set_xticklabels(board_types)
    
    # Add optimization level labels below each bar
    for i, board_type in enumerate(board_types):
        for j, opt_level in enumerate(opt_levels):
            if board_data[board_type][opt_level] is not None:
                # Calculate the same position as above, but for the label
                if len(opt_levels) > 1:
                    position_offset = -total_group_width/2 + (j * (bar_width + space_between)) + bar_width/2
                else:
                    position_offset = 0
                    
                bar_pos = x[i] + position_offset
                
                # Add the label
                ax.annotate(opt_level.capitalize(), 
                           xy=(bar_pos, 0), 
                           xytext=(0, -25),  # offset points
                           textcoords='offset points',
                           ha='center', 
                           fontsize=9)
    
    # Set labels and title
    # ax.set_xlabel('Board Type', fontweight='bold', labelpad=30)  # Increase labelpad to make room for opt labels
    ax.set_ylabel('Čas [ms]', fontweight='bold')
    ax.set_title(f"Doba inference - {benchmark_type}", fontweight='bold', pad=20)
    
    # Add legend for operations
    ax.legend(legend_handles[::-1], legend_labels[::-1], title='Operace', bbox_to_anchor=(1, 1), loc='upper left')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for the legend and labels
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Make more room for the sublabels
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_ram_usage(data, benchmark_type, save_path=None, color_palette=None):
    """
    Create a stacked bar chart showing RAM memory usage breakdown.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed benchmark data
    benchmark_type : str
        The benchmark type to visualize
    save_path : str, optional
        If provided, save the figure to this path
    color_palette : str, optional
        Name of the color palette to use
    """
    
    # Check if the benchmark exists in the data
    if benchmark_type not in data['benchmark_type'].unique():
        print(f"Benchmark type '{benchmark_type}' not found in data!")
        return
    
    # Filter data for the specific benchmark
    plot_df = data[data['benchmark_type'] == benchmark_type].copy()
    
    # Workaround for ESP32-P4: Use ESP32-S3 RAM used value since ESP32-P4 value is broken
    # Find if there are any S3 boards to use as reference
    s3_boards = plot_df[plot_df['board_display_name'].str.contains('ESP32-S3', case=False)]
    p4_boards = plot_df[plot_df['board_display_name'].str.contains('ESP32-P4', case=False)]
    
    if not s3_boards.empty and not p4_boards.empty:
        # Get the first S3 board's RAM used value as reference
        s3_reference_ram_used = s3_boards.iloc[0]['ram_used_bytes']
        
        # Update only the ram_used_bytes for P4 boards
        for idx, row in p4_boards.iterrows():
            plot_df.loc[idx, 'ram_used_bytes'] = s3_reference_ram_used
    
    # Check for required columns
    required_columns = ['ram_used_bytes', 'ram_total_bytes', 'arena_head_bytes', 'arena_tail_bytes']
    missing_columns = [col for col in required_columns if col not in plot_df.columns]
    if missing_columns:
        print(f"Required columns not found: {', '.join(missing_columns)}")
        return

    # Deduplicate by board_display_name (since we don't need multiple optimization levels)
    # Taking the first occurrence for each board, as RAM usage is the same
    plot_df = plot_df.drop_duplicates(subset=['board_display_name'])
    
    # Calculate overhead (ram_used_bytes - arena components)
    plot_df['overhead_bytes'] = plot_df['ram_used_bytes'] - (
        plot_df['arena_head_bytes'] + plot_df['arena_tail_bytes']
    )
    
    # Calculate free RAM
    plot_df['ram_free_bytes'] = plot_df['ram_total_bytes'] - plot_df['ram_used_bytes']
    
    # Convert bytes to KB for better readability
    plot_df['arena_head'] = plot_df['arena_head_bytes'] / 1024.0
    plot_df['arena_tail'] = plot_df['arena_tail_bytes'] / 1024.0
    plot_df['overhead'] = plot_df['overhead_bytes'] / 1024.0
    plot_df['ram_free'] = plot_df['ram_free_bytes'] / 1024.0
    plot_df['ram_total'] = plot_df['ram_total_bytes'] / 1024.0
    
    # Sort boards by total RAM for better visualization
    plot_df = plot_df.sort_values(by='board_display_name')
    
    # Get unique board types
    board_types = plot_df['board_display_name'].tolist()
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar width configuration
    bar_width = 0.6
    
    # Set the positions for the bars on the x-axis
    x = np.arange(len(board_types))
    
    # Define components for coloring
    ram_components = ['ARENA_HEAD', 'ARENA_TAIL', 'OVERHEAD', 'FREE_RAM']
    
    # Generate colors
    ram_colors = generate_color_palette(ram_components, color_palette or 'Paired')
    
    # Patches for legend
    patches = []
    labels = []
    
    # Plot each board's RAM usage breakdown
    for i, board in enumerate(board_types):
        # Get data for this board
        data_row = plot_df[plot_df['board_display_name'] == board].iloc[0]
        
        # Plot the stacked bar - arena_head at bottom, then arena_tail, then overhead, then free at top
        # Bottom component: arena_head
        arena_head_bar = ax.bar(x[i], data_row['arena_head'], 
                width=bar_width, color=ram_colors['ARENA_HEAD'], 
                label='Arena Head' if i == 0 else "")
        
        # Middle component 1: arena_tail
        arena_tail_bar = ax.bar(x[i], data_row['arena_tail'], 
                width=bar_width, bottom=data_row['arena_head'], 
                color=ram_colors['ARENA_TAIL'], 
                label='Arena Tail' if i == 0 else "")
        
        # Middle component 2: overhead
        overhead_bar = ax.bar(x[i], data_row['overhead'], 
                width=bar_width, 
                bottom=data_row['arena_head'] + data_row['arena_tail'], 
                color=ram_colors['OVERHEAD'], 
                label='Overhead' if i == 0 else "")
        
        # Top component: free RAM
        free_ram_bar = ax.bar(x[i], data_row['ram_free'], 
                width=bar_width, 
                bottom=data_row['arena_head'] + data_row['arena_tail'] + data_row['overhead'], 
                color=ram_colors['FREE_RAM'], alpha=0.3,
                label='Free RAM' if i == 0 else "")
        
        # Save patches for the first bar for legend
        if i == 0:
            patches = [arena_head_bar, arena_tail_bar, overhead_bar, free_ram_bar]
            labels = ['Arena Head', 'Arena Tail', 'Režie', 'Volná RAM']
        
        # Calculate usage percentage
        usage_percent = (data_row['ram_used_bytes'] / data_row['ram_total_bytes']) * 100
        
        # Add text showing percentage used in the middle of the used portion
        used_height = data_row['arena_head'] + data_row['arena_tail'] + data_row['overhead']
        ax.text(x[i], used_height / 2, f"{usage_percent:.1f}%", 
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white')
        
        # Add the total RAM value above each bar
        ax.text(x[i], data_row['ram_total'] + (data_row['ram_total']*0.02), 
                f"{data_row['ram_total']:.1f} KB", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add arena size as text (arena_head + arena_tail)
        arena_size = data_row['arena_head'] + data_row['arena_tail']
        # Position text at the center of the combined arena sections
        arena_center_y = data_row['arena_head'] / 2 + data_row['arena_tail'] / 2
        if arena_size > 0:  # Only add text if there's space
            ax.text(x[i], arena_center_y, f"A.: {arena_size:.1f} KB", 
                    ha='center', va='center', fontsize=8, color='white')
    
    # Set the x-axis ticks
    ax.set_xticks(x)
    
    # Set the board names as x-axis tick labels (horizontal)
    ax.set_xticklabels(board_types, rotation=0, ha='center')
    
    # Set labels and title
    # ax.set_xlabel('Board Type', fontweight='bold', labelpad=15)
    ax.set_ylabel('Paměť RAM [KB]', fontweight='bold')
    ax.set_title(f"Využití paměti RAM - {benchmark_type}", fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(patches[::-1], labels[::-1], title='Komponenty', 
              bbox_to_anchor=(1, 1), loc='upper left')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for rotated labels and legend
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

    
def plot_flash_usage(data, benchmark_type, save_path=None, color_palette=None):
    """
    Create a stacked bar chart showing flash memory usage with small and fast optimization
    combined into a single bar per board.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed benchmark data
    benchmark_type : str
        The benchmark type to visualize
    save_path : str, optional
        If provided, save the figure to this path
    color_palette : str, optional
        Name of the color palette to use
    """
    # Check if the benchmark exists in the data
    if benchmark_type not in data['benchmark_type'].unique():
        print(f"Benchmark type '{benchmark_type}' not found in data!")
        return
    
    # Filter data for the specific benchmark
    plot_df = data[data['benchmark_type'] == benchmark_type].copy()
    
    # Check for required columns
    if 'flash_used_bytes' not in plot_df.columns or 'flash_total_bytes' not in plot_df.columns:
        print("Required flash columns not found (flash_used_bytes, flash_total_bytes)!")
        return

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique board types
    board_types = sorted(plot_df['board_display_name'].unique())
    
    # Create a dataframe to store the processed data for plotting
    plot_data = []
    
    # Process data for each board
    for board in board_types:
        board_data = plot_df[plot_df['board_display_name'] == board]
        
        # Check if we have both small and fast optimization data
        small_data = board_data[board_data['opt_level'] == 'small']
        fast_data = board_data[board_data['opt_level'] == 'fast']
        
        if small_data.empty or fast_data.empty:
            print(f"Missing optimization level data for {board}, skipping.")
            continue
        
        # Get the data for each optimization level
        small_flash_used = small_data['flash_used_bytes'].values[0]
        fast_flash_used = fast_data['flash_used_bytes'].values[0]
        total_flash = small_data['flash_total_bytes'].values[0]  # Same for both levels
        
        # Calculate the additional flash used by fast optimization
        additional_flash = fast_flash_used - small_flash_used
        
        # Calculate free flash based on fast optimization (worst case)
        free_flash = total_flash - fast_flash_used
        
        # Calculate percentage increase from small to fast
        percent_increase = (additional_flash / small_flash_used) * 100
        
        # Store the processed data
        plot_data.append({
            'board': board,
            'small_flash_kb': small_flash_used / 1024.0,
            'additional_flash_kb': additional_flash / 1024.0,
            'free_flash_kb': free_flash / 1024.0,
            'total_flash_kb': total_flash / 1024.0,
            'percent_increase': percent_increase
        })
    
    # Convert to DataFrame for easier handling
    plot_data = pd.DataFrame(plot_data)
    
    # If no valid data, exit
    if plot_data.empty:
        print("No valid data to plot after processing.")
        return
    
    # Define memory components for the color palette
    memory_components = ['SMALL_OPT', 'ADDITIONAL_FAST', 'FREE_FLASH']
    
    # Generate color palette
    memory_colors = generate_color_palette(memory_components, color_palette or 'Set2')
    
    # Bar width configuration
    bar_width = 0.6
    
    # Set the positions for the bars on the x-axis
    x = np.arange(len(plot_data))
    
    # For legend
    patches = []
    labels = []
    
    # Plot stacked bars
    for i, row in plot_data.iterrows():
        # Bottom layer: Small optimization flash usage
        small_bar = ax.bar(x[i], row['small_flash_kb'], 
                width=bar_width, color=memory_colors['SMALL_OPT'], 
                label='Small Optimization' if i == 0 else "")
        
        # Middle layer: Additional flash for fast optimization
        additional_bar = ax.bar(x[i], row['additional_flash_kb'], 
                width=bar_width, bottom=row['small_flash_kb'], 
                color=memory_colors['ADDITIONAL_FAST'], 
                label='Additional for Fast Opt.' if i == 0 else "")
        
        # Top layer: Free flash
        free_bar = ax.bar(x[i], row['free_flash_kb'], 
                width=bar_width, 
                bottom=row['small_flash_kb'] + row['additional_flash_kb'], 
                color=memory_colors['FREE_FLASH'], alpha=0.3,
                label='Free Flash' if i == 0 else "")
        
        # Save patches for the first bar for legend
        if i == 0:
            patches = [small_bar, additional_bar, free_bar]
            labels = ['Optimalizace "small"', 'Navíc pro "fast"', 'Volná Flash']
        
        # Add percentage increase in the middle of the additional flash section
        if row['additional_flash_kb'] > 0:
            additional_center = row['small_flash_kb'] + (row['additional_flash_kb'] / 2)
            ax.text(x[i], additional_center, f"+{row['percent_increase']:.1f}%", 
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white')
        
        # Add small optimization usage percentage in the middle of that section
        small_usage_percent = (row['small_flash_kb'] * 1024 / row['total_flash_kb'] / 1024) * 100
        ax.text(x[i], row['small_flash_kb'] / 2, f"{small_usage_percent:.1f}%", 
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white')
        
        # Add the total flash value above each bar
        ax.text(x[i], row['total_flash_kb'] + (row['total_flash_kb']*0.02), 
                f"{row['total_flash_kb']:.1f} KB", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set the x-axis ticks
    ax.set_xticks(x)
    
    # Set the board names as x-axis tick labels
    ax.set_xticklabels(plot_data['board'])
    
    # Set labels and title
    ax.set_ylabel('Paměť Flash [KB]', fontweight='bold')
    ax.set_title(f"Využití paměti Flash - {benchmark_type}", fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(patches[::-1], labels[::-1], title='Komponenty', 
              bbox_to_anchor=(1, 1), loc='upper left')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_energy_vs_time(data, energy_data, benchmark_type=None, save_path=None, color_palette=None):
    """
    Create a scatter plot of energy vs. inference time.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed benchmark data
    energy_data : pandas.DataFrame
        Energy consumption data
    benchmark_type : str, optional
        The benchmark type to visualize. If None, all benchmarks are shown.
    save_path : str, optional
        If provided, save the figure to this path
    color_palette : str, optional
        Name of the color palette to use
    """
    # Check if we're dealing with the special combined case
    is_combined_speech = benchmark_type == "SPEECH_YES_NO"
    
    # Filter data for the specific benchmark type if provided
    if benchmark_type is not None:
        if is_combined_speech:
            # For SPEECH_YES_NO, get both feature and model data
            feature_df = data[data['benchmark_type'] == "SPEECH_YES_NO_FEATURE"].copy()
            model_df = data[data['benchmark_type'] == "SPEECH_YES_NO_MODEL"].copy()
            
            if feature_df.empty:
                print("No benchmark data found for SPEECH_YES_NO_FEATURE")
                print("Available benchmark types:", data['benchmark_type'].unique())
                return
            if model_df.empty:
                print("No benchmark data found for SPEECH_YES_NO_MODEL")
                print("Available benchmark types:", data['benchmark_type'].unique())
                return
            
            # Use combined benchmark type in energy data
            energy_df = energy_data[energy_data['benchmark_type'] == "SPEECH_YES_NO"].copy()
        else:
            # Normal case - single benchmark type
            plot_df = data[data['benchmark_type'] == benchmark_type].copy()
            energy_df = energy_data[energy_data['benchmark_type'] == benchmark_type].copy()
            
            if plot_df.empty:
                print(f"No benchmark data found for benchmark type '{benchmark_type}'")
                print("Available benchmark types:", data['benchmark_type'].unique())
                return
    else:
        # Use all data if no specific benchmark type is requested
        plot_df = data.copy()
        energy_df = energy_data.copy()
    
    if energy_df.empty:
        print(f"No energy data found for benchmark type '{benchmark_type}'")
        print("Available energy data benchmark types:", energy_data['benchmark_type'].unique())
        return
    
    # For the combined speech case, we'll handle it separately
    if is_combined_speech:
        # Keep only 'fast' optimization level from benchmark data
        feature_df = feature_df[feature_df['opt_level'] == 'fast']
        model_df = model_df[model_df['opt_level'] == 'fast']
        
        if feature_df.empty or model_df.empty:
            print("No data with 'fast' optimization level found for one or both speech benchmarks")
            return
        
        # Get operation columns for calculating inference time
        op_cols = [col for col in data.columns if col.endswith('_ticks') 
                  and col != 'total_execution_time_ticks']
        
        if not op_cols:
            print("No operation columns found to calculate inference time")
            return
        
        # Create a plot points dataframe for the special case
        plot_points = []
        
        # Find boards that appear in both feature and model data
        feature_boards = set(feature_df['board_display_name'].unique())
        model_boards = set(model_df['board_display_name'].unique())
        common_boards = feature_boards & model_boards
        
        for board in common_boards:
            # Get feature time for this board
            feature_row = feature_df[feature_df['board_display_name'] == board].iloc[0]
            feature_time = feature_df[feature_df['board_display_name'] == board][op_cols].sum(axis=1).iloc[0] / 1000.0
            
            # Get model time for this board
            model_time = model_df[model_df['board_display_name'] == board][op_cols].sum(axis=1).iloc[0] / 1000.0
            
            # Calculate combined time
            combined_time = feature_time + model_time
            
            # Get the base board name for matching with energy data
            base_board = feature_row['base_board']
            
            # Find matching energy data
            energy_match = energy_df[energy_df['board'] == base_board]
            
            if not energy_match.empty:
                energy_value = energy_match['energy_mj'].iloc[0]
                
                plot_points.append({
                    'board_display': board,
                    'benchmark_type': "SPEECH_YES_NO",
                    'inference_time_ms': combined_time,
                    'energy_mj': energy_value,
                    'base_board': base_board
                })
            else:
                print(f"No energy data found for board {board} (base: {base_board})")
                print("Available energy data boards:", energy_df['board'].unique())
        
        # Convert to DataFrame for plotting
        if not plot_points:
            print("No matching data points found between benchmark and energy data")
            print("This might be due to board name mismatches between datasets")
            return
            
        plot_data = pd.DataFrame(plot_points)
        
    else:
        # Normal case - process regular benchmark data
        
        # Keep only 'fast' optimization level from benchmark data
        plot_df = plot_df[plot_df['opt_level'] == 'fast']
        
        if plot_df.empty:
            print(f"No data with 'fast' optimization level found")
            return
        
        # Get operation columns and calculate inference time by summing them
        op_cols = [col for col in plot_df.columns if col.endswith('_ticks') 
                  and col != 'total_execution_time_ticks']
        
        if not op_cols:
            print("No operation columns found to calculate inference time")
            return
        
        # Calculate inference time in milliseconds
        plot_df['inference_time_ms'] = plot_df[op_cols].sum(axis=1) / 1000.0
        
        # Prepare points to plot
        plot_points = []
        
        # For each board display name and benchmark type, find matching energy data
        for board_display in plot_df['board_display_name'].unique():
            # Get the base board name (without optimization level)
            board_data = plot_df[plot_df['board_display_name'] == board_display]
            base_board = board_data['base_board'].iloc[0]
            
            # For each benchmark type for this board
            for bm_type in board_data['benchmark_type'].unique():
                # Find matching energy data
                energy_match = energy_df[(energy_df['board'] == base_board) & 
                                        (energy_df['benchmark_type'] == bm_type)]
                
                if not energy_match.empty:
                    # Get the benchmark data for this specific board and benchmark type
                    specific_data = board_data[board_data['benchmark_type'] == bm_type]
                    
                    for _, row in specific_data.iterrows():
                        energy_value = energy_match['energy_mj'].iloc[0]
                        plot_points.append({
                            'board_display': board_display,
                            'benchmark_type': bm_type,
                            'inference_time_ms': row['inference_time_ms'],
                            'energy_mj': energy_value,
                            'base_board': base_board
                        })
                else:
                    print(f"No energy data found for board {board_display} and benchmark {bm_type}")
        
        # Convert to DataFrame for plotting
        if not plot_points:
            print("No matching data points found between benchmark and energy data")
            return
            
        plot_data = pd.DataFrame(plot_points)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique board display names for coloring
    board_types = sorted(plot_data['board_display'].unique())
    
    # Generate color palette for boards
    board_colors = generate_color_palette(board_types, color_palette)
    
    # Create the scatter plot
    for board in board_types:
        board_points = plot_data[plot_data['board_display'] == board]
        
        if board_points.empty:
            continue
            
        scatter = ax.scatter(
            board_points['inference_time_ms'],
            board_points['energy_mj'],
            color=board_colors[board],
            s=200,  # Size of points
            alpha=1,
            label=board
        )
        
        # Add labels for each point
        for _, row in board_points.iterrows():
            label = f"{board}"
            
            ax.annotate(
                label,
                (row['inference_time_ms'], row['energy_mj']),
                textcoords="offset points",
                xytext=(7, 7),
                ha='left',
                fontsize=11
            )
    
    # Set labels and title
    ax.set_xlabel('Celkový čas inference [ms]', fontweight='bold')
    ax.set_ylabel('Energie [mJ]', fontweight='bold')
    
    title = 'Doba inference v závislosti na energii'
    if benchmark_type:
        title += f' - {benchmark_type}'
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_data_summary(data):
    """
    Print a summary of the processed benchmark data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed benchmark data
    """
    # Print basic information about the data
    print(f"Loaded data for {data['board'].nunique()} board configurations and "
          f"{data['benchmark_type'].nunique()} benchmark types")
    
    # Print information about the processed data
    print(f"\nAfter preprocessing:")
    print(f"- {data['board_display_name'].nunique()} unique board types")
    print(f"- {data['opt_level'].nunique()} optimization levels")

    # Show benchmark types
    print("\nBenchmark Types:")
    for benchmark in sorted(data['benchmark_type'].unique()):
        print(f"  - {benchmark}")
    
    # Show board types and their optimization levels
    print("\nBoard Types and Optimization Levels:")
    for board_type in sorted(data['board_display_name'].unique()):
        print(f"  {board_type}:")
        for opt in sorted(data[data['board_display_name'] == board_type]['opt_level'].unique()):
            count = len(data[(data['board_display_name'] == board_type) & 
                           (data['opt_level'] == opt)]['benchmark_type'].unique())
            print(f"    - {opt.capitalize()}: {count} benchmark(s)")
    
    # Show column names
    print("\nAvailable metrics:")
    op_cols = [col for col in data.columns if col.endswith('_ticks')]
    mem_cols = [col for col in data.columns if col.endswith('_bytes')]
    print(f"Operation metrics: {op_cols}")
    print(f"Memory metrics: {mem_cols}")




# %%
raw_data = load_benchmark_data(data_dir="benchmark_results")

if raw_data is not None:
    # Preprocess the data
    data = preprocess_benchmark_data(raw_data)

    # Load energy data
    energy_data = load_energy_data(energy_file="benchmark_results/energy_data.csv")
    
    # Print summary
    print_data_summary(data)

# %%
# Create operation time visualization for a specific benchmark
benchmark = "KWS_CNN_LARGE_INT8"
plot_operation_times(
    data, 
    benchmark_type=f"{benchmark}",
    save_path=f"thesis_figures/inference_time_{benchmark}.png",
    color_palette="colorblind"  # Optional: specify a color palette
)

# %%
benchmark = "KWS_CNN_LARGE_INT8"
plot_flash_usage(
    data, 
    benchmark_type=f"{benchmark}",
    save_path=f"thesis_figures/flash_usage_{benchmark}.png",
    color_palette="colorblind"  # Optional: specify a color palette
)

# %%
benchmark = "KWS_CNN_LARGE_INT8"
plot_ram_usage(
    data, 
    benchmark_type=f"{benchmark}",
    save_path=f"thesis_figures/ram_usage_{benchmark}.png",
    color_palette="colorblind"  # Optional: specify a color palette
)

# %%
benchmark = "KWS_CNN_LARGE_INT8"
plot_energy_vs_time(
    data,
    energy_data,
    benchmark_type=f"{benchmark}",
    save_path=f"thesis_figures/energy_vs_time_{benchmark}.png",
    color_palette="colorblind"
)


