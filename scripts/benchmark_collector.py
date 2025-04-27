import serial
import serial.tools.list_ports
import csv
import re
import os
import time
import sys
from datetime import datetime

OUTPUT_DIR = "benchmark_results"  # Directory to store all CSVs
BAUD_RATE = 115200
RUNS_TO_SKIP = 4  # Skip the first 4 runs, collect the 5th

def get_available_ports():
    """Get a list of all currently available serial ports"""
    return [p.device for p in serial.tools.list_ports.comports()]

def find_board_port(upload_port=None):
    """
    Smart port detection function that handles board reconnection.
    
    Args:
        upload_port: The port used for uploading (may no longer be valid)
        
    Returns:
        The most likely port to use for communication
    """
    print_flush("Starting smart port detection...")
    
    # Get initial list of ports
    initial_ports = set(get_available_ports())
    print_flush(f"Initial ports: {', '.join(initial_ports) if initial_ports else 'None'}")
    
    # Try the provided upload port first if it exists
    if upload_port and upload_port in initial_ports:
        print_flush(f"Upload port {upload_port} is available, trying it first")
        return upload_port
    
    # If the upload port isn't available or none was provided, look for typical MCU ports
    for port in initial_ports:
        for p in serial.tools.list_ports.comports():
            if p.device == port:
                # Common patterns for MCU ports
                if any(x in p.description.lower() for x in ["cp210", "ch340", "ftdi", "usb-serial", "ttyacm", "ttyusb"]):
                    print_flush(f"Found likely MCU port: {port} ({p.description})")
                    return port
    
    # Wait a moment for port re-enumeration to complete
    print_flush("No port identified yet. Waiting for board re-enumeration...")
    time.sleep(2)  # Give the board time to re-enumerate
    
    # Get list of ports after possible re-enumeration
    current_ports = set(get_available_ports())
    print_flush(f"Current ports after delay: {', '.join(current_ports) if current_ports else 'None'}")
    
    # Look for new ports that weren't in the initial list
    new_ports = current_ports - initial_ports
    if new_ports:
        print_flush(f"Found new ports after re-enumeration: {', '.join(new_ports)}")
        return list(new_ports)[0]  # Return the first new port
    
    # If no new ports, but we have existing ports, return the first one
    if current_ports:
        print_flush(f"No new ports found. Using first available port: {list(current_ports)[0]}")
        return list(current_ports)[0]
    
    # No ports found at all
    print_flush("No serial ports found!")
    return None

def connect_with_retry(port, baud_rate, max_attempts=10, delay_seconds=1):
    """
    Attempt to connect to the serial port with retries.
    
    Args:
        port: Serial port to connect to
        baud_rate: Baud rate for the connection
        max_attempts: Maximum number of connection attempts
        delay_seconds: Delay between attempts in seconds
        
    Returns:
        Serial connection object or None if connection failed
    """
    print_flush(f"Attempting to connect to {port}...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            connection = serial.Serial(port, baud_rate, timeout=1)
            print_flush(f"Connected to {port} on attempt {attempt}/{max_attempts}")
            return connection
        except (serial.SerialException, FileNotFoundError) as e:
            print_flush(f"Attempt {attempt}/{max_attempts} failed: {str(e)}")
            
            if attempt < max_attempts:
                print_flush(f"Waiting {delay_seconds} second(s) before next attempt...")
                time.sleep(delay_seconds)
            else:
                print_flush(f"Failed to connect after {max_attempts} attempts.")
                # Try alternate ports as a last resort
                alt_ports = get_available_ports()
                if alt_ports and (port not in alt_ports or len(alt_ports) > 1):
                    if port in alt_ports:
                        alt_ports.remove(port)  # Remove already failed port
                    alt_port = alt_ports[0]
                    print_flush(f"Trying alternate port: {alt_port}")
                    try:
                        connection = serial.Serial(alt_port, baud_rate, timeout=1)
                        print_flush(f"Connected to alternate port {alt_port}")
                        return connection
                    except (serial.SerialException, FileNotFoundError) as e2:
                        print_flush(f"Failed to connect to alternate port: {str(e2)}")
                return None

def print_flush(message):
    """Print message and flush immediately to ensure real-time output"""
    print(message)
    sys.stdout.flush()  # Force output to appear immediately

def main():
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Check if environment name and port were passed as arguments
        if len(sys.argv) > 2:
            mcu_type = sys.argv[1]
            provided_port = sys.argv[2]
            print_flush(f"Using environment name from PlatformIO: {mcu_type}")
            print_flush(f"Using upload port from PlatformIO: {provided_port}")
        else:
            mcu_type = sys.argv[1] if len(sys.argv) > 1 else "unknown_board"
            provided_port = None
            print_flush(f"Using environment name: {mcu_type}")
            print_flush("No upload port provided, will auto-detect")
            
        # Get port using smart detection
        port = find_board_port(provided_port)
        
        if not port:
            print_flush("Could not find any serial ports. Aborting.")
            return
            
        print_flush(f"Using port: {port}")
        
        # Try to connect with retry mechanism
        ser = connect_with_retry(port, BAUD_RATE, max_attempts=10, delay_seconds=1)
        
        if not ser:
            print_flush("Could not establish serial connection. Aborting.")
            return
            
        print_flush("Connected to serial port. Waiting for benchmark data...")
        
        # Initialize data structures
        benchmark_data = {}
        operation_times = {}
        memory_stats = {}
        benchmark_runs = {}  # Track runs for each benchmark type separately
        collecting = False
        benchmark_type = None
        completed_benchmarks = set()  # Track which benchmarks have been completed
        
        # Main collection loop
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except serial.SerialException as e:
                print_flush(f"Serial read error: {e}")
                print_flush("Connection may have been lost. Attempting to reconnect...")
                ser.close()
                ser = connect_with_retry(port, BAUD_RATE)
                if not ser:
                    print_flush("Failed to reconnect. Aborting.")
                    return
                continue

            if not line:
                continue
                
            # Check for start marker
            if line.startswith("BENCHMARK_START:"):
                collecting = True
                benchmark_type = line.split(":", 1)[1]
                operation_times = {}
                memory_stats = {}
                
                # Initialize run counter for this benchmark type if needed
                if benchmark_type not in benchmark_runs:
                    benchmark_runs[benchmark_type] = 0
                    
                # Increment run counter for this specific benchmark type
                benchmark_runs[benchmark_type] += 1
                
                print_flush(f"Detected benchmark run #{benchmark_runs[benchmark_type]} for: {benchmark_type}")
                continue
                
            # Check for end marker
            if line == "BENCHMARK_END" and collecting:
                collecting = False
                
                # Skip data collection for early runs
                if benchmark_runs[benchmark_type] <= RUNS_TO_SKIP:
                    print_flush(f"Skipping run #{benchmark_runs[benchmark_type]} for {benchmark_type} (collecting run #{RUNS_TO_SKIP+1})")
                    continue
                
                # If this is the run we want to capture
                if benchmark_runs[benchmark_type] == RUNS_TO_SKIP + 1:
                    # Create complete dataset
                    benchmark_data = {
                        "timestamp": datetime.now().isoformat(),
                        "board": mcu_type,
                        "benchmark_type": benchmark_type,
                        **operation_times,
                        **memory_stats
                    }
                    
                    # Create filename for this board/benchmark combination
                    safe_mcu = str(mcu_type).replace(" ", "_").replace("/", "_")
                    safe_benchmark = str(benchmark_type).replace(" ", "_").replace("/", "_")
                    filename = f"{OUTPUT_DIR}/benchmark_{safe_mcu}_{safe_benchmark}.csv"
                    
                    # Check if file exists to determine if we need to write headers
                    file_exists = os.path.isfile(filename)
                    
                    # Write to CSV
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=list(benchmark_data.keys()))
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(benchmark_data)
                    
                    # Create a human-readable summary file
                    summary_filename = f"{OUTPUT_DIR}/summary_{safe_mcu}_{safe_benchmark}.txt"
                    with open(summary_filename, 'w') as f:
                        f.write(f"Benchmark Summary\n")
                        f.write(f"================\n\n")
                        f.write(f"Board: {mcu_type}\n")
                        f.write(f"Benchmark: {benchmark_type}\n")
                        f.write(f"Timestamp: {benchmark_data['timestamp']}\n\n")
                        
                        f.write(f"Execution Times (ticks):\n")
                        f.write(f"----------------------\n")
                        # Group all the time-related entries
                        time_entries = {k: v for k, v in benchmark_data.items() if k.endswith("_ticks")}
                        # Sort by name but prioritize "total_execution_time_ticks" to be first
                        sorted_times = sorted(time_entries.items(), 
                                             key=lambda x: (0 if x[0] == "total_execution_time_ticks" else 1, x[0]))
                        for key, value in sorted_times:
                            # Keep original operation names
                            original_key = key.replace("_ticks", "").upper()
                            f.write(f"{original_key}: {value}\n")
                        
                        f.write(f"\nMemory Usage (bytes):\n")
                        f.write(f"-------------------\n")
                        memory_entries = {k: v for k, v in benchmark_data.items() if k.endswith("_bytes")}
                        # Sort by name but prioritize arena total/head/tail to be first
                        sorted_memory = sorted(memory_entries.items(), 
                                              key=lambda x: (0 if "arena_total" in x[0] else 
                                                             1 if "arena_head" in x[0] else
                                                             2 if "arena_tail" in x[0] else 3, x[0]))
                        for key, value in sorted_memory:
                            # Keep original format but make it readable
                            readable_key = key.replace("_bytes", "")
                            readable_key = readable_key.replace("_", " ").upper()
                            f.write(f"{readable_key}: {value}\n")
                    
                    print_flush(f"Benchmark data saved to {filename}")
                    print_flush(f"Human-readable summary saved to {summary_filename}")
                    
                    # Mark this benchmark as completed
                    completed_benchmarks.add(benchmark_type)
                    
                    # Check if we've completed all detected benchmark types
                    if completed_benchmarks and len(completed_benchmarks) == len(benchmark_runs):
                        print_flush("All detected benchmark types have been collected!")
                        print_flush(f"Collected data for: {', '.join(completed_benchmarks)}")
                        print_flush("Data collection complete!")
                        return
                continue
            
            # Only process lines if we're in collection mode
            if not collecting:
                continue
                
            # Skip processing for early runs to save resources
            if benchmark_runs.get(benchmark_type, 0) <= RUNS_TO_SKIP:
                continue
                
            # Handle benchmark invoke time (only ticks)
            if "invoke took" in line:
                match = re.search(r'([A-Za-z_\s]+) invoke took (\d+) ticks', line)
                if match:
                    benchmark_name = match.group(1).strip().lower().replace(" ", "_")
                    ticks = match.group(2)
                    operation_times["total_execution_time_ticks"] = ticks
            
            # Handle operation times (only ticks)
            match = re.search(r'([A-Z_0-9]+) took (\d+) ticks', line)
            if match:
                operation = match.group(1).lower()
                ticks = match.group(2)
                operation_times[f"{operation}_ticks"] = ticks
            
            # Handle memory allocation
            match = re.search(r'Arena allocation (\w+) (\d+) bytes', line)
            if match:
                allocation_type = match.group(1).lower()
                bytes_value = match.group(2)
                memory_stats[f"arena_{allocation_type}_bytes"] = bytes_value
            
            # Handle other memory stats
            match = re.search(r"'([^']+)' used (\d+) bytes", line)
            if match:
                resource_name = match.group(1).lower().replace(" ", "_").replace("'", "")
                bytes_value = match.group(2)
                memory_stats[f"{resource_name}_bytes"] = bytes_value
    
    except KeyboardInterrupt:
        print_flush("\nMonitoring stopped by user")
    except Exception as e:
        print_flush(f"Error: {e}")
    finally:
        if 'ser' in locals() and ser and ser.is_open:
            ser.close()
            print_flush("Serial connection closed")

if __name__ == "__main__":
    main()