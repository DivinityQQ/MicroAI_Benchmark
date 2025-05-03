Import("env")

import json
import os
import re
import sys

# Main output directory
OUTPUT_DIR = "benchmark_results"
# Memory info subfolder
MEM_INFO_DIR = os.path.join(OUTPUT_DIR, "mem_info")
# Teensy memory info file
TEENSY_MEM_INFO_FILE = os.path.join(OUTPUT_DIR, "teensy_mem_info.txt")

# Enable output capturing
class OutputCapture:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = []
    
    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.append(text)
    
    def flush(self):
        self.original_stdout.flush()
    
    def getvalue(self):
        return ''.join(self.buffer)

# Start capturing output
original_stdout = sys.stdout
sys.stdout = OutputCapture(original_stdout)

def is_teensy_board(env_name):
    """Check if the environment name indicates a Teensy board"""
    return "teensy" in env_name.lower()

def parse_teensy_memory_file():
    """Parse Teensy memory info from the manually created file"""
    if not os.path.exists(TEENSY_MEM_INFO_FILE):
        print(f"WARNING: Teensy memory info file not found at {TEENSY_MEM_INFO_FILE}")
        return None
    
    try:
        # Read the memory info file
        with open(TEENSY_MEM_INFO_FILE, 'r') as f:
            content = f.read()
        
        # Parse FLASH information
        flash_match = re.search(r'FLASH: code:(\d+), data:(\d+), headers:(\d+)\s+free for files:(\d+)', content)
        ram1_match = re.search(r'RAM1: variables:(\d+), code:(\d+), padding:(\d+)\s+free for local variables:(\d+)', content)
        ram2_match = re.search(r'RAM2: variables:(\d+)\s+free for malloc/new:(\d+)', content)
        
        if not flash_match or not ram1_match or not ram2_match:
            print("WARNING: Could not parse Teensy memory info file - format may be incorrect")
            return None
        
        # Extract FLASH values
        flash_code = int(flash_match.group(1))
        flash_data = int(flash_match.group(2))
        flash_headers = int(flash_match.group(3))
        flash_free = int(flash_match.group(4))
        
        # Extract RAM1 values
        ram1_variables = int(ram1_match.group(1))
        ram1_code = int(ram1_match.group(2))
        ram1_padding = int(ram1_match.group(3))
        ram1_free = int(ram1_match.group(4))
        
        # Extract RAM2 values
        ram2_variables = int(ram2_match.group(1))
        ram2_free = int(ram2_match.group(2))
        
        # Calculate total values according to user's specifications
        flash_used = flash_code + flash_data + flash_headers
        flash_total = flash_used + flash_free
        
        ram_used = ram1_variables + ram1_code + ram1_padding + ram2_variables
        ram_total = ram_used + ram1_free + ram2_free
        
        # Calculate percentages
        flash_percentage = round((flash_used / flash_total) * 100, 1) if flash_total > 0 else 0
        ram_percentage = round((ram_used / ram_total) * 100, 1) if ram_total > 0 else 0
        
        # Create memory info structure
        memory_info = {
            "ram_used_bytes": ram_used,
            "ram_total_bytes": ram_total,
            "ram_percentage": ram_percentage,
            "flash_used_bytes": flash_used,
            "flash_total_bytes": flash_total,
            "flash_percentage": flash_percentage
        }
        
        print(f"Successfully parsed Teensy memory info from {TEENSY_MEM_INFO_FILE}")
        print(f"RAM: {ram_used}/{ram_total} bytes ({ram_percentage}%)")
        print(f"Flash: {flash_used}/{flash_total} bytes ({flash_percentage}%)")
        
        return memory_info
        
    except Exception as e:
        print(f"ERROR parsing Teensy memory info file: {str(e)}")
        return None

def parse_standard_memory_output(output):
    """Parse RAM and Flash usage from standard PlatformIO output"""
    # Extract RAM and Flash usage using regex
    ram_pattern = r'RAM:\s+\[[=\s]+\]\s+(\d+\.\d+)%\s+\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)'
    flash_pattern = r'Flash:\s+\[[=\s]+\]\s+(\d+\.\d+)%\s+\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)'
    
    ram_match = re.search(ram_pattern, output)
    flash_match = re.search(flash_pattern, output)
    
    # Only proceed if we found the memory info
    if not ram_match or not flash_match:
        return None
    
    # Extract the values
    ram_used = int(ram_match.group(2))
    ram_total = int(ram_match.group(3))
    flash_used = int(flash_match.group(2))
    flash_total = int(flash_match.group(3))
    
    # Create memory info structure
    memory_info = {
        "ram_used_bytes": ram_used,
        "ram_total_bytes": ram_total,
        "ram_percentage": round((ram_used / ram_total) * 100, 1) if ram_total > 0 else 0,
        "flash_used_bytes": flash_used,
        "flash_total_bytes": flash_total,
        "flash_percentage": round((flash_used / flash_total) * 100, 1) if flash_total > 0 else 0
    }
    
    return memory_info

def on_size_target(source, target, env):
    """Hook that runs when the size target is executed"""
    print("Collecting memory usage information...")
    
    # Get project directory path
    project_dir = env.subst("$PROJECT_DIR")
    env_name = env.subst("$PIOENV")
    
    # Create output directories if they don't exist
    output_dir = os.path.join(project_dir, OUTPUT_DIR)
    mem_info_dir = os.path.join(project_dir, MEM_INFO_DIR)
    
    for directory in [output_dir, mem_info_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Check if this is a Teensy board
    if is_teensy_board(env_name):
        print(f"Detected Teensy board environment: {env_name}")
        memory_info = parse_teensy_memory_file()
        
        # If no Teensy memory file found, warn but don't fail
        if memory_info is None:
            print("No Teensy memory info file found. If you're building for Teensy, please:")
            print(f"1. Copy the memory usage information to: {TEENSY_MEM_INFO_FILE}")
            print("2. Format should match the 'teensy_size:' output lines")
            print("3. Run the upload or size task again to capture the memory info")
            return
    else:
        # Get the captured output for standard boards
        output = sys.stdout.getvalue() if hasattr(sys.stdout, 'getvalue') else ''
        memory_info = parse_standard_memory_output(output)
        
        # Only proceed if we found the memory info
        if memory_info is None:
            print("WARNING: Could not parse memory usage information from build output")
            print("No memory usage data will be available for benchmarking")
            return
    
    # Print the collected memory info for debugging
    print(f"Memory usage: RAM={memory_info['ram_used_bytes']}/{memory_info['ram_total_bytes']} bytes, Flash={memory_info['flash_used_bytes']}/{memory_info['flash_total_bytes']} bytes")
    
    # Save to a JSON file in mem_info subdirectory
    memory_file = os.path.join(mem_info_dir, f"memory_usage_{env_name}.json")
    with open(memory_file, 'w') as f:
        json.dump(memory_info, f, indent=2)
    
    print(f"Memory usage information saved to {memory_file}")

# Register just the size target hook - this is enough
env.AddPostAction("checkprogsize", on_size_target)