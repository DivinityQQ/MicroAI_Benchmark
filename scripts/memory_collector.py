# Memory usage collector for PlatformIO using regex parsing
# This script captures memory usage directly from PlatformIO's console output

Import("env")

import json
import os
import re
import sys

# Main output directory
OUTPUT_DIR = "benchmark_results"
# Memory info subfolder
MEM_INFO_DIR = os.path.join(OUTPUT_DIR, "mem_info")

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
    
    # Get the captured output
    output = sys.stdout.getvalue() if hasattr(sys.stdout, 'getvalue') else ''
    
    # Extract RAM and Flash usage using regex
    ram_pattern = r'RAM:\s+\[[=\s]+\]\s+(\d+\.\d+)%\s+\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)'
    flash_pattern = r'Flash:\s+\[[=\s]+\]\s+(\d+\.\d+)%\s+\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)'
    
    ram_match = re.search(ram_pattern, output)
    flash_match = re.search(flash_pattern, output)
    
    # Only proceed if we found the memory info
    if not ram_match or not flash_match:
        print("WARNING: Could not parse memory usage information from build output")
        print("No memory usage data will be available for benchmarking")
        return
    
    # Extract the values
    ram_used = int(ram_match.group(2))
    ram_total = int(ram_match.group(3))
    flash_used = int(flash_match.group(2))
    flash_total = int(flash_match.group(3))
    
    print(f"Memory usage: RAM={ram_used}/{ram_total} bytes, Flash={flash_used}/{flash_total} bytes")
    
    # Create a clean memory info structure with just the actual values
    memory_info = {
        "ram_used_bytes": ram_used,
        "ram_total_bytes": ram_total,
        "ram_percentage": round((ram_used / ram_total) * 100, 1) if ram_total > 0 else 0,
        "flash_used_bytes": flash_used,
        "flash_total_bytes": flash_total,
        "flash_percentage": round((flash_used / flash_total) * 100, 1) if flash_total > 0 else 0
    }
    
    # Save to a JSON file in mem_info subdirectory
    memory_file = os.path.join(mem_info_dir, f"memory_usage_{env_name}.json")
    with open(memory_file, 'w') as f:
        json.dump(memory_info, f, indent=2)
    
    print(f"Memory usage information saved to {memory_file}")

# Register just the size target hook - this is enough
env.AddPostAction("checkprogsize", on_size_target)