import os
import re

# Input C array file (e.g., "model.cc" or "model.h")
input_file = "audio_preprocessor_int8_model_data.h"
# Output .tflite file
output_file = "recovered_model.tflite"

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: '{input_file}' not found in directory:\n{os.listdir()}")
    exit(1)

# Read the C array content
with open(input_file, "r") as f:
    content = f.read()

# Extract all hex values (e.g., 0x00, 0x1c) using regex
hex_values = re.findall(r"0x[0-9a-fA-F]{2}", content)

# Convert hex strings to bytes
bytes_data = bytes([int(x, 16) for x in hex_values])

# Write bytes to .tflite file
with open(output_file, "wb") as f:
    f.write(bytes_data)

print(f"Generated {output_file} with {len(bytes_data)} bytes.")