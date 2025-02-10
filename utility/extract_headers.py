import os
import shutil

def extract_headers(source_dir, dest_dir):
    """
    Extract all .h header files from source_dir to dest_dir, 
    preserving the folder structure.

    :param source_dir: The root directory to search for header files.
    :param dest_dir: The destination directory to copy headers.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist!")
        return

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".h"):
                # Construct full source and destination paths
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_dir)
                dest_path = os.path.join(dest_dir, relative_path)

                # Create the destination directory if it doesn't exist
                os.makedirs(dest_path, exist_ok=True)

                # Copy the header file
                shutil.copy(source_path, os.path.join(dest_path, file))
                print(f"Copied: {source_path} -> {os.path.join(dest_path, file)}")

    print(f"\nAll header files have been copied to {dest_dir}.")

# Example usage:
# Replace these with your actual source and destination directories
source_directory = "c:/Users/Dusan/Documents/PlatformIO/Projects/MicroAI_Benchmark/lib/TensorFlow_Lite_ESP_NN/src/"
destination_directory = "c:/Temp/Python/header extractor/TensorFlow_Lite_ESP_NN"

extract_headers(source_directory, destination_directory)
