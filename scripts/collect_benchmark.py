Import("env")

def after_upload(source, target, env):
    import subprocess
    import sys
    import os
    
    # Get the Python interpreter path
    python_path = sys.executable
    
    # Get the project directory (where PlatformIO is running from)
    project_dir = os.getcwd()
    
    # Get environment name directly from PlatformIO
    env_name = env.subst("$PIOENV")
    
    # Get upload port directly from PlatformIO
    upload_port = env.subst("$UPLOAD_PORT")
    
    # Define the path to our collection script
    collector_script = os.path.join(project_dir, "scripts", "benchmark_collector.py")
    
    print(f"Starting benchmark data collection for environment: {env_name}")
    print(f"Using upload port: {upload_port}")
    print(f"Looking for collector script at: {collector_script}")
    
    # Check if the script exists
    if not os.path.exists(collector_script):
        print(f"ERROR: Collector script not found at {collector_script}")
        print("Make sure you've created the scripts/benchmark_collector.py file")
        return
        
    print(f"Collector script found. Using Python: {python_path}")
    
    # Run the collection script with the environment name and upload port as arguments
    try:
        # For better visibility, run synchronously
        print("Running benchmark collector... (this might take a minute)")
        subprocess.run([python_path, collector_script, env_name, upload_port], check=True)
        print("Benchmark collector completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error: Collector script exited with code {e.returncode}")
    except Exception as e:
        print(f"Error starting collector: {e}")
    

# Register the callback
env.AddPostAction("upload", after_upload)