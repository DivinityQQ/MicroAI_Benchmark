# MicroAI Benchmark

A comprehensive benchmarking framework for TensorFlow Lite Micro (TFLM) on various microcontroller platforms.

## Overview

MicroAI Benchmark provides a standardized way to measure and compare the performance of machine learning models running on microcontroller devices. It supports multiple AI benchmarks and a wide range of popular microcontroller platforms. For best performance, TFLM with cmsis-nn or esp-nn is used respectively.

## Supported Benchmarks

- **Person Detection**: Visual person detection
- **Keyword Spotting**: Audio keyword recognition (Hey Siri, OK Google style)
- **Keyword Spotting Scrambled**: Alternative implementation of keyword recognition
- **Speech Yes/No Recognition**: Simple speech command recognition
- **Visual Wake Word Detection**: Vision-based wake word detection
- **Noise Reduction**: Audio noise reduction algorithms

## Supported Platforms

- ESP32 (various variants)
  - ESP32
  - ESP32-S3
  - ESP32-C6
  - ESP32-P4
- Raspberry Pi
  - Raspberry Pi Pico W
  - Raspberry Pi Pico 2
- Arduino
  - Arduino Nano 33 BLE
- Teensy
  - Teensy 4.0
- STM32
  - STM32 Nucleo L552ZE-Q
  - STM32 Nucleo F207ZG

## Build Variants

Each platform supports multiple build configurations:
- `small`: Optimized for code size
- `fast`: Optimized for performance

## Configuration

The benchmark is configured through the `src/config.h` file. You can:
1. Select the benchmark to run
2. Enable/disable profiling and logging
3. Configure benchmark-specific settings

### Selecting a Benchmark

To select a benchmark, edit `src/config.h` and uncomment exactly ONE of the benchmark definition lines:

```c
// Uncomment ONE benchmark to use
// #define USE_PERSON_DETECTION_BENCHMARK
// #define USE_SPEECH_YES_NO_BENCHMARK
// #define USE_KWS_SCRAMBLED_16_8_BENCHMARK
// #define USE_VISUAL_WAKEWORD_BENCHMARK
#define USE_KWS_BENCHMARK
// #define USE_NOISE_REDUCTION_BENCHMARK
```

### Benchmark-Specific Model Selection

Some benchmarks offer multiple model options. For example, for the Keyword Spotting benchmark:

```c
// KEYWORD SPOTTING BENCHMARK SETTINGS
// #define USE_CNN_SMALL_FLOAT32_MODEL
// #define USE_CNN_MEDIUM_FLOAT32_MODEL
// #define USE_CNN_SMALL_INT8_MODEL
...
// #define USE_DS_CNN_SMALL_INT16_MODEL
#define USE_MICRONET_SMALL_INT8_MODEL
// #define USE_MICRONET_MEDIUM_INT8_MODEL
```

### Profiling and Memory Monitoring

Enable or disable performance tracking features:

```c
// Whether to log and/or profile inference time data
// For measuring power consumption, not defining this is recommended
#define ENABLE_PROFILING
#define ENABLE_MEMORY_MONITORING
// #define ENABLE_LOGGING
```

## Usage

### Prerequisites

- [PlatformIO](https://platformio.org/)
- Appropriate hardware for the target platform

### Building and Running

1. Select the desired target environment in `platformio.ini` or use:
   ```
   pio run -e <environment_name>
   ```

2. Upload to your device:
   ```
   pio run -e <environment_name> -t upload
   ```

3. Monitor results:
   ```
   pio device monitor
   ```

## Benchmark Results

To store benchmark results, uncomment the extra scripts in `platformio.ini`:

```ini
[env]
framework = arduino
monitor_speed = 115200
extra_scripts = 
	scripts/memory_collector.py
	scripts/collect_benchmark.py
```

Results are stored in the `benchmark_results` directory. The framework measures:
- Inference time (in ticks/ms)
- Memory usage (RAM/Flash)
- Model-specific performance metrics

## Project Structure

- `src/`: Source code for benchmarks
- `lib/`: Libraries and dependencies
- `precompiled_lib/`: Precompiled TensorFlow Lite libraries for different targets
- `tflm_headers/`: TensorFlow Lite Micro headers
- `scripts/`: Benchmark collection and analysis scripts

## License

This project is licensed under the Apache 2.0 License. 

## Credits

This project uses slightly modified library [tflite-micro-arduino-examples](https://github.com/Gostas/tflite-micro-arduino-examples) for TFLM with CMSIS-NN optimized kernels and [ESP_TF](https://github.com/Nickjgniklu/ESP_TF) `create.sh` script for TFLM with esp-nn kernels.