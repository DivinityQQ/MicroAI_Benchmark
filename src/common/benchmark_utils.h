// benchmark_utils.h
#pragma once

#include "config.h"
#include <Arduino.h>

// Function to get benchmark type string based on config.h defines
const char* get_benchmark_type() {
  // First determine the main benchmark type
  #ifdef USE_PERSON_DETECTION_BENCHMARK
    return "PERSON_DETECTION";
  #elif defined(USE_SPEECH_YES_NO_BENCHMARK)
    return "SPEECH_YES_NO";
  #elif defined(USE_KWS_SCRAMBLED_16_8_BENCHMARK)
    #ifdef USE_8BIT_MODEL
      return "KWS_SCRAMBLED_8BIT";
    #elif defined(USE_STANDARD_MODEL)
      return "KWS_SCRAMBLED_STANDARD";
    #else
      return "KWS_SCRAMBLED";
    #endif
  #elif defined(USE_VISUAL_WAKEWORD_BENCHMARK)
    #ifdef USE_128x128x1_MODEL
      return "VWW_128x128x1";
    #elif defined(USE_96x96x3_MODEL)
      return "VWW_96x96x3";
    #else
      return "VISUAL_WAKEWORD";
    #endif
  #elif defined(USE_KWS_BENCHMARK)
    // For KWS, also include the model type
    #ifdef USE_CNN_SMALL_FLOAT32_MODEL
      return "KWS_CNN_SMALL_FLOAT32";
    #elif defined(USE_CNN_MEDIUM_FLOAT32_MODEL)
      return "KWS_CNN_MEDIUM_FLOAT32";
    #elif defined(USE_CNN_SMALL_INT8_MODEL)
      return "KWS_CNN_SMALL_INT8";
    #elif defined(USE_CNN_MEDIUM_INT8_MODEL)
      return "KWS_CNN_MEDIUM_INT8";
    #elif defined(USE_CNN_LARGE_INT8_MODEL)
      return "KWS_CNN_LARGE_INT8";
    #elif defined(USE_DNN_SMALL_INT8_MODEL)
      return "KWS_DNN_SMALL_INT8";
    #elif defined(USE_DNN_MEDIUM_INT8_MODEL)
      return "KWS_DNN_MEDIUM_INT8";
    #elif defined(USE_DNN_LARGE_INT8_MODEL)
      return "KWS_DNN_LARGE_INT8";
    #elif defined(USE_DNN_SMALL_FLOAT32_MODEL)
      return "KWS_DNN_SMALL_FLOAT32";
    #elif defined(USE_DNN_MEDIUM_FLOAT32_MODEL)
      return "KWS_DNN_MEDIUM_FLOAT32";
    #elif defined(USE_DS_CNN_SMALL_INT8_MODEL)
      return "KWS_DS_CNN_SMALL_INT8";
    #elif defined(USE_DS_CNN_MEDIUM_INT8_MODEL)
      return "KWS_DS_CNN_MEDIUM_INT8";
    #elif defined(USE_DS_CNN_LARGE_INT8_MODEL)
      return "KWS_DS_CNN_LARGE_INT8";
    #elif defined(USE_DS_CNN_SMALL_INT16_MODEL)
      return "KWS_DS_CNN_SMALL_INT16";
    #elif defined(USE_DS_CNN_SMALL_FLOAT32_MODEL)
      return "KWS_DS_CNN_SMALL_FLOAT32";
    #else
      return "KWS";
    #endif
  #elif defined(USE_NOISE_REDUCTION_BENCHMARK)
    return "NOISE_REDUCTION";
  #else
    return "UNKNOWN_BENCHMARK";
  #endif
}

// Helper to print benchmark start marker
void print_benchmark_start() {
  #if defined(ENABLE_PROFILING) || defined(ENABLE_LOGGING)
  Serial.print("BENCHMARK_START:");
  Serial.println(get_benchmark_type());
  #endif
}

// Helper to print custom benchmark start marker
void print_benchmark_start_custom(const char* benchmark_name) {
  #if defined(ENABLE_PROFILING) || defined(ENABLE_LOGGING)
  Serial.print("BENCHMARK_START:");
  Serial.println(benchmark_name);
  #endif
}

// Helper to print benchmark end marker
void print_benchmark_end() {
  #if defined(ENABLE_PROFILING) || defined(ENABLE_LOGGING)
  Serial.println("BENCHMARK_END");
  #endif
}