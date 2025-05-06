/* Copyright 2020-2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "config.h"
#ifdef USE_SPEECH_YES_NO_BENCHMARK

#include <Arduino.h>

#if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4)
// include main library header file
#include <TensorFlow_Lite_ESP_NN.h>
#else
// include main library header file
#include <TensorFlow_Lite_CMSIS_NN.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iterator>

#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#ifdef ENABLE_PROFILING
#include "tensorflow/lite/micro/micro_profiler.h"
#endif
#ifdef ENABLE_MEMORY_MONITORING
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#endif

#include "common/benchmark_utils.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
#ifdef ENABLE_PROFILING
tflite::MicroProfiler profiler;
#endif
const tflite::Model* model = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
#ifdef ENABLE_MEMORY_MONITORING
tflite::RecordingMicroInterpreter* interpreter = nullptr;
#else
tflite::MicroInterpreter* interpreter = nullptr;
#endif
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;

#if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(CONFIG_IDF_TARGET_ESP32P4)
constexpr int scratchBufSize = 4 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 + scratchBufSize;
alignas(16) uint8_t tensor_arena[kTensorArenaSize]; // Maybe we should move this to external

int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

// The name of this function is important for Arduino compatibility.
void speech_yes_no_setup() {

  // Enable serial only when profiling is enabled and you intend to connect the kit to PC,
  // on some boards it might hang otherwise
  #if defined(ENABLE_PROFILING) || defined(ENABLE_LOGGING) || defined(ARDUINO_ARCH_STM32)
  Serial.begin(115200);
  while(!Serial);
  #endif

  #ifdef CONFIG_IDF_TARGET_ESP32C6
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

  // turn the LED off by making the voltage LOW
  digitalWrite(LED_BUILTIN, LOW);
  #endif

  #ifdef ARDUINO_RASPBERRY_PI_PICO_W
  // digitalWrite(PIN_LED, HIGH);

  // Put the SMPS into PWM mode for better power consumption readibility
  // at the cost of higher idle power consumption
  digitalWrite(32+1, HIGH);
  #endif

  #ifdef ARDUINO_RASPBERRY_PI_PICO_2
  // digitalWrite(PIN_LED, HIGH);

  // Put the SMPS into PWM mode for better power consumption readibility
  // at the cost of higher idle power consumption
  pinMode(23, OUTPUT);
  digitalWrite(23, HIGH);
  #endif
  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  #if defined(ENABLE_MEMORY_MONITORING) && defined(ENABLE_PROFILING)
  // Both memory monitoring and profiling enabled
  static tflite::RecordingMicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
  #elif defined(ENABLE_MEMORY_MONITORING)
  // Only memory monitoring enabled
  static tflite::RecordingMicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  #elif defined(ENABLE_PROFILING)
  // Only profiling enabled
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler);
  #else
  // Neither memory monitoring nor profiling enabled
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  #endif
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureCount * kFeatureSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = tflite::GetTensorData<int8_t>(model_input);

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

}

// The name of this function is important for Arduino compatibility.
void speech_yes_no_loop() {
  // Fetch the spectrogram for the current time.
  const int32_t dummy_time = 0;
  int how_many_new_slices = 0;

  print_benchmark_start_custom("SPEECH_YES_NO_FEATURE");

  #ifdef ENABLE_LOGGING
  // Record time before inference
  unsigned long start_time_spectrogram = millis();
  #endif

  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      dummy_time, dummy_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }

  #ifdef ENABLE_LOGGING
  // Record time after inference
  unsigned long end_time_spectrogram = millis();

  // Calculate inference time
  unsigned long inference_time_spectrogram = end_time_spectrogram - start_time_spectrogram;

  TF_LITE_REPORT_ERROR(error_reporter, "Spectrogram builder inference time (ms): %d", inference_time_spectrogram);
  #endif

  print_benchmark_end();

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  print_benchmark_start_custom("SPEECH_YES_NO_MODEL");

  #ifdef ENABLE_PROFILING
  // Start profiling the inference event
  uint32_t event_handle = profiler.BeginEvent("Speech recognition invoke");
  #endif

  #ifdef ENABLE_LOGGING
  // Record time before inference
  unsigned long start_time_recognition = millis();
  #endif

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  #ifdef ENABLE_LOGGING
  // Record time after inference
  unsigned long end_time_recognition = millis();
  #endif

  #ifdef ENABLE_PROFILING
  // End profiling for this event
  profiler.EndEvent(event_handle);
  // Log the profiling data
  profiler.LogTicksPerTag();

  profiler.ClearEvents();
  #endif

  #ifdef ENABLE_MEMORY_MONITORING
  // Print out detailed allocation information:
  interpreter->GetMicroAllocator().PrintAllocations();
  #endif

  #ifdef ENABLE_LOGGING
  // Calculate inference time
  unsigned long inference_time_recognition = end_time_recognition - start_time_recognition;

  TF_LITE_REPORT_ERROR(error_reporter, "Speech recognition inference time (ms): %d", inference_time_recognition);

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);

  // using simple argmax instead of recognizer
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  int max_idx = 0;
  float max_result = 0.0;
  // Dequantize output values and find the max
  for (int i = 0; i < kCategoryCount; i++) {
    float current_result =
        (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
        output_scale;
    if (current_result > max_result) {
      max_result = current_result; // update max result
      max_idx = i; // update category
    }
  }
  if (max_result > 0.8f) {
    TF_LITE_REPORT_ERROR(error_reporter, "Detected %s, score: %f", kCategoryLabels[max_idx],
      static_cast<double>(max_result));
  } else {
    TF_LITE_REPORT_ERROR(error_reporter, "No detection");
  }
  #endif

  print_benchmark_end();
  
  delay(500);
}

#endif // USE_SPEECH_YES_NO_BENCHMARK